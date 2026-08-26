"""Chat API endpoints for regular JSON responses and SSE streams."""

from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse

from app.api.dependencies import (
    AgentRateLimitDependency,
    CurrentUserDependency,
    get_chat_service,
)
from app.api.sse import encode_sse_event
from app.schemas.base import ErrorResponse
from app.schemas.chat import (
    ChatAPIResponse,
    ChatInterruptResponse,
    ChatRequest,
    ChatResumeRequest,
)
from app.services.chat import (
    ChatInterrupt,
    ChatResumeNotAvailableError,
    ChatService,
    ChatTurnResult,
)
from app.services.chat_session_ownership import ChatSessionNotFoundError

router = APIRouter(prefix="/chat", tags=["chat"])

# 声明这个类型是 ChatService,其别名后续只有通过别名注入
ChatServiceDependency = Annotated[
    ChatService,
    Depends(
        get_chat_service
    ),  # FastAPI的依赖注入声明，告诉FastAPI在路由处理时自动调用 用 get_chat_service 来获取 ChatService 实例
]


def _to_api_response(result: ChatTurnResult) -> ChatAPIResponse:
    """把应用层结果转换为稳定的公开 API 响应."""
    if isinstance(result, ChatInterrupt):
        return ChatInterruptResponse(
            status="interrupted",
            thread_id=result.thread_id,
            question=result.question,
        )
    else:
        return result


@router.post(
    "",
    response_model=ChatAPIResponse,
    responses={
        status.HTTP_401_UNAUTHORIZED: {
            "model": ErrorResponse,
            "description": "Bearer token is missing or invalid",
        },
        status.HTTP_404_NOT_FOUND: {
            "model": ErrorResponse,
            "description": "Chat session is absent from the current user scope",
        },
        status.HTTP_409_CONFLICT: {
            "model": ErrorResponse,
            "description": "The chat thread is already being processed",
        },
        status.HTTP_429_TOO_MANY_REQUESTS: {
            "model": ErrorResponse,
            "description": "Agent request rate limit exceeded",
        },
        status.HTTP_503_SERVICE_UNAVAILABLE: {
            "model": ErrorResponse,
            "description": "Chat execution guard or request rate limiter is unavailable",
        },
    },
)
async def create_chat_turn(
    request: ChatRequest,
    service: ChatServiceDependency,
    current_user: CurrentUserDependency,
    _rate_limit: AgentRateLimitDependency,
) -> ChatAPIResponse:
    """使用服务端认证身份开始或继续一轮普通聊天."""
    try:
        result = await service.run_turn(
            request,
            user_id=current_user.user_id,
        )
    except ChatSessionNotFoundError:
        # UUID 不存在和属于其他用户使用相同 404，不形成资源枚举接口。
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Chat session was not found",
        ) from None
    return _to_api_response(result)


@router.post(
    "/resume",
    response_model=ChatAPIResponse,
    responses={
        status.HTTP_401_UNAUTHORIZED: {
            "model": ErrorResponse,
            "description": "Bearer token is missing or invalid",
        },
        status.HTTP_404_NOT_FOUND: {
            "model": ErrorResponse,
            "description": "Chat thread is not available for resume",
        },
        status.HTTP_409_CONFLICT: {
            "model": ErrorResponse,
            "description": "The chat thread is already being processed",
        },
        status.HTTP_429_TOO_MANY_REQUESTS: {
            "model": ErrorResponse,
            "description": "Agent request rate limit exceeded",
        },
        status.HTTP_503_SERVICE_UNAVAILABLE: {
            "model": ErrorResponse,
            "description": "Chat execution guard or request rate limiter is unavailable",
        },
    },
)
async def resume_chat_turn(
    request: ChatResumeRequest,
    service: ChatServiceDependency,
    current_user: CurrentUserDependency,
    _rate_limit: AgentRateLimitDependency,
) -> ChatAPIResponse:
    """只在当前用户自己的 checkpoint 空间恢复暂停的 Agent."""
    try:
        resume = await service.resume_turn(
            request,
            user_id=current_user.user_id,
        )
    except ChatSessionNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Chat session was not found",
        ) from None
    except ChatResumeNotAvailableError:
        # 不区分“当前用户没有该 thread”与“同名 thread 属于其他用户”，避免
        # 攻击者通过状态码或文案探测另一个用户的 Agent 会话是否存在。
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Chat thread is not available for resume",
        ) from None
    return _to_api_response(resume)


@router.post(
    "/stream",
    response_class=StreamingResponse,
    responses={
        200: {
            "description": "Server-Sent Events stream",
            "content": {
                "text/event-stream": {},
            },
        },
        status.HTTP_401_UNAUTHORIZED: {
            "model": ErrorResponse,
            "description": "Bearer token is missing or invalid",
        },
        status.HTTP_429_TOO_MANY_REQUESTS: {
            "model": ErrorResponse,
            "description": "Agent request rate limit exceeded",
        },
        status.HTTP_503_SERVICE_UNAVAILABLE: {
            "model": ErrorResponse,
            "description": "Request rate limiter is unavailable",
        },
    },
)
async def stream_chat_turn(
    request: ChatRequest,
    service: ChatServiceDependency,
    current_user: CurrentUserDependency,
    _rate_limit: AgentRateLimitDependency,
) -> StreamingResponse:
    """以 Server-Sent Events 持续输出一轮 Agent 事件."""

    async def event_generator() -> AsyncIterator[str]:
        # StreamingResponse 会逐个请求生成器中的内容。
        # 客户端断开时，Starlette 会取消当前生成器任务。
        #
        # 这里不捕获 CancelledError，让取消继续进入 ChatService，
        # 这样 LangGraph 和正在进行的模型请求才能停止，而不是在后台继续消耗资源。
        # 不要手动添加 "Connection": "keep-alive"
        async for event in service.stream_turn(
            request,
            user_id=current_user.user_id,
        ):
            yield encode_sse_event(event)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            # SSE 不应该被浏览器或中间缓存保存，否则客户端可能无法立即看到事件。
            "Cache-Control": "no-cache",
            # 告诉 Nginx 等反向代理不要缓存一批事件后再一次性转发。
            "X-Accel-Buffering": "no",
        },
    )
