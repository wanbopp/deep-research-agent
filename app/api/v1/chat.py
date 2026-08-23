"""Chat API endpoints for regular JSON responses and SSE streams."""

from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from app.api.dependencies import get_chat_service
from app.api.sse import encode_sse_event
from app.schemas.chat import (
    ChatAPIResponse,
    ChatInterruptResponse,
    ChatRequest,
    ChatResumeRequest,
)
from app.services.chat import ChatInterrupt, ChatService, ChatTurnResult

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


@router.post("", response_model=ChatAPIResponse)
async def create_chat_turn(
    request: ChatRequest,
    service: ChatServiceDependency,
) -> ChatAPIResponse:
    """开始或继续一轮普通聊天."""
    result = await service.run_turn(request)
    return _to_api_response(result)


@router.post("/resume", response_model=ChatAPIResponse)
async def resume_chat_turn(
    request: ChatResumeRequest,
    service: ChatServiceDependency,
) -> ChatAPIResponse:
    """使用人工回答恢复暂停的 Agent."""
    resume = await service.resume_turn(request)
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
    },
)
async def stream_chat_turn(
    request: ChatRequest,
    service: ChatServiceDependency,
) -> StreamingResponse:
    """以 Server-Sent Events 持续输出一轮 Agent 事件."""

    async def event_generator() -> AsyncIterator[str]:
        # StreamingResponse 会逐个请求生成器中的内容。
        # 客户端断开时，Starlette 会取消当前生成器任务。
        #
        # 这里不捕获 CancelledError，让取消继续进入 ChatService，
        # 这样 LangGraph 和正在进行的模型请求才能停止，而不是在后台继续消耗资源。
        # 不要手动添加 "Connection": "keep-alive"
        async for event in service.stream_turn(request):
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
