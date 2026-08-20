"""Non-streaming chat API endpoints."""

from typing import Annotated

from fastapi import APIRouter, Depends
from app.api.dependencies import get_chat_service
from app.schemas.chat import (
    ChatInterruptResponse,
    ChatRequest,
    ChatResponse,
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

ChatAPIResponse = ChatResponse | ChatInterruptResponse


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
