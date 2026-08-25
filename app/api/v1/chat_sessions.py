"""业务聊天会话的创建、列表与所有权查询 HTTP 入口."""

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status

from app.api.dependencies import (
    CurrentUserDependency,
    get_chat_session_service,
)
from app.schemas.base import ErrorResponse
from app.schemas.chat_session import (
    ChatSessionCreateRequest,
    ChatSessionListResponse,
    ChatSessionResponse,
)
from app.services.chat_sessions import (
    ChatSessionNotFoundError,
    ChatSessionService,
)

router = APIRouter(prefix="/chat/sessions", tags=["chat-sessions"])

# Route 只声明需要业务服务，不构造 AsyncSession 或 Repository。FastAPI 会在每个
# 请求中调用 dependency，并把同一个请求级 Session 注入 service。
ChatSessionServiceDependency = Annotated[
    ChatSessionService,
    Depends(get_chat_session_service),
]


@router.post(
    "",
    response_model=ChatSessionResponse,
    status_code=status.HTTP_201_CREATED,
    responses={
        status.HTTP_401_UNAUTHORIZED: {
            "model": ErrorResponse,
            "description": "Bearer token is missing or invalid",
        },
    },
)
async def create_chat_session(
    request: ChatSessionCreateRequest,
    service: ChatSessionServiceDependency,
    current_user: CurrentUserDependency,
) -> ChatSessionResponse:
    """为当前认证用户创建一个业务聊天会话.

    Args:
        request: 只包含经过规范化的可选标题；不能提交 user_id 或 thread_id。
        service: 当前请求的事务与 Repository 协调器。
        current_user: JWT 验签并经数据库确认的可信用户上下文。

    Returns:
        已提交的业务会话。其 UUID thread_id 可用于后续聊天入口。
    """
    return await service.create(
        user_id=current_user.user_id,
        request=request,
    )


@router.get(
    "",
    response_model=ChatSessionListResponse,
    responses={
        status.HTTP_401_UNAUTHORIZED: {
            "model": ErrorResponse,
            "description": "Bearer token is missing or invalid",
        },
    },
)
async def list_chat_sessions(
    service: ChatSessionServiceDependency,
    current_user: CurrentUserDependency,
) -> ChatSessionListResponse:
    """只列出当前认证用户拥有的业务会话.

    Args:
        service: 当前请求的业务会话 application service。
        current_user: 服务端认证得到的可信身份；客户端不能在 query 中替换它。

    Returns:
        当前用户的稳定排序会话集合；无数据时 sessions 为空数组。
    """
    return await service.list_owned(user_id=current_user.user_id)


@router.get(
    "/{thread_id}",
    response_model=ChatSessionResponse,
    responses={
        status.HTTP_401_UNAUTHORIZED: {
            "model": ErrorResponse,
            "description": "Bearer token is missing or invalid",
        },
        status.HTTP_404_NOT_FOUND: {
            "model": ErrorResponse,
            "description": "Chat session is absent from the current user scope",
        },
    },
)
async def read_chat_session(
    thread_id: UUID,
    service: ChatSessionServiceDependency,
    current_user: CurrentUserDependency,
) -> ChatSessionResponse:
    """在当前用户作用域内读取一条业务会话.

    Args:
        thread_id: 公开业务会话 UUID。它是标识符，不是授权凭据。
        service: 当前请求的业务会话 application service。
        current_user: 可信用户上下文，提供所有权 SQL 的 user_id。

    Returns:
        同时匹配 thread_id 和当前 user_id 的业务会话。

    Raises:
        HTTPException: 会话不存在或属于其他用户时统一返回 404，不泄漏 UUID 是否
            真实存在于其他用户作用域。
    """
    try:
        return await service.get_owned(
            session_id=thread_id,
            user_id=current_user.user_id,
        )
    except ChatSessionNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Chat session was not found",
        ) from None
