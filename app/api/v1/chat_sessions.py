"""业务聊天会话的创建、列表与所有权查询 HTTP 入口."""

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Response, status

from app.api.dependencies import (
    CurrentUserDependency,
    get_chat_session_cleanup_service,
    get_chat_session_service,
)
from app.schemas.base import ErrorResponse
from app.schemas.chat_session import (
    ChatSessionCreateRequest,
    ChatSessionListResponse,
    ChatSessionResponse,
)
from app.services.chat_session_ownership import ChatSessionNotFoundError
from app.services.chat_session_cleanup import ChatSessionCleanupService
from app.services.chat_sessions import ChatSessionService

router = APIRouter(prefix="/chat/sessions", tags=["chat-sessions"])

# Route 只声明需要业务服务，不构造 AsyncSession 或 Repository。FastAPI 会在每个
# 请求中调用 dependency，并把同一个请求级 Session 注入 service。
ChatSessionServiceDependency = Annotated[
    ChatSessionService,
    Depends(get_chat_session_service),
]

# 删除协调器是 lifespan 级共享服务，而不是请求级 ORM service。类型别名把
# Python 类型和 FastAPI dependency 放在一起，使 route 只表达“需要清理能力”，
# 不需要知道 saver、Redis guard、sessionmaker 或 internal key 的组合过程。
ChatSessionCleanupServiceDependency = Annotated[
    ChatSessionCleanupService,
    Depends(get_chat_session_cleanup_service),
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


@router.delete(
    "/{thread_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    response_class=Response,
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
        status.HTTP_503_SERVICE_UNAVAILABLE: {
            "model": ErrorResponse,
            "description": "The chat session cannot be safely cleaned up right now",
        },
    },
)
async def delete_chat_session(
    thread_id: UUID,
    cleanup_service: ChatSessionCleanupServiceDependency,
    current_user: CurrentUserDependency,
) -> Response:
    """删除当前用户拥有的业务会话及其 LangGraph checkpoint.

    Args:
        thread_id: 客户端可见的业务会话 UUID。cleanup service 会使用可信
            ``current_user.user_id`` 把它转换为内部 checkpoint/guard key。
        cleanup_service: lifespan 共享的三阶段清理协调器。route 不直接操作
            ORM、Redis、saver，也不自行管理事务。
        current_user: JWT 验签并经数据库确认的可信身份。客户端无法通过 path
            或 body 替换这里的 user_id。

    Returns:
        空的 HTTP 204 响应。204 表示目标会话和 checkpoint 已完成清理，因此
        不能携带 JSON body；客户端不需要再解析成功响应模型。

    Raises:
        HTTPException: 会话不存在、已经删除或属于其他用户时统一返回 404，避免
            删除接口成为资源枚举工具。
        ChatThreadBusyError: 同一内部 thread 正在执行 Graph 或另一次清理。由
            全局 handler 转换为稳定 409。
        ChatExecutionGuardUnavailableError: Redis 无法保证互斥时由全局 handler
            fail-closed 为 503，不会降级为无锁删除。
        ChatCheckpointCleanupError: saver 清理失败时由全局 handler 转换为可重试
            503；业务行保持 deleting，后续 DELETE 可以继续清理。

    Notes:
        ``ChatSessionCleanupStateError`` 没有在这里转换。它表示服务端持久状态
        违反内部不变量，应该保留为 500 并留下 traceback，而不是伪装成 4xx。
    """
    try:
        await cleanup_service.delete_owned(
            session_id=thread_id,
            user_id=current_user.user_id,
        )
    except ChatSessionNotFoundError:
        # absent、cross-user、已经物理删除三种情况保持相同状态码和文案。
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Chat session was not found",
        ) from None

    # 显式构造 Response，确保 204 不会被序列化成 JSON null 或空对象。
    return Response(status_code=status.HTTP_204_NO_CONTENT)
