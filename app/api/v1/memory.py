"""当前用户长期记忆的列表与删除 HTTP 入口.

记忆由后台提取写入、聊天图检索使用；本路由只给用户提供对自己记忆的审阅
与管理能力，不提供创建入口（写入仍由可信的后台提取链路负责）。
"""

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Response, status

from app.api.dependencies import CurrentUserDependency, get_memory_service
from app.schemas.base import ErrorResponse
from app.schemas.memory import MemoryItem, MemoryListResponse, MemoryResponse
from app.services.memory import MemoryUnavailableError
from app.services.memory_service import MemoryService

router = APIRouter(prefix="/memory", tags=["memory"])

# MemoryService 是 lifespan 级共享服务；路由只声明依赖，不构造 Session。
MemoryServiceDependency = Annotated[
    MemoryService,
    Depends(get_memory_service),
]


def _to_response(item: MemoryItem) -> MemoryResponse:
    """把应用层记忆条目映射为不回显归属的公开视图."""
    return MemoryResponse(
        memory_id=item.id,
        content=item.content,
        kind=item.kind,
        source_thread_id=item.source_thread_id,
        created_at=item.created_at,
    )


@router.get(
    "",
    response_model=MemoryListResponse,
    responses={
        status.HTTP_401_UNAUTHORIZED: {
            "model": ErrorResponse,
            "description": "Bearer token is missing or invalid",
        },
        status.HTTP_503_SERVICE_UNAVAILABLE: {
            "model": ErrorResponse,
            "description": "Memory backend is temporarily unavailable",
        },
    },
)
async def list_memories(
    memory_service: MemoryServiceDependency,
    current_user: CurrentUserDependency,
) -> MemoryListResponse:
    """列出当前认证用户的全部长期记忆.

    Args:
        memory_service: lifespan 共享的记忆应用服务；user 作用域由可信
            ``current_user.user_id`` 决定，客户端不能通过参数替换。
        current_user: JWT 验签并经数据库确认的可信用户上下文。

    Returns:
        按创建时间倒序排列的记忆；无数据时 memories 为空数组。

    Raises:
        HTTPException: 记忆后端不可用时返回 503，客户端可稍后重试；不把故障
            伪装成空列表，避免用户误以为记忆丢失。
    """
    try:
        items = await memory_service.list(user_id=current_user.user_id)
    except MemoryUnavailableError:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Memory backend is temporarily unavailable",
        ) from None

    return MemoryListResponse(
        memories=tuple(_to_response(item) for item in items),
    )


@router.delete(
    "/{memory_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    response_class=Response,
    responses={
        status.HTTP_401_UNAUTHORIZED: {
            "model": ErrorResponse,
            "description": "Bearer token is missing or invalid",
        },
        status.HTTP_503_SERVICE_UNAVAILABLE: {
            "model": ErrorResponse,
            "description": "Memory backend is temporarily unavailable",
        },
    },
)
async def delete_memory(
    memory_id: UUID,
    memory_service: MemoryServiceDependency,
    current_user: CurrentUserDependency,
) -> Response:
    """删除当前用户拥有的一条记忆.

    Args:
        memory_id: 客户端给出的记忆 UUID。它是标识符，不是授权凭据；删除条件
            始终叠加可信 ``current_user.user_id``。
        memory_service: lifespan 共享的记忆应用服务，负责存储删除与搜索缓存
            失效。
        current_user: JWT 验签并经数据库确认的可信用户上下文。

    Returns:
        空的 HTTP 204 响应。删除是幂等的：不存在或属于其他用户的 memory_id
        同样返回 204，避免删除接口成为资源枚举工具。

    Raises:
        HTTPException: 记忆后端不可用时返回 503，客户端可以安全重试同一个请求。
    """
    try:
        await memory_service.delete(
            user_id=current_user.user_id,
            memory_id=memory_id,
        )
    except MemoryUnavailableError:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Memory backend is temporarily unavailable",
        ) from None

    # 显式构造 Response，确保 204 不会被序列化成 JSON null 或空对象。
    return Response(status_code=status.HTTP_204_NO_CONTENT)


__all__ = ["router"]
