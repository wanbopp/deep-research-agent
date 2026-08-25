"""Chat Agent 执行前的业务会话所有权协议.

本模块只定义应用层需要的能力和稳定错误，不依赖 SQLAlchemy、FastAPI、Redis
或 LangGraph。ChatService 将来只知道“验证当前用户是否拥有会话”，不需要知道
验证来自 PostgreSQL、测试替身还是其他持久化后端。
"""

from typing import Protocol
from uuid import UUID


class ChatSessionNotFoundError(LookupError):
    """当前用户作用域中不存在指定业务会话."""

    def __init__(self) -> None:
        """使用固定文本，避免泄漏相同 UUID 是否属于其他用户."""
        super().__init__("Chat session was not found")


class ChatSessionOwnershipVerifier(Protocol):
    """定义 ChatService 所依赖的业务会话授权能力.

    这是结构化协议，不提供数据库实现。任何对象只要实现相同的异步
    ``require_owned`` 方法，就能注入 ChatService；调用方不需要继承本类。
    """

    async def require_owned(
        self,
        *,
        session_id: UUID,
        user_id: UUID,
    ) -> None:
        """要求业务会话存在于指定用户作用域.

        Args:
            session_id: 客户端使用的公开业务会话 UUID。
            user_id: 认证链提供的可信用户 UUID，不能来自请求正文或模型。

        Returns:
            所有权成立时返回 None。调用方只关心是否通过，不需要 ORM 实体。

        Raises:
            ChatSessionNotFoundError: 会话不存在或属于其他用户。两种情况使用相同
                错误，防止调用方泄漏资源是否存在于其他用户作用域。
        """
        ...


__all__ = [
    "ChatSessionNotFoundError",
    "ChatSessionOwnershipVerifier",
]
