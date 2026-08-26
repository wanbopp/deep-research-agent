"""Chat Agent 执行前的业务会话所有权协议.

本模块只定义应用层需要的能力和稳定错误，不依赖 SQLAlchemy、FastAPI、Redis
或 LangGraph。ChatService 将来只知道“验证当前用户是否拥有会话”，不需要知道
验证来自 PostgreSQL、测试替身还是其他持久化后端。
"""

from collections.abc import Iterable
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


class InProcessChatSessionOwnershipVerifier:
    """为直接构造 ChatService 的单进程 smoke 提供显式所有权集合.

    它不会替换模型、Graph 或 checkpoint，只替换业务 owner 查询的存储位置。
    每个允许的 ``(user_id, session_id)`` 必须在构造时明确登记，未知组合一律
    fail-closed。production lifespan 不得使用本实现，必须注入 PostgreSQL verifier。
    """

    def __init__(
        self,
        owned_sessions: Iterable[tuple[UUID, UUID]],
    ) -> None:
        """保存当前 smoke 明确拥有的用户与会话组合.

        Args:
            owned_sessions: ``(user_id, session_id)`` 二元组集合。转换为 frozenset
                后不可变，能够被同一个单进程 ChatService 并发读取。
        """
        self._owned_sessions = frozenset(owned_sessions)

    async def require_owned(
        self,
        *,
        session_id: UUID,
        user_id: UUID,
    ) -> None:
        """接受已登记组合，拒绝未知或跨用户组合.

        Args:
            session_id: smoke 使用的公开业务会话 UUID。
            user_id: smoke 显式提供的可信用户 UUID。

        Raises:
            ChatSessionNotFoundError: 当前组合未登记。错误语义与 PostgreSQL 实现
                一致，因此 ChatService 和 route 不需要识别具体 verifier 类型。
        """
        if (user_id, session_id) not in self._owned_sessions:
            raise ChatSessionNotFoundError


__all__ = [
    "ChatSessionNotFoundError",
    "ChatSessionOwnershipVerifier",
    "InProcessChatSessionOwnershipVerifier",
]
