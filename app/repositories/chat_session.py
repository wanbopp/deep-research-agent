"""聊天会话数据访问边界."""

from uuid import UUID

from sqlmodel import select

from app.models import ChatSession
from app.repositories.base import RepositoryBase


class ChatSessionRepository(RepositoryBase):
    """封装聊天会话的创建和用户所有权查询."""

    async def create(self, *, user_id: UUID, title: str = "New chat") -> ChatSession:
        """在当前事务中创建属于指定用户的业务会话."""
        session = ChatSession(user_id=user_id, title=title)
        return await self._persist(session, resource="chat_session")

    async def get_by_id(self, session_id: UUID, *, user_id: UUID) -> ChatSession | None:
        """同时按会话 ID 和用户 ID 查询，避免跨用户读取资源."""
        statement = select(ChatSession).where(
            ChatSession.id == session_id,
            ChatSession.user_id == user_id,
        )
        result = await self._session.execute(statement)
        return result.scalar_one_or_none()

    async def list_by_user(self, user_id: UUID) -> tuple[ChatSession, ...]:
        """稳定地列出一个用户拥有的全部业务会话."""
        statement = (
            select(ChatSession)
            .where(ChatSession.user_id == user_id)
            # 这里使用已选中表的列标签，是为了绕开 SQLModel 类字段的静态类型
            # 局限；SQLAlchemy 会把标签解析为 chat_sessions 的对应列。
            .order_by("created_at", "id")
        )
        result = await self._session.execute(statement)
        return tuple(result.scalars().all())
