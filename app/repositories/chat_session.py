"""聊天会话数据访问边界."""

from uuid import UUID

from sqlmodel import select

from app.models import ChatSession, ChatSessionStatus, utc_now
from app.repositories.base import RepositoryBase


class ChatSessionRepository(RepositoryBase):
    """封装聊天会话的创建和用户所有权查询."""

    async def create(self, *, user_id: UUID, title: str = "New chat") -> ChatSession:
        """在当前事务中创建属于指定用户的业务会话."""
        session = ChatSession(user_id=user_id, title=title)
        return await self._persist(session, resource="chat_session")

    async def get_by_id(self, session_id: UUID, *, user_id: UUID) -> ChatSession | None:
        """查询当前用户仍可正常访问的 active 会话."""
        statement = select(ChatSession).where(
            ChatSession.id == session_id,
            ChatSession.user_id == user_id,
            ChatSession.status == ChatSessionStatus.ACTIVE,
        )
        result = await self._session.execute(statement)
        return result.scalar_one_or_none()

    async def list_by_user(self, user_id: UUID) -> tuple[ChatSession, ...]:
        """稳定地列出一个用户拥有的全部业务会话."""
        statement = (
            select(ChatSession)
            .where(
                ChatSession.user_id == user_id,
                ChatSession.status == ChatSessionStatus.ACTIVE,
            )
            # 这里使用已选中表的列标签，是为了绕开 SQLModel 类字段的静态类型
            # 局限；SQLAlchemy 会把标签解析为 chat_sessions 的对应列。
            .order_by("created_at", "id")
        )
        result = await self._session.execute(statement)
        return tuple(result.scalars().all())

    async def get_for_cleanup(
        self,
        session_id: UUID,
        *,
        user_id: UUID,
    ) -> ChatSession | None:
        """锁定 owner-scoped 行，包括已经进入 deleting 的会话.

        普通查询必须隐藏 deleting；cleanup 重试却必须重新找到它。两种用途不能
        复用同一查询，否则失败恢复会把自己的持久 tombstone 当作 404。
        """
        statement = (
            select(ChatSession)
            .where(
                ChatSession.id == session_id,
                ChatSession.user_id == user_id,
            )
            .with_for_update()
        )
        result = await self._session.execute(statement)
        return result.scalar_one_or_none()

    async def mark_deleting(self, chat_session: ChatSession) -> ChatSession:
        """把会话推进到 deleting，但不提交外层事务.

        Args:
            chat_session: ``get_for_cleanup`` 在当前事务中锁定的 ORM 实体。

        Returns:
            已 flush 和 refresh 的同一业务实体。
        """
        if chat_session.status != ChatSessionStatus.DELETING:
            chat_session.status = ChatSessionStatus.DELETING
            chat_session.updated_at = utc_now()

        await self._session.flush()
        await self._session.refresh(chat_session)
        return chat_session

    async def delete(self, chat_session: ChatSession) -> None:
        """删除已经清空 checkpoint 的业务行，但不提交事务."""
        await self._session.delete(chat_session)
        await self._session.flush()
