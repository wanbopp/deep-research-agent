"""聊天会话数据访问边界."""

from datetime import datetime
from typing import cast
from uuid import UUID

from sqlalchemy import Table, or_, update
from sqlmodel import select

from app.models import (
    DEFAULT_CHAT_SESSION_TITLE,
    ChatSession,
    ChatSessionStatus,
    utc_now,
)
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

    async def claim_title_generation(
        self,
        session_id: UUID,
        *,
        user_id: UUID,
        claim_token: UUID,
        claimed_at: datetime,
        stale_before: datetime,
    ) -> bool:
        """使用单条条件更新申请会话自动命名租约.

        Args:
            session_id: 已完成一轮聊天的公开业务会话 UUID。
            user_id: 认证链提供的可信 owner UUID。
            claim_token: 本次后台任务生成的随机所有权凭证。
            claimed_at: 本次申请的 UTC 时间。
            stale_before: 早于该时间的旧租约允许被接管。

        Returns:
            当前任务成功取得租约时为 ``True``；无资格或已被其他 worker 占有时
            为 ``False``。调用方只有在 True 时才能产生真实模型费用。

        Notes:
            检查默认标题、租约状态和写入新 token 必须位于同一条 SQL 中。
            ``SELECT`` 后再 ``UPDATE`` 会让多个 worker 同时通过检查。
        """
        table = self._chat_session_table()
        statement = (
            update(table)
            .where(
                table.c.id == session_id,
                table.c.user_id == user_id,
                table.c.status == ChatSessionStatus.ACTIVE.value,
                table.c.title == DEFAULT_CHAT_SESSION_TITLE,
                table.c.title_generated_at.is_(None),
                or_(
                    table.c.title_claim_token.is_(None),
                    table.c.title_claimed_at < stale_before,
                ),
            )
            .values(
                title_claim_token=claim_token,
                title_claimed_at=claimed_at,
            )
            .returning(table.c.id)
        )
        result = await self._session.execute(statement)
        return result.scalar_one_or_none() is not None

    async def complete_title_generation(
        self,
        session_id: UUID,
        *,
        user_id: UUID,
        claim_token: UUID,
        title: str,
        completed_at: datetime,
    ) -> bool:
        """只允许当前租约持有者提交仍未被用户修改的标题.

        最终条件再次检查默认标题和 token。用户在模型调用期间手动改名，或者
        旧 worker 的租约已被接管时，本次更新返回 False，不会覆盖较新的事实。
        """
        table = self._chat_session_table()
        statement = (
            update(table)
            .where(
                table.c.id == session_id,
                table.c.user_id == user_id,
                table.c.status == ChatSessionStatus.ACTIVE.value,
                table.c.title == DEFAULT_CHAT_SESSION_TITLE,
                table.c.title_generated_at.is_(None),
                table.c.title_claim_token == claim_token,
            )
            .values(
                title=title,
                title_generated_at=completed_at,
                title_claim_token=None,
                title_claimed_at=None,
                updated_at=completed_at,
            )
            .returning(table.c.id)
        )
        result = await self._session.execute(statement)
        return result.scalar_one_or_none() is not None

    async def release_title_generation_claim(
        self,
        session_id: UUID,
        *,
        user_id: UUID,
        claim_token: UUID,
    ) -> bool:
        """在生成失败或任务取消后释放仍属于当前任务的租约.

        token 条件保证迟到的旧任务不能清除新 worker 已经取得的租约。释放失败
        不代表标题错误；可能是用户改名、会话删除或租约已被接管。
        """
        table = self._chat_session_table()
        statement = (
            update(table)
            .where(
                table.c.id == session_id,
                table.c.user_id == user_id,
                table.c.title_claim_token == claim_token,
                table.c.title_generated_at.is_(None),
            )
            .values(
                title_claim_token=None,
                title_claimed_at=None,
            )
            .returning(table.c.id)
        )
        result = await self._session.execute(statement)
        return result.scalar_one_or_none() is not None

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

    @staticmethod
    def _chat_session_table() -> Table:
        """取得 SQLAlchemy Table，集中隔离 SQLModel 的动态 ``__table__`` 属性."""
        return cast(Table, vars(ChatSession)["__table__"])
