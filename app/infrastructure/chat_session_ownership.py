"""使用请求独立 SQLAlchemy Session 验证业务聊天会话所有权."""

from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.repositories import ChatSessionRepository
from app.services.chat_session_ownership import ChatSessionNotFoundError


class PostgresChatSessionOwnershipVerifier:
    """通过 PostgreSQL owner-scoped 查询实现会话授权协议.

    这个对象可以安全地由 lifespan 创建并长期共享，因为它只保存并发安全的
    ``async_sessionmaker``。每次调用都会创建一个新的短生命周期 AsyncSession，
    不会让两个并发 Graph 共享事务、identity map 或失败状态。
    """

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
    ) -> None:
        """保存创建短生命周期数据库工作单元的工厂.

        Args:
            session_factory: lifespan 拥有的 ORM Session 工厂。verifier 只借用它
                创建 Session，不关闭工厂背后的 Engine。
        """
        self._session_factory = session_factory

    async def require_owned(
        self,
        *,
        session_id: UUID,
        user_id: UUID,
    ) -> None:
        """使用 ``session_id + user_id`` 查询验证会话所有权.

        Args:
            session_id: 公开业务会话 UUID。
            user_id: JWT 验签并经用户表确认的可信 UUID。

        Returns:
            查询命中时返回 None；ORM 实体不会离开本方法。

        Raises:
            ChatSessionNotFoundError: 查询不到匹配行。不存在和属于其他用户都进入
                相同分支，调用方不能据此枚举其他用户资源。
            SQLAlchemyError: 数据库不可用或 SQL 执行失败时原样向上传播。这里不能
                把基础设施故障伪装成 404，否则系统会错误地声称资源不存在。
        """
        # Session 和事务都严格限制在一次授权检查中。正常返回会结束只读事务；
        # 查询异常则由 begin() rollback，随后 Session 上下文负责关闭资源。
        async with self._session_factory() as session:
            async with session.begin():
                chat_session = await ChatSessionRepository(session).get_by_id(
                    session_id,
                    user_id=user_id,
                )

        if chat_session is None:
            raise ChatSessionNotFoundError


__all__ = ["PostgresChatSessionOwnershipVerifier"]
