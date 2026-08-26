"""基于 PostgreSQL/pgvector 的用户长期记忆存储适配器."""

from typing import cast
from uuid import UUID

from sqlalchemy import Table, delete
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlmodel import select

from app.core.logging import logger
from app.models import (
    MEMORY_EMBEDDING_DIMENSIONS,
    ChatSession,
    ChatSessionStatus,
    Memory,
)
from app.schemas.memory import MemoryCreate, MemoryItem, MemoryKind, MemoryQuery
from app.services.embeddings import TextEmbedder
from app.services.memory import (
    MemorySourceNotFoundError,
    MemoryUnavailableError,
)


class PostgresMemoryStore:
    """用真实 Embedding 和 pgvector 实现 ``MemoryStore``.

    store 保存的是 ``async_sessionmaker``，不是可变的 ``AsyncSession``。每个方法
    都创建自己的短生命周期 Session，因此同一个 store 可以被多个请求并发复用，
    而不会让事务、identity map 或 rollback 状态串到其他请求。
    """

    def __init__(
        self,
        *,
        session_factory: async_sessionmaker[AsyncSession],
        embedder: TextEmbedder,
    ) -> None:
        """保存资源工厂，并在第一次请求前验证向量维度契约.

        Args:
            session_factory: 应用 lifespan 拥有的 SQLAlchemy 异步 Session 工厂。
                store 每次操作只借用一个 Session，不关闭工厂背后的 Engine。
            embedder: 真实文本向量化适配器。可以由多个并发调用复用，但不得在
                实现内部保存某次用户请求的文本、向量或身份状态。

        Raises:
            ValueError: provider 维度与 PostgreSQL ``VECTOR(1536)`` 不一致。
        """
        if embedder.dimensions != MEMORY_EMBEDDING_DIMENSIONS:
            raise ValueError("Embedding dimensions must match the PostgreSQL vector schema")
        self._session_factory = session_factory
        self._embedder = embedder

    async def search(
        self,
        *,
        user_id: UUID,
        query: MemoryQuery,
    ) -> tuple[MemoryItem, ...]:
        """在 SQL 中完成用户过滤、可选分类过滤和余弦距离排序.

        Args:
            user_id: 认证链给出的可信用户 UUID，也是检索 namespace。
            query: 已校验的文本、分类集合和返回上限。

        Returns:
            与查询语义最接近的记忆元组；普通未命中返回空元组。

        Raises:
            MemoryUnavailableError: provider 或 PostgreSQL 无法可靠完成查询。

        Notes:
            当前先做 owner-scoped 精确余弦排序。单用户记忆量较小时，这比过早引入
            HNSW 参数更容易验证；规模增长后可以在不改变 MemoryStore 协议的前提
            下添加向量索引。
        """
        # 网络 I/O 发生在借出数据库连接之前。慢 provider 不会占用 ORM pool。
        query_vector = await self._embedder.embed_query(query.text)
        memory_table = self._memory_table()
        distance = memory_table.c.embedding.cosine_distance(list(query_vector))

        statement = select(Memory).where(memory_table.c.user_id == user_id)
        if query.kinds is not None:
            statement = statement.where(memory_table.c.kind.in_(sorted(kind.value for kind in query.kinds)))
        statement = statement.order_by(
            distance,
            memory_table.c.created_at,
            memory_table.c.id,
        ).limit(query.limit)

        try:
            async with self._session_factory() as session:
                result = await session.execute(statement)
                rows = result.scalars().all()
        except SQLAlchemyError as exc:
            self._log_database_failure(operation="search", exc=exc)
            raise MemoryUnavailableError() from None

        try:
            return tuple(self._to_item(row) for row in rows)
        except ValueError as exc:
            # 非法 kind 或审计时间意味着数据库状态已偏离应用契约，不能把坏数据
            # 当作正常检索结果交给 Agent。
            self._log_database_failure(operation="decode", exc=exc)
            raise MemoryUnavailableError() from None

    async def add(
        self,
        *,
        user_id: UUID,
        memory: MemoryCreate,
    ) -> MemoryItem:
        """验证来源所有权、请求真实向量，并在一个短事务中写入记忆.

        Args:
            user_id: 来自认证链的可信用户 UUID。
            memory: 不含 user_id 和向量的严格候选记忆。

        Returns:
            不暴露 embedding 的权威 ``MemoryItem``。

        Raises:
            MemorySourceNotFoundError: 来源会话不存在、不是 active 或属于其他用户。
            MemoryUnavailableError: provider 或 PostgreSQL 无法安全完成写入。
        """
        # 第一次短查询用于尽早拒绝非法来源，避免为明显无效的请求支付模型成本。
        await self._require_active_source(
            user_id=user_id,
            source_thread_id=memory.source_thread_id,
            lock_row=False,
        )

        # provider 请求不能放进 session.begin()。外部网络慢或重试时，数据库连接
        # 应立即归还连接池，而不是长期占用事务和潜在行锁。
        (embedding,) = await self._embedder.embed_documents((memory.content,))

        try:
            async with self._session_factory() as session:
                async with session.begin():
                    # 在最终提交点再次验证并锁住来源行，关闭“初查后会话被删除”
                    # 的竞态窗口。锁只持续到当前很短的 INSERT 事务结束。
                    await self._require_active_source_in_session(
                        session=session,
                        user_id=user_id,
                        source_thread_id=memory.source_thread_id,
                        lock_row=True,
                    )
                    row = Memory(
                        user_id=user_id,
                        content=memory.content,
                        kind=memory.kind.value,
                        source_thread_id=memory.source_thread_id,
                        embedding=list(embedding),
                    )
                    session.add(row)
                    await session.flush()

                    # 在 commit 前把 ORM 行转换为不可变 schema。即使调用方提供的
                    # sessionmaker 使用 expire_on_commit=True，也不会在事务外触发
                    # 隐式数据库查询。
                    item = self._to_item(row)
        except MemorySourceNotFoundError:
            raise
        except SQLAlchemyError as exc:
            self._log_database_failure(operation="add", exc=exc)
            raise MemoryUnavailableError() from None

        return item

    async def delete(
        self,
        *,
        user_id: UUID,
        memory_id: UUID,
    ) -> None:
        """使用 ``memory_id + user_id`` 幂等删除，不泄露资源是否存在.

        Args:
            user_id: 认证链给出的可信用户 UUID。
            memory_id: 客户端希望删除的记忆 UUID。

        Raises:
            MemoryUnavailableError: PostgreSQL 无法完成删除事务。
        """
        memory_table = self._memory_table()
        statement = delete(Memory).where(
            memory_table.c.id == memory_id,
            memory_table.c.user_id == user_id,
        )
        try:
            async with self._session_factory() as session:
                async with session.begin():
                    await session.execute(statement)
        except SQLAlchemyError as exc:
            self._log_database_failure(operation="delete", exc=exc)
            raise MemoryUnavailableError() from None

        # 不检查 rowcount：不存在和属于其他用户都视为成功，调用方无法利用响应
        # 差异探测其他用户是否拥有某个 UUID。

    async def _require_active_source(
        self,
        *,
        user_id: UUID,
        source_thread_id: UUID,
        lock_row: bool,
    ) -> None:
        """在独立短 Session 中验证来源会话归属."""
        try:
            async with self._session_factory() as session:
                await self._require_active_source_in_session(
                    session=session,
                    user_id=user_id,
                    source_thread_id=source_thread_id,
                    lock_row=lock_row,
                )
        except MemorySourceNotFoundError:
            raise
        except SQLAlchemyError as exc:
            self._log_database_failure(operation="source_check", exc=exc)
            raise MemoryUnavailableError() from None

    @staticmethod
    async def _require_active_source_in_session(
        *,
        session: AsyncSession,
        user_id: UUID,
        source_thread_id: UUID,
        lock_row: bool,
    ) -> None:
        """在调用方 Session 中执行 owner-scoped 来源查询.

        Args:
            session: 当前操作独享的短生命周期 AsyncSession。
            user_id: 可信用户 UUID。
            source_thread_id: 候选记忆声明的业务会话 UUID。
            lock_row: True 时为最终写入获取行锁；初步检查保持 False。

        Raises:
            MemorySourceNotFoundError: 联合条件没有命中 active 会话。
            SQLAlchemyError: 查询或加锁失败，交由调用方统一映射。
        """
        statement = select(ChatSession.id).where(
            ChatSession.id == source_thread_id,
            ChatSession.user_id == user_id,
            ChatSession.status == ChatSessionStatus.ACTIVE,
        )
        if lock_row:
            statement = statement.with_for_update()

        result = await session.execute(statement)
        if result.scalar_one_or_none() is None:
            raise MemorySourceNotFoundError()

    @staticmethod
    def _to_item(row: Memory) -> MemoryItem:
        """把数据库内部行投影成不含高维向量的应用模型."""
        return MemoryItem(
            id=row.id,
            user_id=row.user_id,
            content=row.content,
            kind=MemoryKind(row.kind),
            source_thread_id=row.source_thread_id,
            created_at=row.created_at,
            updated_at=row.updated_at,
        )

    @staticmethod
    def _memory_table() -> Table:
        """取得 SQLAlchemy Table，以访问 pgvector 专用比较运算符."""
        # SQLModel 的 Python 字段类型是 list[float]，但 cosine_distance 是运行时
        # Column comparator 提供的方法。转成 Table 后，Pyright 与 SQLAlchemy 都能
        # 明确知道这里正在构造 SQL 表达式，而不是对普通 list 调用方法。
        # ``__table__`` 由 SQLModel 的 metaclass 在运行时注入，类型桩没有声明；
        # 从类命名空间读取只隔离这一处框架魔术，不降低其他业务字段的类型检查。
        return cast(Table, vars(Memory)["__table__"])

    @staticmethod
    def _log_database_failure(*, operation: str, exc: Exception) -> None:
        """记录可观测但不泄露用户数据的数据库失败摘要."""
        logger.warning(
            "memory_database_operation_failed",
            operation=operation,
            error_type=type(exc).__name__,
        )


__all__ = ["PostgresMemoryStore"]
