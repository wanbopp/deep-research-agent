"""可重试地协调业务 ChatSession 与 LangGraph checkpoint 清理.

这里解决的不是普通的单表删除，而是一次操作需要修改两套独立事务资源：

* SQLAlchemy ORM 连接负责应用拥有的 ``chat_sessions`` 表；
* LangGraph ``AsyncPostgresSaver`` 连接负责 checkpoints、blobs 和 writes。

两套连接无法组成当前项目中的同一个数据库事务，因此本模块不伪装成“跨连接原子
删除”，而是用持久状态、幂等操作和同 key guard 组成可恢复工作流。
"""

from collections.abc import Callable
from typing import Protocol
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.models import ChatSessionStatus
from app.repositories import ChatSessionRepository
from app.services.chat_guard import ChatExecutionGuard
from app.services.chat_session_ownership import ChatSessionNotFoundError
from app.services.cache import Cache
from app.services.chat_session_cache import invalidate_chat_session_list_cache


class ChatCheckpointStore(Protocol):
    """声明 cleanup coordinator 需要的最小 checkpointer 能力."""

    async def adelete_thread(self, thread_id: str) -> None:
        """幂等删除一个内部 thread 的全部 checkpoint 数据."""
        ...


InternalThreadIdFactory = Callable[[UUID, UUID], str]


class ChatCheckpointCleanupError(RuntimeError):
    """checkpoint 清理失败；业务会话保留 deleting，允许安全重试."""

    def __init__(self) -> None:
        """使用固定文本，避免把底层 SQL 或连接信息带到公开边界."""
        super().__init__("Chat checkpoint cleanup failed")


class ChatSessionCleanupStateError(RuntimeError):
    """持久状态不符合 cleanup coordinator 的内部不变量."""


class ChatSessionCleanupService:
    """使用 guard 和持久删除状态协调两套 PostgreSQL 连接.

    ORM ``AsyncSession`` 与 ``AsyncPostgresSaver`` 使用不同连接池，因此无法共享
    一个原子事务。本服务采用可恢复的三阶段流程。状态迁移可以读成：

    ``active``
      -> 阶段 1：ORM 提交 ``deleting``，持久记录删除意图
      -> 阶段 2：幂等删除 LangGraph checkpoint
      -> 阶段 3：ORM 物理删除 ``deleting`` 业务行
      -> 会话完全消失

    为什么先提交 ``deleting``，而不是先删除业务行：

    * 阶段 1 提交前失败：ORM 事务回滚，会话仍为 active，可以重新发起删除；
    * 阶段 1 提交后失败：deleting 是持久 tombstone，普通 Agent 查询不可见；
    * 阶段 2 部分删除后失败：按同一 thread_id 再次 DELETE，结果仍然一致；
    * 阶段 2 成功、阶段 3 前失败：重试会再次执行安全的空 DELETE，再删除 tombstone；
    * 阶段 3 成功：业务行和 checkpoint 都不存在，删除完成。

    因此 ``deleting`` 不是临时 Python 标志，而是进程崩溃后仍存在的恢复坐标。
    普通会话查询只接受 ``active``，所以半完成清理不会重新进入 Agent。

    同一个 internal thread guard 覆盖整个三阶段流程，保证 Graph 不能在阶段之间
    写入新 checkpoint。异常或任务取消离开 ``async with`` 时，guard 仍由上下文
    管理器释放；已经提交的 deleting 状态则保留下来，供下一次调用继续恢复。
    """

    def __init__(
        self,
        *,
        session_factory: async_sessionmaker[AsyncSession],
        checkpoint_store: ChatCheckpointStore,
        execution_guard: ChatExecutionGuard,
        internal_thread_id_factory: InternalThreadIdFactory,
        cache: Cache,
    ) -> None:
        """保存可跨请求共享的无状态依赖.

        Args:
            session_factory: lifespan 拥有的 ORM Session 工厂。每个阶段都会创建
                独立短生命周期 Session，绝不在共享 service 中保存 AsyncSession。
            checkpoint_store: production 使用 ``AsyncPostgresSaver``。这里只依赖
                ``adelete_thread``，便于 smoke 注入一次性故障而不伪造 Agent。
            execution_guard: 与 ChatService 共用的内部 thread 执行权接口。
            internal_thread_id_factory: 必须与 ChatService 使用同一映射函数，保证
                删除锁、checkpoint key 和 Graph 执行锁属于同一并发域。
            cache: lifespan 共享的缓存协议。阶段 1 提交 deleting 后用于尽力
                失效用户列表；缓存失败不能中断后续 checkpoint 清理。
        """
        self._session_factory = session_factory
        self._checkpoint_store = checkpoint_store
        self._execution_guard = execution_guard
        self._internal_thread_id_factory = internal_thread_id_factory
        self._cache = cache

    async def delete_owned(
        self,
        *,
        session_id: UUID,
        user_id: UUID,
    ) -> None:
        """清理当前用户拥有的业务会话及其全部 checkpoint.

        Args:
            session_id: 客户端可见的业务会话 UUID。
            user_id: 已由认证链确认且仍存在于 users 表中的可信 UUID。

        Raises:
            ChatSessionNotFoundError: 会话不存在或属于其他用户。两种情况故意使用
                相同错误，避免通过删除入口枚举其他用户资源。
            ChatThreadBusyError: 相同内部 thread 正在执行 Graph 或另一次清理。
            ChatExecutionGuardUnavailableError: Redis 无法保证安全互斥。
            ChatCheckpointCleanupError: saver 删除失败。此时业务行保持 deleting，
                后续调用可从第二阶段继续。
            ChatSessionCleanupStateError: 数据库出现 coordinator 不认识的状态。

        Returns:
            没有返回值。方法正常结束表示业务会话行及其 checkpoint 已完成清理。
        """
        # 阶段 0：只用可信 user_id 和业务 session_id 构造内部 key。
        # 这个工厂与 ChatService 共用，因此 guard 锁住的 key 和 saver 删除的 key
        # 正是 Graph 执行时使用的 configurable.thread_id。
        internal_thread_id = self._internal_thread_id_factory(
            user_id,
            session_id,
        )

        # guard 覆盖阶段 1、2、3，而不仅覆盖 adelete_thread。否则 Graph 可能在
        # 业务行标记 deleting 与 checkpoint 删除之间写入一个新版本，造成“刚删完
        # 又出现新 checkpoint”的竞态。async with 还保证正常返回、异常和取消时
        # 都会执行 guard 的退出逻辑。
        async with self._execution_guard.hold(internal_thread_id):
            # 阶段 1：先提交 durable deletion intent。
            # 这次调用返回时 deleting 已经 commit，不依赖后续 Python 代码继续运行。
            await self._mark_deleting(
                session_id=session_id,
                user_id=user_id,
            )

            # deleting 一旦提交，普通 list_by_user 就已隐藏该会话，因此必须在
            # 这个可见性变化点失效列表缓存，不能等到 checkpoint 和业务 tombstone
            # 全部删除。失效采用 fail-open，不改变三阶段恢复状态和后续重试方向。
            await invalidate_chat_session_list_cache(
                self._cache,
                user_id=user_id,
                reason="deleting",
            )

            try:
                # 阶段 2：清理执行层状态。
                # 当前 saver 依次删除 checkpoints、blobs 和 writes。即使底层在
                # 中途只完成部分 DELETE，再按同一 thread_id 调用仍然安全：删除
                # 已不存在的行不会重新创建数据，这就是本阶段依赖的幂等性。
                await self._checkpoint_store.adelete_thread(internal_thread_id)
            except Exception as error:
                # 不把 deleting 回滚为 active：阶段 1 已独立提交，而 tombstone 是
                # 故障恢复所需的持久证据。下一次删除会重新进入阶段 1，并因状态已经
                # 是 deleting 而直接继续。底层异常仅通过 __cause__ 留给服务端诊断，
                # 对外错误文本不包含 SQL、连接串或内部 thread key。
                raise ChatCheckpointCleanupError from error

            # 阶段 3：只有 checkpoint 删除成功后，才物理删除业务 tombstone。
            # 这个顺序避免“业务行先消失，后续却失去 owner-scoped 重试入口”。
            await self._delete_marked_session(
                session_id=session_id,
                user_id=user_id,
            )

    async def _mark_deleting(
        self,
        *,
        session_id: UUID,
        user_id: UUID,
    ) -> None:
        """在独立 ORM 事务中持久化删除意图.

        已经是 deleting 时不回退为 active，而是直接进入下一阶段；这正是失败后
        重试能够继续工作的关键。``session.begin()`` 正常退出时提交，异常退出时
        回滚，因此调用方只有在本方法正常返回后才能假设 tombstone 已持久化。
        """
        async with self._session_factory() as session:
            async with session.begin():
                repository = ChatSessionRepository(session)
                # get_for_cleanup 同时使用 session_id + user_id，并执行 SELECT FOR
                # UPDATE。owner 过滤避免越权，行锁避免两个数据库清理事务同时修改
                # 同一个业务行；跨进程的完整工作流互斥仍由外层 Redis guard 负责。
                chat_session = await repository.get_for_cleanup(
                    session_id,
                    user_id=user_id,
                )
                if chat_session is None:
                    raise ChatSessionNotFoundError

                await repository.mark_deleting(chat_session)

    async def _delete_marked_session(
        self,
        *,
        session_id: UUID,
        user_id: UUID,
    ) -> None:
        """Checkpoint 清理成功后，在新事务中删除业务 tombstone.

        该事务与阶段 1 分离，因为阶段 2 使用 saver 的另一条连接。方法正常返回时，
        ``session.begin()`` 已提交物理删除；异常时只回滚本阶段，deleting 仍可重试。
        """
        async with self._session_factory() as session:
            async with session.begin():
                repository = ChatSessionRepository(session)
                chat_session = await repository.get_for_cleanup(
                    session_id,
                    user_id=user_id,
                )

                # 正常 guard 路径中该行应存在。若之前的阶段 3 已经提交，但调用方
                # 在收到成功结果前断线，重试时这里会查不到行。目标状态其实已经
                # 满足，因此按成功处理比重新制造 tombstone 或报告内部错误更安全。
                if chat_session is None:
                    return

                if chat_session.status != ChatSessionStatus.DELETING:
                    raise ChatSessionCleanupStateError("Chat session must be deleting before final cleanup")

                await repository.delete(chat_session)


__all__ = [
    "ChatCheckpointCleanupError",
    "ChatCheckpointStore",
    "ChatSessionCleanupService",
    "ChatSessionCleanupStateError",
    "InternalThreadIdFactory",
]
