"""验证 ChatSession 与 LangGraph checkpoint 的可重试清理流程.

本 smoke 不发起模型请求。它用最小 LangGraph 节点向真实 PostgreSQL saver 写入
checkpoint，再注入一次 checkpoint 删除故障，验证：

1. 业务行先提交 ``active -> deleting``；
2. 普通查询和 Agent ownership 立即隐藏 deleting 会话；
3. 故障后 Redis guard 释放，checkpoint 暂时保留；
4. 新的 production cleanup service 根据持久状态完成重试；
5. 目标 thread 的 checkpoints、blobs、writes 全部清空；
6. 其他会话 checkpoint 不变，ResearchTask 外键被置为 NULL；
7. cross-user、absent 和重复删除共享安全 not-found 语义。

模型决策没有发生，因此不需要也不允许 fake LLM。10F-F 会在此机制证据之上，
再使用真实 provider 验证完整 HTTP/Agent 删除闭环。
"""

import asyncio
import json
import os
import selectors
from hashlib import sha256
from pathlib import Path
from time import perf_counter
from typing import TypedDict
from uuid import UUID, uuid4

import psycopg
from alembic import command
from alembic.config import Config
from fastapi import FastAPI
from langgraph.graph import END, START, StateGraph
from psycopg import sql
from psycopg.conninfo import make_conninfo

from app.core.config import Settings, settings
from app.infrastructure.chat_guard import (
    CHAT_EXECUTION_LOCK_PREFIX,
    RedisChatExecutionGuard,
)
from app.infrastructure.database import build_orm_database_url
from app.infrastructure.lifespan import (
    CHAT_GUARD_LEASE_SECONDS,
    get_application_cache,
    get_application_chat_cleanup_service,
    get_application_resources,
    lifespan,
)
from app.infrastructure.resources import ApplicationResources
from app.models import (
    ChatSession,
    ChatSessionStatus,
    ResearchTask,
    ResearchTaskStatus,
    User,
)
from app.repositories import ChatSessionRepository
from app.services.chat import ChatService
from app.services.chat_session_cleanup import (
    ChatCheckpointCleanupError,
    ChatCheckpointStore,
    ChatSessionCleanupService,
)
from app.services.chat_session_ownership import ChatSessionNotFoundError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONNECTION_TIMEOUT_SECONDS = 10
TOTAL_TIMEOUT_SECONDS = 120.0
CHECKPOINT_TABLES = (
    "checkpoints",
    "checkpoint_blobs",
    "checkpoint_writes",
)


class CleanupProbeState(TypedDict):
    """最小 LangGraph 状态，只用于产生真实 checkpoint 行."""

    marker: str


class FailBeforeCheckpointDelete:
    """第一次调用时模拟 saver 故障，且不伪造任何成功结果."""

    def __init__(self, delegate: ChatCheckpointStore) -> None:
        """保存真实 saver，并初始化调用计数.

        Args:
            delegate: lifespan 已 setup 的真实 ``AsyncPostgresSaver``。
        """
        self._delegate = delegate
        self.call_count = 0

    async def adelete_thread(self, thread_id: str) -> None:
        """第一次抛出受控故障，后续调用委托真实 saver.

        Args:
            thread_id: cleanup coordinator 构造的内部 checkpoint key。

        Raises:
            RuntimeError: 第一次调用固定抛出，用于验证 deleting 持久状态。
        """
        self.call_count += 1
        if self.call_count == 1:
            raise RuntimeError("injected checkpoint cleanup failure")
        await self._delegate.adelete_thread(thread_id)


def _elapsed_ms(started_at: float) -> float:
    """返回适合安全摘要的单调时钟毫秒数."""
    return round((perf_counter() - started_at) * 1000, 2)


def _conninfo(database: str) -> str:
    """构造只交给 psycopg 使用且绝不输出的连接信息."""
    return make_conninfo(
        host=settings.POSTGRES_HOST,
        port=settings.POSTGRES_PORT,
        dbname=database,
        user=settings.POSTGRES_USER,
        password=settings.POSTGRES_PASSWORD,
        connect_timeout=CONNECTION_TIMEOUT_SECONDS,
    )


def _temporary_database_url(database: str) -> str:
    """构造 Alembic 迁移随机数据库时使用的临时 URL."""
    return build_orm_database_url(settings).set(database=database).render_as_string(hide_password=False)


def _create_database(admin_database: str, test_database: str) -> None:
    """使用 autocommit 管理连接创建随机数据库."""
    with psycopg.connect(_conninfo(admin_database), autocommit=True) as connection:
        connection.execute(
            sql.SQL("CREATE DATABASE {}").format(sql.Identifier(test_database)),
        )


def _drop_database(admin_database: str, test_database: str) -> None:
    """终止残留连接并只删除当前 smoke 创建的数据库."""
    with psycopg.connect(_conninfo(admin_database), autocommit=True) as connection:
        connection.execute(
            """
            SELECT pg_terminate_backend(pid)
            FROM pg_stat_activity
            WHERE datname = %s AND pid <> pg_backend_pid()
            """,
            (test_database,),
        )
        connection.execute(
            sql.SQL("DROP DATABASE IF EXISTS {}").format(
                sql.Identifier(test_database),
            )
        )


def _runtime_settings(database: str) -> Settings:
    """复制真实基础设施配置，仅替换随机 PostgreSQL 数据库名."""
    config = Settings()
    config.POSTGRES_DB = database
    config.POSTGRES_PSYCOPG_POOL_SIZE = 3
    config.POSTGRES_ORM_POOL_SIZE = 2
    config.POSTGRES_ORM_MAX_OVERFLOW = 0
    return config


def _selector_loop_factory() -> asyncio.AbstractEventLoop:
    """创建 Windows psycopg 异步连接要求的 Selector event loop."""
    return asyncio.SelectorEventLoop(selectors.SelectSelector())


def _persist_marker(state: CleanupProbeState) -> CleanupProbeState:
    """原样返回 marker，使最小 Graph 产生一次真实状态更新."""
    return {"marker": state["marker"]}


def _build_probe_graph(resources: ApplicationResources):  # noqa: ANN202
    """构建不含 LLM 的最小 Graph，并绑定 production PostgreSQL saver.

    Args:
        resources: lifespan 已 setup 的共享基础设施资源。

    Returns:
        只执行一个确定性节点、但使用真实 checkpointer 的 compiled graph。
    """
    builder = StateGraph(CleanupProbeState)
    builder.add_node("persist", _persist_marker)
    builder.add_edge(START, "persist")
    builder.add_edge("persist", END)
    return builder.compile(checkpointer=resources.checkpointer)


def _internal_thread_id(user_id: UUID, session_id: UUID) -> str:
    """复用 ChatService 的唯一内部 key 映射，供 smoke 查询状态."""
    return ChatService._build_checkpoint_thread_id(user_id, session_id)


def _lock_name(internal_thread_id: str) -> str:
    """把内部 key 转换成与 production guard 相同的 Redis 摘要 key."""
    digest = sha256(internal_thread_id.encode("utf-8")).hexdigest()
    return f"{CHAT_EXECUTION_LOCK_PREFIX}{digest}"


async def _checkpoint_row_counts(
    resources: ApplicationResources,
    *,
    internal_thread_id: str,
) -> dict[str, int]:
    """读取目标内部 thread 在三张 saver 表中的行数.

    表名来自本文件固定白名单并使用 ``sql.Identifier``，thread ID 则始终作为
    参数传递。这里不会把动态字符串拼进 SQL。
    """
    counts: dict[str, int] = {}
    async with resources.postgres_pool.connection() as connection:
        async with connection.cursor() as cursor:
            for table_name in CHECKPOINT_TABLES:
                await cursor.execute(
                    sql.SQL("SELECT COUNT(*) AS row_count FROM {} WHERE thread_id = %s").format(
                        sql.Identifier(table_name)
                    ),
                    (internal_thread_id,),
                )
                row = await cursor.fetchone()
                if row is None:
                    raise RuntimeError("checkpoint count query returned no row")
                counts[table_name] = int(row["row_count"])
    return counts


async def _exercise_cleanup(database: str) -> dict[str, bool | float | int]:
    """在一代 production lifespan 中执行故障与恢复验收.

    Args:
        database: 已迁移到 Alembic head 的随机数据库。

    Returns:
        只包含布尔值、计数和耗时的脱敏摘要。
    """
    started_at = perf_counter()
    owner_id = UUID("77777777-7777-4777-8777-777777777777")
    other_user_id = UUID("88888888-8888-4888-8888-888888888888")
    target_session_id = uuid4()
    untouched_session_id = uuid4()
    absent_session_id = uuid4()
    target_internal_id = _internal_thread_id(owner_id, target_session_id)
    untouched_internal_id = _internal_thread_id(owner_id, untouched_session_id)

    app = FastAPI()
    async with asyncio.timeout(TOTAL_TIMEOUT_SECONDS):
        async with lifespan(app, config=_runtime_settings(database)):
            resources = get_application_resources(app)
            production_cleanup = get_application_chat_cleanup_service(app)
            cleanup_service_is_shared = production_cleanup is get_application_chat_cleanup_service(app)

            # 先持久化真实用户、会话和一条引用目标会话的 ResearchTask。显式
            # flush 父行可以让外键写入顺序不依赖 ORM relationship。
            async with resources.orm_session_factory() as session:
                async with session.begin():
                    session.add_all(
                        [
                            User(
                                id=owner_id,
                                email=f"cleanup-owner-{uuid4().hex}@example.com",
                                password_hash="smoke-only-not-a-real-credential",
                            ),
                            User(
                                id=other_user_id,
                                email=f"cleanup-other-{uuid4().hex}@example.com",
                                password_hash="smoke-only-not-a-real-credential",
                            ),
                        ]
                    )
                    await session.flush()

                    target_session = ChatSession(
                        id=target_session_id,
                        user_id=owner_id,
                        title="Cleanup target",
                    )
                    untouched_session = ChatSession(
                        id=untouched_session_id,
                        user_id=owner_id,
                        title="Cleanup control",
                    )
                    session.add_all([target_session, untouched_session])
                    await session.flush()

                    research_task = ResearchTask(
                        user_id=owner_id,
                        chat_session_id=target_session_id,
                        topic="Cleanup foreign-key probe",
                        status=ResearchTaskStatus.PENDING,
                    )
                    session.add(research_task)
                    await session.flush()
                    research_task_id = research_task.id

            # 最小 Graph 只负责通过 LangGraph 公共 API 写真实 checkpoint。marker
            # 不包含用户数据，也不会进入模型或外部网络。
            graph = _build_probe_graph(resources)
            await graph.ainvoke(
                {"marker": "target"},
                ChatService._build_config(
                    user_id=owner_id,
                    public_thread_id=target_session_id,
                ),
            )
            await graph.ainvoke(
                {"marker": "untouched"},
                ChatService._build_config(
                    user_id=owner_id,
                    public_thread_id=untouched_session_id,
                ),
            )

            target_counts_before = await _checkpoint_row_counts(
                resources,
                internal_thread_id=target_internal_id,
            )
            untouched_counts_before = await _checkpoint_row_counts(
                resources,
                internal_thread_id=untouched_internal_id,
            )
            both_threads_have_checkpoint_rows = all(count > 0 for count in target_counts_before.values()) and all(
                count > 0 for count in untouched_counts_before.values()
            )

            # 不存在和跨用户必须在写 deleting 前失败。production service 使用
            # 真实 PostgreSQL、Redis 与 saver；这里没有替换任何成功分支。
            try:
                await production_cleanup.delete_owned(
                    session_id=target_session_id,
                    user_id=other_user_id,
                )
            except ChatSessionNotFoundError:
                cross_user_rejected = True
            else:
                cross_user_rejected = False

            try:
                await production_cleanup.delete_owned(
                    session_id=absent_session_id,
                    user_id=owner_id,
                )
            except ChatSessionNotFoundError:
                absent_rejected = True
            else:
                absent_rejected = False

            # 故障 service 仍使用真实 ORM 与 Redis，仅把第一次 checkpoint 删除
            # 替换为明确失败。它验证的不是 saver 实现，而是 coordinator 的恢复语义。
            fail_before_delete = FailBeforeCheckpointDelete(resources.checkpointer)
            fault_service = ChatSessionCleanupService(
                session_factory=resources.orm_session_factory,
                checkpoint_store=fail_before_delete,
                execution_guard=RedisChatExecutionGuard(
                    resources.redis_client,
                    lease_seconds=CHAT_GUARD_LEASE_SECONDS,
                ),
                internal_thread_id_factory=ChatService._build_checkpoint_thread_id,
                cache=get_application_cache(app),
            )

            try:
                await fault_service.delete_owned(
                    session_id=target_session_id,
                    user_id=owner_id,
                )
            except ChatCheckpointCleanupError:
                injected_failure_observed = True
            else:
                injected_failure_observed = False

            async with resources.orm_session_factory() as session:
                async with session.begin():
                    repository = ChatSessionRepository(session)
                    cleanup_row = await repository.get_for_cleanup(
                        target_session_id,
                        user_id=owner_id,
                    )
                    ordinary_row = await repository.get_by_id(
                        target_session_id,
                        user_id=owner_id,
                    )
            deleting_state_persisted = (
                cleanup_row is not None and cleanup_row.status == ChatSessionStatus.DELETING and ordinary_row is None
            )

            counts_after_failure = await _checkpoint_row_counts(
                resources,
                internal_thread_id=target_internal_id,
            )
            checkpoint_preserved_after_injected_failure = counts_after_failure == target_counts_before
            failure_lock_released = not bool(await resources.redis_client.exists(_lock_name(target_internal_id)))

            # 使用 lifespan 发布的 production service 重试。新的 service 实例没有
            # 上一次调用的内存信息，只能依靠数据库 deleting 状态恢复。
            await production_cleanup.delete_owned(
                session_id=target_session_id,
                user_id=owner_id,
            )

            target_counts_after = await _checkpoint_row_counts(
                resources,
                internal_thread_id=target_internal_id,
            )
            untouched_counts_after = await _checkpoint_row_counts(
                resources,
                internal_thread_id=untouched_internal_id,
            )

            async with resources.orm_session_factory() as session:
                async with session.begin():
                    repository = ChatSessionRepository(session)
                    target_after = await repository.get_for_cleanup(
                        target_session_id,
                        user_id=owner_id,
                    )
                    untouched_after = await repository.get_by_id(
                        untouched_session_id,
                        user_id=owner_id,
                    )
                    task_after = await session.get(ResearchTask, research_task_id)

            target_business_row_deleted = target_after is None
            target_checkpoint_rows_deleted = all(count == 0 for count in target_counts_after.values())
            untouched_session_preserved = (
                untouched_after is not None and untouched_counts_after == untouched_counts_before
            )
            research_task_reference_cleared = task_after is not None and task_after.chat_session_id is None

            # 物理删除后的再次调用与任意不存在 UUID 使用相同 not-found。数据库和
            # checkpoint 的最终状态不再变化，因此操作结果仍满足幂等性。
            try:
                await production_cleanup.delete_owned(
                    session_id=target_session_id,
                    user_id=owner_id,
                )
            except ChatSessionNotFoundError:
                repeated_delete_is_safe_not_found = True
            else:
                repeated_delete_is_safe_not_found = False

            final_lock_released = not bool(await resources.redis_client.exists(_lock_name(target_internal_id)))

        state_removed_after_shutdown = (
            not hasattr(app.state, "resources")
            and not hasattr(app.state, "chat_service")
            and not hasattr(app.state, "chat_session_cleanup_service")
        )
        postgres_pool_closed_after_shutdown = resources.postgres_pool.closed

    elapsed_ms = _elapsed_ms(started_at)
    return {
        "cleanup_service_is_shared": cleanup_service_is_shared,
        "both_threads_have_checkpoint_rows": both_threads_have_checkpoint_rows,
        "cross_user_rejected": cross_user_rejected,
        "absent_rejected": absent_rejected,
        "injected_failure_observed": injected_failure_observed,
        "deleting_state_persisted": deleting_state_persisted,
        "checkpoint_preserved_after_injected_failure": (checkpoint_preserved_after_injected_failure),
        "failure_lock_released": failure_lock_released,
        "target_business_row_deleted": target_business_row_deleted,
        "target_checkpoint_rows_deleted": target_checkpoint_rows_deleted,
        "untouched_session_preserved": untouched_session_preserved,
        "research_task_reference_cleared": research_task_reference_cleared,
        "repeated_delete_is_safe_not_found": repeated_delete_is_safe_not_found,
        "final_lock_released": final_lock_released,
        "state_removed_after_shutdown": state_removed_after_shutdown,
        "postgres_pool_closed_after_shutdown": postgres_pool_closed_after_shutdown,
        "fault_store_call_count": fail_before_delete.call_count,
        "target_rows_before": sum(target_counts_before.values()),
        "target_rows_after": sum(target_counts_after.values()),
        "untouched_rows_after": sum(untouched_counts_after.values()),
        "within_total_budget": elapsed_ms <= TOTAL_TIMEOUT_SECONDS * 1000,
        "elapsed_ms": elapsed_ms,
    }


def _run_smoke() -> dict[str, object]:
    """迁移随机数据库、运行异步 Gate 并保证最终清理."""
    started_at = perf_counter()
    admin_database = settings.POSTGRES_DB
    test_database = f"deep_research_chat_cleanup_{uuid4().hex[:10]}"
    previous_override = os.environ.get("ALEMBIC_DATABASE_URL")
    database_created = False
    cleanup_ok = False
    checks: dict[str, bool | float | int]

    try:
        _create_database(admin_database, test_database)
        database_created = True
        os.environ["ALEMBIC_DATABASE_URL"] = _temporary_database_url(test_database)
        command.upgrade(Config(str(PROJECT_ROOT / "alembic.ini")), "head")
        checks = asyncio.run(
            _exercise_cleanup(test_database),
            loop_factory=_selector_loop_factory,
        )
    finally:
        if previous_override is None:
            os.environ.pop("ALEMBIC_DATABASE_URL", None)
        else:
            os.environ["ALEMBIC_DATABASE_URL"] = previous_override

        if database_created:
            try:
                _drop_database(admin_database, test_database)
            except Exception:
                cleanup_ok = False
            else:
                cleanup_ok = True

    if database_created and not cleanup_ok:
        raise RuntimeError("temporary Chat cleanup database cleanup failed")

    boolean_checks = tuple(value for value in checks.values() if isinstance(value, bool))
    ok = bool(boolean_checks) and all(boolean_checks) and cleanup_ok
    return {
        "ok": ok,
        **checks,
        "cleanup_ok": cleanup_ok,
        "total_elapsed_ms": _elapsed_ms(started_at),
    }


def main() -> int:
    """打印脱敏单行 JSON，并返回 shell 可判断的退出码."""
    started_at = perf_counter()
    try:
        summary = _run_smoke()
    except Exception as error:
        # 数据库与 Redis 异常可能带连接信息，因此顶层只公开异常类型。
        print(
            json.dumps(
                {
                    "ok": False,
                    "error_type": type(error).__name__,
                    "elapsed_ms": _elapsed_ms(started_at),
                }
            )
        )
        return 1

    print(json.dumps(summary))
    return 0 if summary["ok"] is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
