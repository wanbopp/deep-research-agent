"""端到端验证 production Chat ownership、execution guard 与真实 provider.

本 Gate 覆盖 Checkpoint 10F-C4 最关键的应用行为：

1. 随机 PostgreSQL 数据库执行正式 Alembic migration，再启动 production lifespan；
2. owner 请求通过 PostgreSQL verifier，并由真实 provider 完成 Agent 回答；
3. cross-user 与 absent 请求在 Graph 前失败，且真实 Redis 锁仍由 finally 释放；
4. 第一个同 session 请求取得真实 Redis 锁后，第二个请求必须 fail-fast busy；
5. busy 请求不能写 checkpoint，不同 session 仍可并发完成真实模型请求；
6. SSE ownership/busy 分支返回稳定 error 事件，不调用模型；
7. shutdown 后连接池关闭、app.state 撤下，随机数据库和 Redis 锁均被清理。

所有会产生模型行为的路径都使用当前真实 provider。脚本不会输出 Prompt、模型正文、
用户 UUID、公开/内部 thread ID、Redis key、owner token、数据库名、连接串或 API key。
"""

import asyncio
import json
import os
import selectors
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from hashlib import sha256
from pathlib import Path
from time import perf_counter
from unittest.mock import patch
from uuid import UUID, uuid4

import psycopg
from alembic import command
from alembic.config import Config
from fastapi import FastAPI
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from psycopg import sql
from psycopg.conninfo import make_conninfo
from redis.asyncio import Redis

import app.infrastructure.lifespan as lifespan_module
from app.agents.chat.graph import ChatGraph
from app.agents.chat.runtime import create_chat_runtime
from app.core.config import Settings, settings
from app.infrastructure.database import build_orm_database_url
from app.infrastructure.chat_guard import (
    CHAT_EXECUTION_LOCK_PREFIX,
    RedisChatExecutionGuard,
)
from app.infrastructure.lifespan import (
    CHAT_GUARD_LEASE_SECONDS,
    get_application_chat_service,
    get_application_resources,
    lifespan,
)
from app.models.chat_session import ChatSession
from app.models.user import User
from app.schemas.chat import ChatRequest, ChatResponse, ErrorStreamEvent
from app.services.chat import ChatService
from app.services.chat_guard import ChatThreadBusyError
from app.services.chat_session_ownership import ChatSessionNotFoundError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONNECTION_TIMEOUT_SECONDS = 10
TOTAL_TIMEOUT_SECONDS = 300.0
LOCK_OBSERVATION_TIMEOUT_SECONDS = 15.0
BUSY_BUDGET_SECONDS = 0.5

FIRST_EXPECTED_REPLY = "REAL_GUARD_PRIMARY_OK"
DIFFERENT_FIRST_REPLY = "REAL_GUARD_PARALLEL_A_OK"
DIFFERENT_SECOND_REPLY = "REAL_GUARD_PARALLEL_B_OK"


def _elapsed_ms(started_at: float) -> float:
    """返回毫秒耗时，不记录基础设施或模型内容."""
    return round((perf_counter() - started_at) * 1000, 2)


def _conninfo(database: str) -> str:
    """构造只交给 psycopg 使用、绝不打印的连接串.

    Args:
        database: 管理数据库或本次随机临时数据库名称。

    Returns:
        正确转义密码和连接参数的 psycopg conninfo。
    """
    return make_conninfo(
        host=settings.POSTGRES_HOST,
        port=settings.POSTGRES_PORT,
        dbname=database,
        user=settings.POSTGRES_USER,
        password=settings.POSTGRES_PASSWORD,
        connect_timeout=CONNECTION_TIMEOUT_SECONDS,
    )


def _create_database(admin_database: str, test_database: str) -> None:
    """在事务外创建随机临时数据库."""
    with psycopg.connect(_conninfo(admin_database), autocommit=True) as connection:
        connection.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(test_database)))


def _temporary_database_url(database: str) -> str:
    """构造只交给 Alembic 使用且不会输出的临时数据库 URL.

    Args:
        database: 本 Gate 随机创建的数据库名称。

    Returns:
        指向随机数据库的 SQLAlchemy URL 字符串。
    """
    return build_orm_database_url(settings).set(database=database).render_as_string(hide_password=False)


def _drop_database(admin_database: str, test_database: str) -> None:
    """终止残留连接，并且只删除本 Gate 创建的随机数据库."""
    with psycopg.connect(_conninfo(admin_database), autocommit=True) as connection:
        connection.execute(
            """
            SELECT pg_terminate_backend(pid)
            FROM pg_stat_activity
            WHERE datname = %s AND pid <> pg_backend_pid()
            """,
            (test_database,),
        )
        connection.execute(sql.SQL("DROP DATABASE IF EXISTS {}").format(sql.Identifier(test_database)))


def _runtime_settings(database: str) -> Settings:
    """复制真实配置，只把 PostgreSQL 切换到随机数据库.

    Args:
        database: 本次 Gate 的随机临时数据库。

    Returns:
        保留真实 provider、Redis 和其他服务配置的 Settings 副本。
    """
    config = Settings()
    config.POSTGRES_DB = database
    # 两个不同 thread 会并发写 checkpoint，需要保留足够的 saver 连接预算。
    config.POSTGRES_PSYCOPG_POOL_SIZE = 5
    config.POSTGRES_ORM_POOL_SIZE = 1
    config.POSTGRES_ORM_MAX_OVERFLOW = 0
    return config


def _selector_loop_factory() -> asyncio.AbstractEventLoop:
    """创建 Windows 下 psycopg 异步连接要求的 Selector event loop."""
    return asyncio.SelectorEventLoop(selectors.SelectSelector())


def _internal_thread_id(user_id: UUID, public_thread_id: UUID) -> str:
    """复现 ChatService 的可信 checkpoint 身份映射，仅供 Gate 查询状态."""
    return f"user:{user_id.hex}:thread:{public_thread_id}"


def _lock_name(user_id: UUID, public_thread_id: UUID) -> str:
    """计算期望的摘要锁名，返回值只交给 Redis exists，不进入输出."""
    internal_thread_id = _internal_thread_id(user_id, public_thread_id)
    digest = sha256(internal_thread_id.encode("utf-8")).hexdigest()
    return f"{CHAT_EXECUTION_LOCK_PREFIX}{digest}"


async def _wait_for_lock_state(
    redis_client: Redis,
    *,
    lock_names: tuple[str, ...],
    expected_exists: bool,
) -> None:
    """轮询真实 Redis，直到指定锁全部达到期望状态.

    Args:
        redis_client: lifespan 共享的异步 Redis client。
        lock_names: 只在内存中使用的摘要锁名。
        expected_exists: True 表示等待全部存在，False 表示等待全部消失。

    Raises:
        TimeoutError: 在预算内没有观察到期望锁状态。
    """
    deadline = asyncio.get_running_loop().time() + LOCK_OBSERVATION_TIMEOUT_SECONDS
    while asyncio.get_running_loop().time() < deadline:
        states = [bool(await redis_client.exists(lock_name)) for lock_name in lock_names]
        if all(state is expected_exists for state in states):
            return
        await asyncio.sleep(0.01)

    raise TimeoutError("expected Redis lock state was not observed")


def _response_matches(result: object, expected: str) -> bool:
    """判断应用结果是否为指定真实模型固定回复，不输出正文."""
    return isinstance(result, ChatResponse) and result.message.content.strip() == expected


async def _exercise_guard(database: str) -> dict[str, bool | float | str]:
    """在一代 production lifespan 中执行真实 guard Gate.

    Args:
        database: 随机临时 PostgreSQL 数据库。

    Returns:
        仅包含模型名、布尔判定和耗时的安全摘要。

    Raises:
        Exception: provider、Redis、PostgreSQL、Graph 或生命周期错误。顶层只输出
            异常类型，并仍执行数据库清理。
    """
    started_at = perf_counter()
    user_id = UUID("55555555-5555-4555-8555-555555555555")
    other_user_id = UUID("66666666-6666-4666-8666-666666666666")
    # 公开会话标识已经是业务实体 UUID。随机值既避免与其他 smoke 冲突，
    # 也让类型层尽早发现把任意名称字符串当作 session_id 的旧用法。
    same_thread = uuid4()
    parallel_first_thread = uuid4()
    parallel_second_thread = uuid4()
    stream_busy_thread = uuid4()
    absent_thread = uuid4()
    rejected_marker = f"REJECTED-{uuid4().hex.upper()}"

    # 这两个 Event 只负责让 smoke 获得确定的并发顺序：第一个请求拿到真实 Redis
    # 锁后先停住，第二个请求完成 busy 验证后再放行。它们不替换 Graph 或模型。
    first_guard_entered = asyncio.Event()
    allow_first_graph = asyncio.Event()
    same_internal_thread_id = _internal_thread_id(user_id, same_thread)

    first_prompt = f"Do not call any tool. Reply with exactly {FIRST_EXPECTED_REPLY} and nothing else."
    rejected_prompt = f"This request must be rejected before Graph execution. Marker: {rejected_marker}"
    parallel_first_prompt = f"Do not call any tool. Reply with exactly {DIFFERENT_FIRST_REPLY} and nothing else."
    parallel_second_prompt = f"Do not call any tool. Reply with exactly {DIFFERENT_SECOND_REPLY} and nothing else."
    ownership_probe_prompt = "This ownership probe must never reach the Graph or model."

    captured_graphs: list[ChatGraph] = []

    class ObservedRedisChatExecutionGuard(RedisChatExecutionGuard):
        """在真实 Redis guard 外增加一次性 smoke 同步屏障.

        生产代码仍执行父类的真实锁获取与 owner-token 释放。屏障位于父类成功
        ``yield`` 之后、ChatService 进入 Graph 之前，因此可确定地证明：第二个
        同 key 请求被拒绝时，第一个请求确实持有执行权，而且尚未调用模型。
        """

        def __init__(
            self,
            redis_client: Redis,
            *,
            lease_seconds: float,
        ) -> None:
            """保存真实 Redis 配置，并初始化一次性观察状态.

            Args:
                redis_client: lifespan 创建的真实异步 Redis client。
                lease_seconds: 与生产装配相同的锁租约秒数。
            """
            super().__init__(
                redis_client,
                lease_seconds=lease_seconds,
            )
            self._same_thread_observed = False

        @asynccontextmanager
        async def hold(self, internal_thread_id: str) -> AsyncIterator[None]:
            """委托真实锁，并仅暂停第一个目标 thread 请求.

            Args:
                internal_thread_id: ChatService 构造的可信内部 checkpoint key。

            Yields:
                放行后把控制权交回 ChatService，使其继续执行真实 Graph。
            """
            async with super().hold(internal_thread_id):
                if internal_thread_id == same_internal_thread_id and not self._same_thread_observed:
                    self._same_thread_observed = True
                    first_guard_entered.set()
                    await allow_first_graph.wait()
                yield

    def capture_runtime(
        *,
        checkpointer: BaseCheckpointSaver[str] | None = None,
    ) -> ChatGraph:
        """捕获 lifespan 构建的真实 Graph，同时保持 production 装配路径.

        Args:
            checkpointer: lifespan 已 setup 的真实 PostgreSQL saver。

        Returns:
            使用真实 provider、tools 和传入 saver 编译的 ChatGraph。
        """
        graph = create_chat_runtime(checkpointer=checkpointer)
        captured_graphs.append(graph)
        return graph

    app = FastAPI()
    async with asyncio.timeout(TOTAL_TIMEOUT_SECONDS):
        # 两个 patch 都只增加可观测性：Graph factory 仍调用生产构造函数；guard
        # 子类仍委托真实 Redis 实现。所有模型行为继续使用当前真实 provider。
        with (
            patch.object(lifespan_module, "create_chat_runtime", capture_runtime),
            patch.object(
                lifespan_module,
                "RedisChatExecutionGuard",
                ObservedRedisChatExecutionGuard,
            ),
        ):
            async with lifespan(app, config=_runtime_settings(database)):
                service = get_application_chat_service(app)
                resources = get_application_resources(app)

                # Alembic 已经建立业务表；这里仅写入两个真实用户和 A 拥有的
                # session。B 用户存在但不拥有这些 session，absent_thread 则完全
                # 没有业务行，因而可验证两种情况共享相同的安全拒绝语义。
                async with resources.orm_session_factory() as session:
                    session.add_all(
                        [
                            User(
                                id=user_id,
                                email=f"guard-owner-{uuid4().hex}@example.com",
                                password_hash="smoke-only-not-a-real-credential",
                            ),
                            User(
                                id=other_user_id,
                                email=f"guard-other-{uuid4().hex}@example.com",
                                password_hash="smoke-only-not-a-real-credential",
                            ),
                        ]
                    )

                    # User 与 ChatSession 之间只声明数据库 foreign key，没有配置
                    # ORM relationship。显式 flush 用户可保证外键父行先落库，也让
                    # fixture 构建顺序不依赖 SQLAlchemy 对无关系对象的排序细节。
                    await session.flush()

                    session.add_all(
                        [
                            ChatSession(
                                id=session_id,
                                user_id=user_id,
                                title="Execution guard smoke",
                            )
                            for session_id in (
                                same_thread,
                                parallel_first_thread,
                                parallel_second_thread,
                                stream_busy_thread,
                            )
                        ]
                    )
                    await session.commit()

                if len(captured_graphs) != 1:
                    raise RuntimeError("lifespan must build exactly one Chat graph")
                graph = captured_graphs[0]

                # 先验证 ownership 的拒绝路径。两个 patch 只是调用计数器：若
                # verifier 正确位于 Graph 前，它们的 call_count 必须始终为 0。
                # 这里不替换 Graph 返回值，也不使用 fake LLM；授权路径稍后仍会
                # 通过同一个真实 Graph 发起真实 provider 请求。
                cross_user_lock = _lock_name(other_user_id, same_thread)
                absent_lock = _lock_name(user_id, absent_thread)
                with (
                    patch.object(graph, "ainvoke", wraps=graph.ainvoke) as ownership_ainvoke_spy,
                    patch.object(graph, "astream", wraps=graph.astream) as ownership_astream_spy,
                ):
                    try:
                        await service.run_turn(
                            ChatRequest(
                                thread_id=same_thread,
                                message=ownership_probe_prompt,
                            ),
                            user_id=other_user_id,
                        )
                    except ChatSessionNotFoundError:
                        cross_user_rejected = True
                    else:
                        cross_user_rejected = False

                    try:
                        await service.run_turn(
                            ChatRequest(
                                thread_id=absent_thread,
                                message=ownership_probe_prompt,
                            ),
                            user_id=user_id,
                        )
                    except ChatSessionNotFoundError:
                        absent_session_rejected = True
                    else:
                        absent_session_rejected = False

                    # StreamingResponse 不能在生成器开始后改 HTTP status，因此
                    # stream_turn 把相同业务错误转换为稳定的流内 error event。
                    ownership_stream_events: list[ErrorStreamEvent] = []
                    async for event in service.stream_turn(
                        ChatRequest(
                            thread_id=same_thread,
                            message=ownership_probe_prompt,
                        ),
                        user_id=other_user_id,
                    ):
                        if isinstance(event, ErrorStreamEvent):
                            ownership_stream_events.append(event)

                    ownership_failures_skipped_graph = (
                        ownership_ainvoke_spy.call_count == 0 and ownership_astream_spy.call_count == 0
                    )

                ownership_stream_error_matches = (
                    len(ownership_stream_events) == 1 and ownership_stream_events[0].code == "CHAT_SESSION_NOT_FOUND"
                )

                # ChatService 先取 Redis guard 再查所有权，以便未来删除流程与执行
                # 共用同一临界区。因此失败后还必须证明两把短暂锁均已释放。
                await _wait_for_lock_state(
                    resources.redis_client,
                    lock_names=(cross_user_lock, absent_lock),
                    expected_exists=False,
                )
                ownership_failure_locks_released = True

                # 即使错误映射正确，也要检查 PostgreSQL saver 中没有产生状态。
                # 空 snapshot 证明无权输入没有成为 HumanMessage 或 checkpoint 分支。
                cross_user_snapshot = await graph.aget_state(
                    ChatService._build_config(
                        user_id=other_user_id,
                        public_thread_id=same_thread,
                    )
                )
                absent_snapshot = await graph.aget_state(
                    ChatService._build_config(
                        user_id=user_id,
                        public_thread_id=absent_thread,
                    )
                )
                ownership_failures_left_no_checkpoint = (
                    not cross_user_snapshot.values
                    and not cross_user_snapshot.next
                    and not absent_snapshot.values
                    and not absent_snapshot.next
                )

                same_lock_name = _lock_name(user_id, same_thread)
                first_task = asyncio.create_task(
                    service.run_turn(
                        ChatRequest(thread_id=same_thread, message=first_prompt),
                        user_id=user_id,
                    )
                )

                # Event 证明第一个请求已经由真实 guard 取得执行权，并停在 Graph
                # 之前；Redis exists 则从基础设施侧独立证明锁 key 确实存在。
                await asyncio.wait_for(
                    first_guard_entered.wait(),
                    timeout=LOCK_OBSERVATION_TIMEOUT_SECONDS,
                )
                await _wait_for_lock_state(
                    resources.redis_client,
                    lock_names=(same_lock_name,),
                    expected_exists=True,
                )

                busy_started_at = perf_counter()
                try:
                    try:
                        await service.run_turn(
                            ChatRequest(
                                thread_id=same_thread,
                                message=rejected_prompt,
                            ),
                            user_id=user_id,
                        )
                    except ChatThreadBusyError:
                        same_thread_rejected = True
                    else:
                        same_thread_rejected = False
                finally:
                    # 即使 busy 断言失败，也必须放行第一个请求，避免遗留后台 task
                    # 和真实 Redis 锁。父类 finally 会在该请求结束后释放 owner token。
                    allow_first_graph.set()
                busy_elapsed_ms = _elapsed_ms(busy_started_at)

                first_result = await first_task
                await _wait_for_lock_state(
                    resources.redis_client,
                    lock_names=(same_lock_name,),
                    expected_exists=False,
                )

                # 直接读取同一个 production Graph 的 checkpoint，只检查被拒绝的
                # marker 是否缺席，不输出任何消息内容。
                same_snapshot = await graph.aget_state(
                    ChatService._build_config(
                        user_id=user_id,
                        public_thread_id=same_thread,
                    )
                )
                same_messages = same_snapshot.values.get("messages", [])
                busy_input_absent_from_checkpoint = not any(
                    isinstance(message, HumanMessage)
                    and isinstance(message.content, str)
                    and rejected_marker in message.content
                    for message in same_messages
                )

                # 两个不同内部 key 应能同时存在于 Redis。观察到两把锁同时存在，
                # 比单纯比较总耗时更稳定，因为 provider 延迟和并发调度会波动。
                parallel_first_lock = _lock_name(user_id, parallel_first_thread)
                parallel_second_lock = _lock_name(user_id, parallel_second_thread)
                parallel_first_task = asyncio.create_task(
                    service.run_turn(
                        ChatRequest(
                            thread_id=parallel_first_thread,
                            message=parallel_first_prompt,
                        ),
                        user_id=user_id,
                    )
                )
                parallel_second_task = asyncio.create_task(
                    service.run_turn(
                        ChatRequest(
                            thread_id=parallel_second_thread,
                            message=parallel_second_prompt,
                        ),
                        user_id=user_id,
                    )
                )
                await _wait_for_lock_state(
                    resources.redis_client,
                    lock_names=(parallel_first_lock, parallel_second_lock),
                    expected_exists=True,
                )
                different_keys_overlapped = True
                parallel_first_result, parallel_second_result = await asyncio.gather(
                    parallel_first_task,
                    parallel_second_task,
                )
                await _wait_for_lock_state(
                    resources.redis_client,
                    lock_names=(parallel_first_lock, parallel_second_lock),
                    expected_exists=False,
                )

                # SSE generator 在第一次迭代时才申请 guard。外部先持有同 key，便可
                # 确定验证 busy 事件而不产生额外模型请求。
                stream_internal_id = _internal_thread_id(user_id, stream_busy_thread)
                external_guard = RedisChatExecutionGuard(
                    resources.redis_client,
                    lease_seconds=CHAT_GUARD_LEASE_SECONDS,
                )
                stream_events: list[ErrorStreamEvent] = []
                async with external_guard.hold(stream_internal_id):
                    async for event in service.stream_turn(
                        ChatRequest(
                            thread_id=stream_busy_thread,
                            message="This stream request must not enter the Graph.",
                        ),
                        user_id=user_id,
                    ):
                        if isinstance(event, ErrorStreamEvent):
                            stream_events.append(event)

                stream_busy_event_matches = len(stream_events) == 1 and stream_events[0].code == "CHAT_THREAD_BUSY"

            state_removed_after_shutdown = not hasattr(app.state, "resources") and not hasattr(
                app.state, "chat_service"
            )
            pool_closed_after_shutdown = resources.postgres_pool.closed

    elapsed_ms = _elapsed_ms(started_at)
    return {
        "model": settings.DEFAULT_LLM_MODEL,
        "cross_user_rejected": cross_user_rejected,
        "absent_session_rejected": absent_session_rejected,
        "ownership_stream_error_matches": ownership_stream_error_matches,
        "ownership_failures_skipped_graph": ownership_failures_skipped_graph,
        "ownership_failure_locks_released": ownership_failure_locks_released,
        "ownership_failures_left_no_checkpoint": ownership_failures_left_no_checkpoint,
        "same_thread_rejected": same_thread_rejected,
        "busy_failed_fast": busy_elapsed_ms <= BUSY_BUDGET_SECONDS * 1000,
        "busy_input_absent_from_checkpoint": busy_input_absent_from_checkpoint,
        "first_real_response_matches": _response_matches(
            first_result,
            FIRST_EXPECTED_REPLY,
        ),
        "different_keys_overlapped": different_keys_overlapped,
        "parallel_real_responses_match": (
            _response_matches(parallel_first_result, DIFFERENT_FIRST_REPLY)
            and _response_matches(parallel_second_result, DIFFERENT_SECOND_REPLY)
        ),
        "stream_busy_event_matches": stream_busy_event_matches,
        "state_removed_after_shutdown": state_removed_after_shutdown,
        "pool_closed_after_shutdown": pool_closed_after_shutdown,
        "within_total_budget": elapsed_ms <= TOTAL_TIMEOUT_SECONDS * 1000,
        "busy_elapsed_ms": busy_elapsed_ms,
        "elapsed_ms": elapsed_ms,
    }


def _run_smoke() -> dict[str, object]:
    """迁移随机数据库、执行真实 Gate，并保证最终清理."""
    started_at = perf_counter()
    admin_database = settings.POSTGRES_DB
    test_database = f"deep_research_chat_guard_{uuid4().hex[:10]}"
    previous_override = os.environ.get("ALEMBIC_DATABASE_URL")
    database_created = False
    cleanup_ok = False
    checks: dict[str, bool | float | str]

    try:
        _create_database(admin_database, test_database)
        database_created = True

        # 使用部署时相同的 Alembic revision 建立业务表。LangGraph checkpoint
        # 表仍由 lifespan 中的 saver.setup() 管理，验证两套 migration ownership
        # 可以在同一空数据库中按正式顺序协作。
        os.environ["ALEMBIC_DATABASE_URL"] = _temporary_database_url(test_database)
        command.upgrade(Config(str(PROJECT_ROOT / "alembic.ini")), "head")

        checks = asyncio.run(
            _exercise_guard(test_database),
            loop_factory=_selector_loop_factory,
        )
    finally:
        # 环境变量只在当前进程中临时覆盖。无论 Gate 是否成功，都必须恢复调用者
        # 原值，避免后续命令意外连接一个即将删除的随机数据库。
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
        raise RuntimeError("temporary Chat guard database cleanup failed")

    boolean_checks = tuple(value for value in checks.values() if isinstance(value, bool))
    return {
        "ok": bool(boolean_checks) and all(boolean_checks) and cleanup_ok,
        **checks,
        "cleanup_ok": cleanup_ok,
        "total_elapsed_ms": _elapsed_ms(started_at),
    }


def main() -> int:
    """打印安全单行 JSON，并返回 shell 友好的进程退出码."""
    started_at = perf_counter()
    try:
        summary = _run_smoke()
    except Exception as error:
        # provider 和基础设施异常可能携带内部诊断信息，只公开异常类名。
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
