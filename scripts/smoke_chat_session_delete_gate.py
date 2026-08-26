"""使用真实 provider 验收 ChatSession 删除与 Agent 执行的生产组合.

本 Gate 覆盖 Checkpoint 10F-F 的核心行为：

1. 随机 PostgreSQL 数据库执行正式 Alembic migration；
2. production lifespan 创建真实 Graph、Redis guard 和 AsyncPostgresSaver；
3. 正式 Auth/ChatSession/Chat route 创建业务资源并调用当前真实 provider；
4. 同一 internal key 被 Chat 持有时，DELETE 必须 fail-fast 409；
5. Chat 完成后 DELETE 必须清空业务行和三类 saver 数据；
6. 删除后的 read/run/resume 在 Graph 前返回 404，对照会话仍能继续真实对话；
7. shutdown 后 app.state、连接池、Redis 锁和随机数据库均被清理。

脚本不替换模型、不伪造模型输出。对真实 Redis guard 的子类只增加一次性同步
屏障，实际加锁、owner token、lease 和释放仍全部委托 production 实现。最终 JSON
不包含 prompt、模型正文、token、邮箱、UUID、内部 key、Redis key 或连接信息。
"""

import asyncio
import json
import os
import secrets
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
from asgi_correlation_id import CorrelationIdMiddleware
from fastapi import FastAPI, status
from httpx import ASGITransport, AsyncClient, Response
from langgraph.checkpoint.base import BaseCheckpointSaver
from psycopg import sql
from psycopg.conninfo import make_conninfo
from pydantic import SecretStr
from redis.asyncio import Redis

import app.infrastructure.lifespan as lifespan_module
from app.agents.chat.graph import ChatGraph
from app.agents.chat.runtime import create_chat_runtime
from app.api.dependencies import get_token_service
from app.api.v1.auth import router as auth_router
from app.api.v1.chat import router as chat_router
from app.api.v1.chat_sessions import router as chat_sessions_router
from app.core.config import Settings, settings
from app.core.exception_handlers import register_exception_handlers
from app.infrastructure.chat_guard import (
    CHAT_EXECUTION_LOCK_PREFIX,
    RedisChatExecutionGuard,
)
from app.infrastructure.database import build_orm_database_url
from app.infrastructure.lifespan import (
    get_application_resources,
    lifespan,
)
from app.infrastructure.resources import ApplicationResources
from app.models import ChatSessionStatus
from app.repositories import ChatSessionRepository
from app.services.auth import TokenService
from app.services.chat import ChatService

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONNECTION_TIMEOUT_SECONDS = 10
TOTAL_TIMEOUT_SECONDS = 330.0
LOCK_OBSERVATION_TIMEOUT_SECONDS = 15.0
BUSY_BUDGET_SECONDS = 1.0
TEMPORARY_DATABASE_PREFIX = "deep_research_chat_delete_gate_"
CHECKPOINT_TABLES = (
    "checkpoints",
    "checkpoint_blobs",
    "checkpoint_writes",
)

TARGET_EXPECTED_REPLY = "REAL_DELETE_TARGET_OK"
CONTROL_FIRST_EXPECTED_REPLY = "REAL_DELETE_CONTROL_A_OK"
CONTROL_SECOND_EXPECTED_REPLY = "REAL_DELETE_CONTROL_B_OK"


def _elapsed_ms(started_at: float) -> float:
    """返回可安全输出的单调时钟毫秒数."""
    return round((perf_counter() - started_at) * 1000, 2)


def _conninfo(database: str) -> str:
    """构造只交给 psycopg 使用且绝不输出的连接参数."""
    return make_conninfo(
        host=settings.POSTGRES_HOST,
        port=settings.POSTGRES_PORT,
        dbname=database,
        user=settings.POSTGRES_USER,
        password=settings.POSTGRES_PASSWORD,
        connect_timeout=CONNECTION_TIMEOUT_SECONDS,
    )


def _temporary_database_url(database: str) -> str:
    """为正式 Alembic migration 构造随机数据库 URL."""
    return build_orm_database_url(settings).set(database=database).render_as_string(hide_password=False)


def _create_database(admin_database: str, test_database: str) -> None:
    """创建具有固定安全前缀的随机隔离数据库.

    Args:
        admin_database: 已存在且允许执行 CREATE DATABASE 的管理库。
        test_database: 本 Gate 独占的随机数据库名称。

    Raises:
        ValueError: 数据库名不带固定 Gate 前缀时拒绝执行。
    """
    if not test_database.startswith(TEMPORARY_DATABASE_PREFIX):
        raise ValueError("refusing to create a database without the Gate prefix")
    with psycopg.connect(_conninfo(admin_database), autocommit=True) as connection:
        connection.execute(
            sql.SQL("CREATE DATABASE {}").format(sql.Identifier(test_database)),
        )


def _drop_database(admin_database: str, test_database: str) -> None:
    """终止残留连接并只删除当前 Gate 创建的数据库."""
    if not test_database.startswith(TEMPORARY_DATABASE_PREFIX):
        raise ValueError("refusing to drop a database without the Gate prefix")
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
    """保留真实 provider/Redis 配置，只替换数据库并缩小连接池."""
    config = Settings()
    config.POSTGRES_DB = database
    config.POSTGRES_PSYCOPG_POOL_SIZE = 4
    config.POSTGRES_ORM_POOL_SIZE = 3
    config.POSTGRES_ORM_MAX_OVERFLOW = 0
    return config


def _selector_loop_factory() -> asyncio.AbstractEventLoop:
    """创建 Windows psycopg 异步驱动要求的 Selector event loop."""
    return asyncio.SelectorEventLoop(selectors.SelectSelector())


def _authorization_header(token: str) -> dict[str, str]:
    """构造只在当前进程内使用且绝不输出的 Bearer header."""
    return {"Authorization": f"Bearer {token}"}


def _error_signature(response: Response) -> tuple[int, object, object]:
    """提取统一错误中不包含随机 request_id 的稳定部分."""
    body = response.json()
    error = body.get("error", {})
    return response.status_code, error.get("code"), error.get("message")


def _completed_response_matches(response: Response, expected: str) -> bool:
    """验证真实 Chat HTTP 响应形状和固定正文，不把正文写入摘要."""
    if response.status_code != status.HTTP_200_OK:
        return False
    body = response.json()
    message = body.get("message")
    return body.get("status") == "completed" and isinstance(message, dict) and message.get("content") == expected


async def _register_user(
    client: AsyncClient,
    *,
    email: str,
    password: str,
) -> str:
    """通过正式注册 route 创建用户并返回仅供本进程使用的 token."""
    response = await client.post(
        "/api/v1/auth/register",
        json={"email": email, "password": password},
    )
    response.raise_for_status()
    token = response.json().get("access_token")
    if not isinstance(token, str) or not token:
        raise RuntimeError("registration response did not contain an access token")
    return token


async def _create_session(
    client: AsyncClient,
    *,
    headers: dict[str, str],
    title: str,
) -> UUID:
    """通过正式 ChatSession API 创建并解析公开 UUID."""
    response = await client.post(
        "/api/v1/chat/sessions",
        headers=headers,
        json={"title": title},
    )
    response.raise_for_status()
    return UUID(str(response.json()["thread_id"]))


async def _checkpoint_row_counts(
    resources: ApplicationResources,
    *,
    internal_thread_id: str,
) -> dict[str, int]:
    """按固定表白名单统计一个内部 thread 的真实 saver 行.

    Args:
        resources: lifespan 创建的真实 ApplicationResources。
        internal_thread_id: 只作为 SQL 参数传递的可信内部 key。

    Returns:
        三张固定 saver 表各自的行数。表名用 ``sql.Identifier``，值不拼 SQL。
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


def _lock_name(internal_thread_id: str) -> str:
    """计算 production Redis guard 使用的摘要 key，仅供 exists 检查."""
    digest = sha256(internal_thread_id.encode("utf-8")).hexdigest()
    return f"{CHAT_EXECUTION_LOCK_PREFIX}{digest}"


async def _wait_for_lock_state(
    redis_client: Redis,
    *,
    lock_name: str,
    expected_exists: bool,
) -> None:
    """在固定预算内观察真实 Redis 锁是否达到目标状态."""
    deadline = asyncio.get_running_loop().time() + LOCK_OBSERVATION_TIMEOUT_SECONDS
    while asyncio.get_running_loop().time() < deadline:
        if bool(await redis_client.exists(lock_name)) is expected_exists:
            return
        await asyncio.sleep(0.01)
    raise TimeoutError("expected Redis lock state was not observed")


async def _exercise_delete_gate(database: str) -> dict[str, bool | float | int | str]:
    """在 production lifespan 中执行真实 provider 删除总 Gate.

    Args:
        database: 已升级到 Alembic head 的随机 PostgreSQL 数据库。

    Returns:
        只含模型名、布尔检查、计数和耗时的脱敏摘要。

    Raises:
        Exception: provider、Graph、Redis、PostgreSQL、HTTP 或生命周期不符合预期
            时向上传播；顶层仍会在 finally 中删除随机数据库。
    """
    started_at = perf_counter()
    token_service = TokenService(secret_key=SecretStr(secrets.token_urlsafe(48)))

    # 这两个 Event 只控制确定的并发顺序。第一个 Chat 请求先取得真实 Redis 锁，
    # DELETE 完成 busy 断言后再允许它进入真实 Graph/provider。
    first_guard_entered = asyncio.Event()
    allow_target_graph = asyncio.Event()
    observed_internal_thread_id: str | None = None
    captured_graphs: list[ChatGraph] = []

    class ObservedRedisChatExecutionGuard(RedisChatExecutionGuard):
        """在 production Redis guard 外增加一次性、仅供 smoke 的同步屏障."""

        def __init__(
            self,
            redis_client: Redis,
            *,
            lease_seconds: float,
        ) -> None:
            """保存真实 Redis 配置并初始化目标 thread 观察状态."""
            super().__init__(redis_client, lease_seconds=lease_seconds)
            self._target_observed = False

        @asynccontextmanager
        async def hold(self, internal_thread_id: str) -> AsyncIterator[None]:
            """先委托真实锁，再仅暂停第一个目标 Chat 请求.

            Args:
                internal_thread_id: ChatService/cleanup coordinator 使用的可信内部 key。

            Yields:
                smoke 放行后继续执行原生产调用方。第二个同 key DELETE 会在父类
                获取锁时直接抛 busy，无法进入本屏障。
            """
            async with super().hold(internal_thread_id):
                if (
                    observed_internal_thread_id is not None
                    and internal_thread_id == observed_internal_thread_id
                    and not self._target_observed
                ):
                    self._target_observed = True
                    first_guard_entered.set()
                    await allow_target_graph.wait()
                yield

    def capture_runtime(
        *,
        checkpointer: BaseCheckpointSaver[str] | None = None,
    ) -> ChatGraph:
        """捕获 production Graph，但仍调用真实 runtime factory 和 provider."""
        graph = create_chat_runtime(checkpointer=checkpointer)
        captured_graphs.append(graph)
        return graph

    app = FastAPI()
    register_exception_handlers(app)
    app.add_middleware(CorrelationIdMiddleware)
    app.include_router(auth_router, prefix="/api/v1")
    app.include_router(chat_router, prefix="/api/v1")
    app.include_router(chat_sessions_router, prefix="/api/v1")
    app.dependency_overrides[get_token_service] = lambda: token_service

    async with asyncio.timeout(TOTAL_TIMEOUT_SECONDS):
        # patch 只增加 Graph 捕获和 Redis 同步观察；所有模型行为与 checkpoint
        # 写入仍走 production 对象。
        with (
            patch.object(lifespan_module, "create_chat_runtime", capture_runtime),
            patch.object(
                lifespan_module,
                "RedisChatExecutionGuard",
                ObservedRedisChatExecutionGuard,
            ),
        ):
            async with lifespan(app, config=_runtime_settings(database)):
                resources = get_application_resources(app)
                if len(captured_graphs) != 1:
                    raise RuntimeError("lifespan must build exactly one Chat graph")
                graph = captured_graphs[0]

                transport = ASGITransport(app=app, raise_app_exceptions=False)
                async with AsyncClient(transport=transport, base_url="http://testserver") as client:
                    suffix = uuid4().hex
                    password = "Real-Delete-Gate-2026!"
                    token = await _register_user(
                        client,
                        email=f"delete-gate-{suffix}@example.com",
                        password=password,
                    )
                    headers = _authorization_header(token)
                    user_id = token_service.decode_access_token(token).sub

                    target_id = await _create_session(
                        client,
                        headers=headers,
                        title="Real provider deletion target",
                    )
                    control_id = await _create_session(
                        client,
                        headers=headers,
                        title="Real provider deletion control",
                    )
                    observed_internal_thread_id = ChatService._build_checkpoint_thread_id(
                        user_id,
                        target_id,
                    )
                    control_internal_thread_id = ChatService._build_checkpoint_thread_id(
                        user_id,
                        control_id,
                    )
                    target_lock_name = _lock_name(observed_internal_thread_id)

                    # 启动目标 Chat。Observed guard 会在真实 Redis 锁成功后、Graph
                    # 调用前暂停，因此接下来的 DELETE 必定与一个已获得执行权的
                    # production Chat 请求竞争相同 key。
                    target_task = asyncio.create_task(
                        client.post(
                            "/api/v1/chat",
                            headers=headers,
                            json={
                                "thread_id": str(target_id),
                                "message": (
                                    "Do not call any tool. Reply with exactly "
                                    f"{TARGET_EXPECTED_REPLY} and nothing else."
                                ),
                            },
                        )
                    )

                    try:
                        await asyncio.wait_for(
                            first_guard_entered.wait(),
                            timeout=LOCK_OBSERVATION_TIMEOUT_SECONDS,
                        )
                        await _wait_for_lock_state(
                            resources.redis_client,
                            lock_name=target_lock_name,
                            expected_exists=True,
                        )

                        busy_started_at = perf_counter()
                        busy_delete = await client.delete(
                            f"/api/v1/chat/sessions/{target_id}",
                            headers=headers,
                        )
                        busy_elapsed_ms = _elapsed_ms(busy_started_at)

                        # busy DELETE 必须在阶段 1 之前失败：业务行仍为 active，且
                        # Graph 尚未放行，所以目标 saver 行也应仍为 0。
                        async with resources.orm_session_factory() as session:
                            async with session.begin():
                                target_during_busy = await ChatSessionRepository(session).get_for_cleanup(
                                    target_id,
                                    user_id=user_id,
                                )
                        target_counts_during_busy = await _checkpoint_row_counts(
                            resources,
                            internal_thread_id=observed_internal_thread_id,
                        )
                    finally:
                        # 无论 busy 断言路径发生什么，都放行真实 Chat，避免遗留 task
                        # 或依赖 lease 超时才释放的 Redis 锁。
                        allow_target_graph.set()

                    target_response = await target_task
                    await _wait_for_lock_state(
                        resources.redis_client,
                        lock_name=target_lock_name,
                        expected_exists=False,
                    )
                    target_counts_before_delete = await _checkpoint_row_counts(
                        resources,
                        internal_thread_id=observed_internal_thread_id,
                    )

                    # 对照会话执行独立真实 provider 请求，随后记录其 saver 行。目标
                    # DELETE 必须完全不改变这些行。
                    control_first = await client.post(
                        "/api/v1/chat",
                        headers=headers,
                        json={
                            "thread_id": str(control_id),
                            "message": (
                                "Do not call any tool. Reply with exactly "
                                f"{CONTROL_FIRST_EXPECTED_REPLY} and nothing else."
                            ),
                        },
                    )
                    control_counts_before_delete = await _checkpoint_row_counts(
                        resources,
                        internal_thread_id=control_internal_thread_id,
                    )

                    owner_delete = await client.delete(
                        f"/api/v1/chat/sessions/{target_id}",
                        headers=headers,
                    )
                    target_counts_after_delete = await _checkpoint_row_counts(
                        resources,
                        internal_thread_id=observed_internal_thread_id,
                    )
                    control_counts_after_delete = await _checkpoint_row_counts(
                        resources,
                        internal_thread_id=control_internal_thread_id,
                    )

                    async with resources.orm_session_factory() as session:
                        async with session.begin():
                            repository = ChatSessionRepository(session)
                            target_after_delete = await repository.get_for_cleanup(
                                target_id,
                                user_id=user_id,
                            )
                            control_after_delete = await repository.get_by_id(
                                control_id,
                                user_id=user_id,
                            )

                    target_snapshot_after_delete = await graph.aget_state(
                        ChatService._build_config(
                            user_id=user_id,
                            public_thread_id=target_id,
                        )
                    )

                    # 删除后的三个公开入口必须在 Graph 前拒绝。spy 只观察调用次数，
                    # 不替换返回值；call_count 为 0 证明没有第四个 provider 请求。
                    with patch.object(graph, "ainvoke", wraps=graph.ainvoke) as deleted_graph_spy:
                        deleted_read = await client.get(
                            f"/api/v1/chat/sessions/{target_id}",
                            headers=headers,
                        )
                        deleted_run = await client.post(
                            "/api/v1/chat",
                            headers=headers,
                            json={
                                "thread_id": str(target_id),
                                "message": "This deleted session must not call the model.",
                            },
                        )
                        deleted_resume = await client.post(
                            "/api/v1/chat/resume",
                            headers=headers,
                            json={
                                "thread_id": str(target_id),
                                "response": "This deleted session must not resume.",
                            },
                        )
                        deleted_requests_skipped_graph = deleted_graph_spy.call_count == 0

                    # 对照会话继续发起第三次真实 provider 请求，证明目标删除没有
                    # 破坏共享 Graph、saver 或同用户的其他 internal key。
                    control_second = await client.post(
                        "/api/v1/chat",
                        headers=headers,
                        json={
                            "thread_id": str(control_id),
                            "message": (
                                "Do not call any tool. Reply with exactly "
                                f"{CONTROL_SECOND_EXPECTED_REPLY} and nothing else."
                            ),
                        },
                    )
                    control_counts_final = await _checkpoint_row_counts(
                        resources,
                        internal_thread_id=control_internal_thread_id,
                    )
                    target_lock_released_after_delete = not bool(await resources.redis_client.exists(target_lock_name))

                not_found_signature = (
                    404,
                    "HTTP_ERROR",
                    "Chat session was not found",
                )
                checks: dict[str, bool | float | int | str] = {
                    "model": settings.DEFAULT_LLM_MODEL,
                    "busy_delete_is_409": _error_signature(busy_delete)
                    == (
                        409,
                        "CHAT_THREAD_BUSY",
                        "Chat thread is already being processed",
                    ),
                    "busy_delete_is_fail_fast": busy_elapsed_ms <= BUSY_BUDGET_SECONDS * 1000,
                    "busy_keeps_session_active": (
                        target_during_busy is not None and target_during_busy.status == ChatSessionStatus.ACTIVE
                    ),
                    "busy_writes_no_checkpoint": all(count == 0 for count in target_counts_during_busy.values()),
                    "target_real_response_matches": _completed_response_matches(
                        target_response,
                        TARGET_EXPECTED_REPLY,
                    ),
                    "target_has_real_checkpoint": all(count > 0 for count in target_counts_before_delete.values()),
                    "control_first_real_response_matches": _completed_response_matches(
                        control_first,
                        CONTROL_FIRST_EXPECTED_REPLY,
                    ),
                    "control_has_real_checkpoint": all(count > 0 for count in control_counts_before_delete.values()),
                    "owner_delete_is_empty_204": (owner_delete.status_code == 204 and owner_delete.content == b""),
                    "target_business_row_deleted": target_after_delete is None,
                    "target_checkpoint_rows_deleted": all(count == 0 for count in target_counts_after_delete.values()),
                    "target_graph_state_deleted": (
                        not target_snapshot_after_delete.values and not target_snapshot_after_delete.next
                    ),
                    "control_unchanged_by_target_delete": (
                        control_after_delete is not None
                        and control_counts_after_delete == control_counts_before_delete
                    ),
                    "deleted_read_is_404": _error_signature(deleted_read) == not_found_signature,
                    "deleted_run_is_404": _error_signature(deleted_run) == not_found_signature,
                    "deleted_resume_is_404": _error_signature(deleted_resume) == not_found_signature,
                    "deleted_requests_skipped_graph": deleted_requests_skipped_graph,
                    "control_second_real_response_matches": _completed_response_matches(
                        control_second,
                        CONTROL_SECOND_EXPECTED_REPLY,
                    ),
                    "control_checkpoint_remains": all(count > 0 for count in control_counts_final.values()),
                    "target_lock_released_after_delete": target_lock_released_after_delete,
                    "real_provider_response_count": 3,
                    "busy_elapsed_ms": busy_elapsed_ms,
                }

            checks["state_removed_after_shutdown"] = (
                not hasattr(app.state, "resources")
                and not hasattr(app.state, "chat_service")
                and not hasattr(app.state, "chat_session_cleanup_service")
            )
            checks["postgres_pool_closed_after_shutdown"] = resources.postgres_pool.closed

    elapsed_ms = _elapsed_ms(started_at)
    checks["within_total_budget"] = elapsed_ms <= TOTAL_TIMEOUT_SECONDS * 1000
    checks["elapsed_ms"] = elapsed_ms
    return checks


def _run_smoke() -> dict[str, object]:
    """迁移随机数据库、运行真实 Gate，并在 finally 中保证删除."""
    started_at = perf_counter()
    admin_database = settings.POSTGRES_DB
    test_database = f"{TEMPORARY_DATABASE_PREFIX}{uuid4().hex[:10]}"
    previous_override = os.environ.get("ALEMBIC_DATABASE_URL")
    database_created = False
    cleanup_ok = False
    checks: dict[str, bool | float | int | str] = {}

    try:
        _create_database(admin_database, test_database)
        database_created = True
        os.environ["ALEMBIC_DATABASE_URL"] = _temporary_database_url(test_database)
        command.upgrade(Config(str(PROJECT_ROOT / "alembic.ini")), "head")
        checks = asyncio.run(
            _exercise_delete_gate(test_database),
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
        raise RuntimeError("temporary real delete Gate database cleanup failed")

    boolean_checks = tuple(value for value in checks.values() if isinstance(value, bool))
    ok = bool(boolean_checks) and all(boolean_checks) and cleanup_ok
    return {
        "ok": ok,
        **checks,
        "cleanup_ok": cleanup_ok,
        "total_elapsed_ms": _elapsed_ms(started_at),
    }


def main() -> int:
    """打印单行脱敏 JSON，并返回 shell 可判断的退出码."""
    started_at = perf_counter()
    try:
        summary = _run_smoke()
    except Exception as error:
        # provider/基础设施异常可能包含敏感诊断，因此只公开异常类型。
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
