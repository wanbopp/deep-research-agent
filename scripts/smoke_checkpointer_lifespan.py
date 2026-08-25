"""Verify PostgreSQL-backed ChatService ownership in the real lifespan.

Checkpoint 10A 已证明 saver 可以读写，10B 已证明 saver 的生命周期。本 smoke 继续
覆盖 10C：startup 使用同一个 saver 编译 production Chat graph、创建唯一 ChatService，
请求依赖只读取该服务，shutdown 后 accessor 再次拒绝访问。

脚本使用随机临时 PostgreSQL 数据库，并调用真实 Neo4j/Redis probe。它会构造真实
LLM/Tool/Graph 对象但不会执行图，因此不会发送 provider 请求，也不会输出连接参数、
数据库名、API key、thread ID 或 checkpoint 正文。
"""

import asyncio
import json
import os
import selectors
from pathlib import Path
from time import perf_counter
from unittest.mock import patch
from uuid import uuid4

import psycopg
from alembic import command
from alembic.config import Config
from fastapi import FastAPI
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg import sql
from psycopg.conninfo import make_conninfo
from starlette.requests import Request
from starlette.types import Scope

from app.agents.chat.graph import ChatGraph
from app.agents.chat.runtime import create_chat_runtime
from app.api.dependencies import get_chat_service
from app.core.config import Settings, settings
from app.infrastructure.database import build_orm_database_url
import app.infrastructure.lifespan as lifespan_module
from app.infrastructure.lifespan import (
    get_application_chat_service,
    get_application_resources,
    lifespan,
)
from app.infrastructure.resources import ApplicationResources

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONNECTION_TIMEOUT_SECONDS = 10
TOTAL_TIMEOUT_SECONDS = 20.0
CHECKPOINT_TABLES = frozenset(
    {
        "checkpoint_migrations",
        "checkpoints",
        "checkpoint_blobs",
        "checkpoint_writes",
    }
)


def _elapsed_ms(started_at: float) -> float:
    """Return elapsed milliseconds without infrastructure details."""
    return round((perf_counter() - started_at) * 1000, 2)


def _conninfo(database: str) -> str:
    """Build a private psycopg connection string for database administration."""
    return make_conninfo(
        host=settings.POSTGRES_HOST,
        port=settings.POSTGRES_PORT,
        dbname=database,
        user=settings.POSTGRES_USER,
        password=settings.POSTGRES_PASSWORD,
        connect_timeout=CONNECTION_TIMEOUT_SECONDS,
    )


def _temporary_database_url(database: str) -> str:
    """Build the temporary Alembic URL kept only in this process."""
    return build_orm_database_url(settings).set(database=database).render_as_string(hide_password=False)


def _create_database(admin_database: str, test_database: str) -> None:
    """Create one safely quoted random database outside a transaction."""
    with psycopg.connect(_conninfo(admin_database), autocommit=True) as connection:
        connection.execute(
            sql.SQL("CREATE DATABASE {}").format(sql.Identifier(test_database)),
        )


def _drop_database(admin_database: str, test_database: str) -> None:
    """Terminate residual sessions and remove only this smoke database."""
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
    """Copy settings and redirect only PostgreSQL resources to the random database."""
    config = Settings()
    config.POSTGRES_DB = database
    config.POSTGRES_PSYCOPG_POOL_SIZE = 2
    config.POSTGRES_ORM_POOL_SIZE = 2
    config.POSTGRES_ORM_MAX_OVERFLOW = 0
    return config


def _selector_loop_factory() -> asyncio.AbstractEventLoop:
    """Create the Windows-compatible event loop used by psycopg async."""
    return asyncio.SelectorEventLoop(selectors.SelectSelector())


async def _exercise_lifespan(database: str) -> dict[str, bool | float | int]:
    """Enter an isolated real lifespan and inspect Graph/service ownership.

    Args:
        database: Random migrated PostgreSQL database used by both pool and saver.

    Returns:
        Sanitized lifecycle booleans, migration count and elapsed time.

    Raises:
        Exception: Real probe/setup failures propagate to the safe top-level boundary.
    """
    started_at = perf_counter()
    app = FastAPI()
    state_absent_before_startup = not hasattr(app.state, "resources") and not hasattr(
        app.state,
        "chat_service",
    )

    # 严格 accessor 只能在 lifespan 的 yield 区间使用。startup 前拒绝访问，
    # 可以防止测试或请求悄悄得到一个绕开 PostgreSQL 的临时内存 Graph。
    try:
        get_application_chat_service(app)
    except RuntimeError as error:
        service_accessor_rejects_before_startup = str(error) == "Application chat service is not initialized"
    else:
        service_accessor_rejects_before_startup = False

    captured_checkpointers: list[BaseCheckpointSaver[str] | None] = []
    captured_graphs: list[ChatGraph] = []

    def capture_runtime(
        *,
        checkpointer: BaseCheckpointSaver[str] | None = None,
    ) -> ChatGraph:
        """Capture the real production assembly boundary.

        Args:
            checkpointer: lifespan 注入 production runtime 的 saver。10C 要求它必须
                是当前 ApplicationResources 中已经 setup 的 PostgreSQL saver。

        Returns:
            由真实 create_chat_runtime 构造的完整 ChatGraph。脚本只检查对象关系，
            不调用 ainvoke/astream，因此不会发送模型请求。
        """
        captured_checkpointers.append(checkpointer)
        graph = create_chat_runtime(checkpointer=checkpointer)
        captured_graphs.append(graph)
        return graph

    async with asyncio.timeout(TOTAL_TIMEOUT_SECONDS):
        # patch 的只是 lifespan 模块查找 create_chat_runtime 的位置。wrapper 仍调用
        # 真实工厂；这样既能证明注入身份，又不会使用 fake Graph 掩盖组装问题。
        with patch.object(lifespan_module, "create_chat_runtime", capture_runtime):
            async with lifespan(app, config=_runtime_settings(database)):
                first_resources = get_application_resources(app)
                second_resources = get_application_resources(app)
                first_service = get_application_chat_service(app)
                second_service = get_application_chat_service(app)

                # FastAPI dependency 通过 Request.app 找到当前应用，不能依赖无参数
                # lru_cache。最小 Scope 足以验证这条纯本地依赖解析路径。
                scope: Scope = {"type": "http", "app": app}
                dependency_service = get_chat_service(Request(scope))

                state_available_after_setup = hasattr(app.state, "resources") and hasattr(
                    app.state,
                    "chat_service",
                )
                resource_identity_is_stable = first_resources is second_resources
                service_identity_is_stable = first_service is second_service is dependency_service
                runtime_constructed_once = len(captured_graphs) == 1
                runtime_received_lifespan_saver = (
                    len(captured_checkpointers) == 1 and captured_checkpointers[0] is first_resources.checkpointer
                )
                graph_uses_lifespan_saver = (
                    len(captured_graphs) == 1 and captured_graphs[0].checkpointer is first_resources.checkpointer
                )
                saver_reuses_lifespan_pool = first_resources.checkpointer.conn is first_resources.postgres_pool
                pool_open_after_setup = not first_resources.postgres_pool.closed

                # aget_tuple() reaches the real checkpoint tables. No row is expected because
                # 10C only compiles Graph；真实状态写入和跨 Graph 恢复留给 10D。
                missing_config: RunnableConfig = {
                    "configurable": {
                        "thread_id": f"lifespan-smoke-{uuid4().hex}",
                    }
                }
                empty_checkpoint_read_succeeded = await first_resources.checkpointer.aget_tuple(missing_config) is None

                async with first_resources.postgres_pool.connection() as connection:
                    table_cursor = await connection.execute(
                        """
                        SELECT tablename
                        FROM pg_catalog.pg_tables
                        WHERE schemaname = 'public'
                        ORDER BY tablename
                        """
                    )
                    rows = await table_cursor.fetchall()
                    table_names = frozenset(str(row["tablename"]) for row in rows)

                    migration_cursor = await connection.execute(
                        "SELECT COUNT(*) AS migration_count FROM checkpoint_migrations"
                    )
                    migration_row = await migration_cursor.fetchone()
                    if migration_row is None:
                        raise RuntimeError("checkpoint migration query returned no row")
                    migration_count = int(migration_row["migration_count"])

                checkpoint_tables_ready_before_publish = CHECKPOINT_TABLES <= table_names
                all_internal_migrations_applied = migration_count == len(first_resources.checkpointer.MIGRATIONS)

        # These checks happen only after lifespan's finally block and AsyncExitStack finish.
        state_removed_after_shutdown = not hasattr(app.state, "resources") and not hasattr(
            app.state,
            "chat_service",
        )
        pool_closed_after_shutdown = first_resources.postgres_pool.closed
        try:
            get_application_resources(app)
        except RuntimeError:
            accessor_rejects_after_shutdown = True
        else:
            accessor_rejects_after_shutdown = False

        try:
            get_application_chat_service(app)
        except RuntimeError as error:
            service_accessor_rejects_after_shutdown = str(error) == "Application chat service is not initialized"
        else:
            service_accessor_rejects_after_shutdown = False

        # 正常真实 setup 已在上面完成。这里仅注入一个“setup 抛异常”的控制流，
        # 验证我们自己的 lifespan/AsyncExitStack 行为，不把它当作数据库能力证据。
        failure_app = FastAPI()
        captured_resources: list[ApplicationResources] = []
        real_factory = lifespan_module.create_application_resources

        def capture_resources(config: Settings) -> ApplicationResources:
            """Capture the lazily constructed resources before synthetic setup failure."""
            resources = real_factory(config)
            captured_resources.append(resources)
            return resources

        async def fail_setup(_checkpointer: object) -> None:
            """Represent an arbitrary saver setup failure without exposing diagnostics."""
            raise RuntimeError("synthetic checkpointer setup failure")

        with (
            patch.object(
                lifespan_module,
                "create_application_resources",
                capture_resources,
            ),
            patch.object(
                AsyncPostgresSaver,
                "setup",
                fail_setup,
            ),
        ):
            try:
                async with lifespan(failure_app, config=_runtime_settings(database)):
                    raise AssertionError("failed setup must never reach lifespan yield")
            except RuntimeError as error:
                setup_failure_is_safe = str(error) == "PostgreSQL checkpointer setup failed"
            else:
                setup_failure_is_safe = False

        setup_failure_blocks_publish_and_cleans_up = (
            setup_failure_is_safe
            and len(captured_resources) == 1
            and not hasattr(failure_app.state, "resources")
            and not hasattr(failure_app.state, "chat_service")
            and captured_resources[0].postgres_pool.closed
        )

    elapsed_ms = _elapsed_ms(started_at)
    within_total_budget = elapsed_ms <= TOTAL_TIMEOUT_SECONDS * 1000
    return {
        "state_absent_before_startup": state_absent_before_startup,
        "service_accessor_rejects_before_startup": service_accessor_rejects_before_startup,
        "state_available_after_setup": state_available_after_setup,
        "resource_identity_is_stable": resource_identity_is_stable,
        "service_identity_is_stable": service_identity_is_stable,
        "runtime_constructed_once": runtime_constructed_once,
        "runtime_received_lifespan_saver": runtime_received_lifespan_saver,
        "graph_uses_lifespan_saver": graph_uses_lifespan_saver,
        "saver_reuses_lifespan_pool": saver_reuses_lifespan_pool,
        "pool_open_after_setup": pool_open_after_setup,
        "empty_checkpoint_read_succeeded": empty_checkpoint_read_succeeded,
        "checkpoint_tables_ready_before_publish": checkpoint_tables_ready_before_publish,
        "all_internal_migrations_applied": all_internal_migrations_applied,
        "migration_count": migration_count,
        "state_removed_after_shutdown": state_removed_after_shutdown,
        "pool_closed_after_shutdown": pool_closed_after_shutdown,
        "accessor_rejects_after_shutdown": accessor_rejects_after_shutdown,
        "service_accessor_rejects_after_shutdown": service_accessor_rejects_after_shutdown,
        "setup_failure_blocks_publish_and_cleans_up": setup_failure_blocks_publish_and_cleans_up,
        "within_total_budget": within_total_budget,
        "elapsed_ms": elapsed_ms,
    }


def _run_smoke() -> dict[str, object]:
    """Create, migrate, exercise and delete a random PostgreSQL database."""
    started_at = perf_counter()
    admin_database = settings.POSTGRES_DB
    test_database = f"deep_research_checkpointer_life_{uuid4().hex[:8]}"
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
            _exercise_lifespan(test_database),
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
        raise RuntimeError("temporary checkpointer lifespan database cleanup failed")

    boolean_checks = tuple(value for value in checks.values() if isinstance(value, bool))
    ok = bool(boolean_checks) and all(boolean_checks) and cleanup_ok
    return {
        "ok": ok,
        **checks,
        "cleanup_ok": cleanup_ok,
        "total_elapsed_ms": _elapsed_ms(started_at),
    }


def main() -> int:
    """Print one safe JSON lifecycle summary and return a process code."""
    started_at = perf_counter()
    try:
        summary = _run_smoke()
    except Exception as error:
        # Exception strings may include connection diagnostics; print only the type.
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
