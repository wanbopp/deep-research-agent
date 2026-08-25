"""Verify AsyncPostgresSaver ownership inside the real application lifespan.

Checkpoint 10A 已证明 saver 本身可以 setup 和读写；本 smoke 聚焦 10B 的对象所有权
与顺序：factory 构造、pool 打开、required probe、saver setup、app.state 发布，以及
shutdown 后 state 移除与 pool 关闭。

脚本使用随机临时 PostgreSQL 数据库，但仍调用真实 Neo4j/Redis probe。它不编译 Chat
graph、不调用 LLM，也不输出连接参数、数据库名、thread ID 或 checkpoint 正文。
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
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg import sql
from psycopg.conninfo import make_conninfo

from app.core.config import Settings, settings
from app.infrastructure.database import build_orm_database_url
import app.infrastructure.lifespan as lifespan_module
from app.infrastructure.lifespan import get_application_resources, lifespan
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
    """Enter an isolated real lifespan and inspect checkpointer ownership.

    Args:
        database: Random migrated PostgreSQL database used by both pool and saver.

    Returns:
        Sanitized lifecycle booleans, migration count and elapsed time.

    Raises:
        Exception: Real probe/setup failures propagate to the safe top-level boundary.
    """
    started_at = perf_counter()
    app = FastAPI()
    state_absent_before_startup = not hasattr(app.state, "resources")

    async with asyncio.timeout(TOTAL_TIMEOUT_SECONDS):
        async with lifespan(app, config=_runtime_settings(database)):
            first_resources = get_application_resources(app)
            second_resources = get_application_resources(app)

            state_available_after_setup = hasattr(app.state, "resources")
            resource_identity_is_stable = first_resources is second_resources
            saver_reuses_lifespan_pool = first_resources.checkpointer.conn is first_resources.postgres_pool
            pool_open_after_setup = not first_resources.postgres_pool.closed

            # aget_tuple() reaches the real checkpoint tables. No row is expected because
            # 10B only initializes infrastructure; graph writes begin in 10C/10D.
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
        state_removed_after_shutdown = not hasattr(app.state, "resources")
        pool_closed_after_shutdown = first_resources.postgres_pool.closed
        try:
            get_application_resources(app)
        except RuntimeError:
            accessor_rejects_after_shutdown = True
        else:
            accessor_rejects_after_shutdown = False

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
            and captured_resources[0].postgres_pool.closed
        )

    elapsed_ms = _elapsed_ms(started_at)
    within_total_budget = elapsed_ms <= TOTAL_TIMEOUT_SECONDS * 1000
    return {
        "state_absent_before_startup": state_absent_before_startup,
        "state_available_after_setup": state_available_after_setup,
        "resource_identity_is_stable": resource_identity_is_stable,
        "saver_reuses_lifespan_pool": saver_reuses_lifespan_pool,
        "pool_open_after_setup": pool_open_after_setup,
        "empty_checkpoint_read_succeeded": empty_checkpoint_read_succeeded,
        "checkpoint_tables_ready_before_publish": checkpoint_tables_ready_before_publish,
        "all_internal_migrations_applied": all_internal_migrations_applied,
        "migration_count": migration_count,
        "state_removed_after_shutdown": state_removed_after_shutdown,
        "pool_closed_after_shutdown": pool_closed_after_shutdown,
        "accessor_rejects_after_shutdown": accessor_rejects_after_shutdown,
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
