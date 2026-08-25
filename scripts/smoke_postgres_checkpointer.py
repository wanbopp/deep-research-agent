"""Verify the PostgreSQL checkpointer mechanism without calling an LLM.

Checkpoint 10A 先验证存储驱动，而不是立刻改 production Chat runtime。本脚本会：

1. 创建随机临时 PostgreSQL 数据库并执行正式业务 Alembic migration；
2. 使用正式 ``create_application_resources`` 得到项目已经预算的 psycopg pool；
3. 构造 ``AsyncPostgresSaver`` 并连续执行两次 ``setup()``；
4. 用最小确定性 StateGraph 写入、读取真实 checkpoint；
5. 检查 LangGraph 自有表和 migration 版本，并运行 ``alembic check``；
6. 关闭全部惰性资源并删除临时数据库。

这里没有模型、Prompt、token 或用户数据。最终 JSON 只输出版本、布尔结果、计数和
耗时，不输出数据库名、地址、用户名、密码、连接 URL、thread ID 或 checkpoint 正文。
"""

import asyncio
import json
import os
import selectors
from pathlib import Path
from time import perf_counter
from typing import TypedDict, cast
from uuid import uuid4

import psycopg
from alembic import command
from alembic.config import Config
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.graph import END, START, StateGraph
from psycopg import sql
from psycopg.conninfo import make_conninfo

from app.core.config import Settings, settings
from app.infrastructure.database import build_orm_database_url
from app.infrastructure.factory import create_application_resources

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONNECTION_TIMEOUT_SECONDS = 10

BUSINESS_TABLES = frozenset(
    {
        "users",
        "chat_sessions",
        "documents",
        "research_tasks",
    }
)
CHECKPOINT_TABLES = frozenset(
    {
        "checkpoint_migrations",
        "checkpoints",
        "checkpoint_blobs",
        "checkpoint_writes",
    }
)


class _SmokeState(TypedDict):
    """Minimal graph state used only to force one real checkpoint write."""

    counter: int


def _increment(state: _SmokeState) -> _SmokeState:
    """Return one deterministic state update without any provider or external tool.

    Args:
        state: Current graph state loaded by LangGraph. The initial input uses counter 0.

    Returns:
        A replacement counter incremented exactly once.
    """
    return {"counter": state["counter"] + 1}


def _elapsed_ms(started_at: float) -> float:
    """Return elapsed milliseconds without exposing infrastructure configuration."""
    return round((perf_counter() - started_at) * 1000, 2)


def _conninfo(database: str) -> str:
    """Build a psycopg connection string that is never printed or logged."""
    return make_conninfo(
        host=settings.POSTGRES_HOST,
        port=settings.POSTGRES_PORT,
        dbname=database,
        user=settings.POSTGRES_USER,
        password=settings.POSTGRES_PASSWORD,
        connect_timeout=CONNECTION_TIMEOUT_SECONDS,
    )


def _temporary_database_url(database: str) -> str:
    """Build the in-process Alembic URL for the random database."""
    return build_orm_database_url(settings).set(database=database).render_as_string(hide_password=False)


def _create_database(admin_database: str, test_database: str) -> None:
    """Create the random database outside a transaction using a quoted identifier."""
    with psycopg.connect(_conninfo(admin_database), autocommit=True) as connection:
        connection.execute(
            sql.SQL("CREATE DATABASE {}").format(sql.Identifier(test_database)),
        )


def _drop_database(admin_database: str, test_database: str) -> None:
    """Terminate residual sessions and delete only the random smoke database."""
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
    """Copy settings and isolate only the database plus a small smoke pool budget."""
    config = Settings()
    config.POSTGRES_DB = database
    config.POSTGRES_PSYCOPG_POOL_SIZE = 2
    config.POSTGRES_ORM_POOL_SIZE = 2
    config.POSTGRES_ORM_MAX_OVERFLOW = 0
    return config


def _selector_loop_factory() -> asyncio.AbstractEventLoop:
    """Create the Selector loop required by psycopg async connections on Windows."""
    return asyncio.SelectorEventLoop(selectors.SelectSelector())


async def _exercise_checkpointer(database: str) -> dict[str, bool | int | float]:
    """Run saver setup and graph persistence through the production resource factory.

    Args:
        database: Random database that has already reached the business Alembic head.

    Returns:
        Sanitized mechanism checks and execution time.

    Raises:
        Exception: Pool configuration, saver migration or graph persistence failures are
            allowed to propagate. The outer layer reports only the exception type and
            still closes resources and drops the random database.
    """
    started_at = perf_counter()
    resources = create_application_resources(_runtime_settings(database))

    try:
        # The factory deliberately creates the pool with open=False; lifespan normally owns
        # this transition. The smoke mirrors only the PostgreSQL part of startup.
        await resources.postgres_pool.open()

        # This identity assertion proves saver and readiness probe share the already-budgeted
        # pool. A second pool would increase each process's database connection ceiling.
        saver = AsyncPostgresSaver(resources.postgres_pool)
        saver_reuses_application_pool = saver.conn is resources.postgres_pool

        # setup() is a component-owned migration runner. Calling it twice verifies that
        # checkpoint_migrations lets repeated process startup converge safely.
        await saver.setup()
        await saver.setup()

        async with resources.postgres_pool.connection() as connection:
            # Saver migrations contain CREATE INDEX CONCURRENTLY. The shared pool must use
            # autocommit; otherwise setup runs inside a transaction and PostgreSQL rejects it.
            pool_connection_is_checkpointer_ready = connection.autocommit is True and connection.prepare_threshold == 0

            table_cursor = await connection.execute(
                """
                SELECT tablename
                FROM pg_catalog.pg_tables
                WHERE schemaname = 'public'
                ORDER BY tablename
                """
            )
            table_rows = await table_cursor.fetchall()
            public_tables = frozenset(str(row["tablename"] if isinstance(row, dict) else row[0]) for row in table_rows)

            migration_cursor = await connection.execute(
                "SELECT COUNT(*) AS migration_count FROM checkpoint_migrations"
            )
            migration_row = await migration_cursor.fetchone()
            if migration_row is None:
                raise RuntimeError("checkpoint migration count query returned no row")
            migration_count = int(
                migration_row["migration_count"] if isinstance(migration_row, dict) else migration_row[0]
            )

        # This graph is intentionally tiny and deterministic. LangGraph, not our node,
        # invokes saver.aput()/aget_tuple() around graph execution.
        builder = StateGraph(_SmokeState)
        builder.add_node("increment", _increment)
        builder.add_edge(START, "increment")
        builder.add_edge("increment", END)
        graph = builder.compile(checkpointer=saver)

        config: RunnableConfig = {
            "configurable": {
                "thread_id": f"postgres-checkpointer-smoke-{uuid4().hex}",
            }
        }
        result = cast(_SmokeState, await graph.ainvoke({"counter": 0}, config=config))
        snapshot = await graph.aget_state(config)
        checkpoint_tuple = await saver.aget_tuple(config)

        graph_round_trip_succeeded = (
            result["counter"] == 1
            and snapshot.values.get("counter") == 1
            and checkpoint_tuple is not None
            and not snapshot.next
        )
        checkpoint_tables_match = CHECKPOINT_TABLES <= public_tables
        business_tables_preserved = BUSINESS_TABLES <= public_tables
        setup_is_idempotent = migration_count == len(saver.MIGRATIONS)

        return {
            "saver_reuses_application_pool": saver_reuses_application_pool,
            "pool_connection_is_checkpointer_ready": pool_connection_is_checkpointer_ready,
            "checkpoint_tables_match": checkpoint_tables_match,
            "business_tables_preserved": business_tables_preserved,
            "setup_is_idempotent": setup_is_idempotent,
            "migration_count": migration_count,
            "expected_migration_count": len(saver.MIGRATIONS),
            "graph_round_trip_succeeded": graph_round_trip_succeeded,
            "elapsed_ms": _elapsed_ms(started_at),
        }
    finally:
        # Factory constructed all four resource types even though this smoke only opened
        # PostgreSQL. Closing lazy Neo4j/Redis and disposing the lazy ORM engine is safe and
        # keeps the ownership pattern identical to application lifespan cleanup.
        await resources.redis_client.aclose()
        await resources.neo4j_driver.close()
        await resources.orm_engine.dispose()
        await resources.postgres_pool.close()


def _run_smoke() -> dict[str, object]:
    """Create, migrate, verify and remove one isolated PostgreSQL database."""
    started_at = perf_counter()
    admin_database = settings.POSTGRES_DB
    test_database = f"deep_research_checkpointer_{uuid4().hex[:10]}"
    previous_override = os.environ.get("ALEMBIC_DATABASE_URL")
    database_created = False
    cleanup_ok = False
    alembic_ignores_checkpoint_tables = False
    checks: dict[str, bool | int | float]

    try:
        _create_database(admin_database, test_database)
        database_created = True
        os.environ["ALEMBIC_DATABASE_URL"] = _temporary_database_url(test_database)

        # Business migration runs first. Saver setup later adds external tables to the same
        # database, which is the exact situation Alembic's ownership filter must tolerate.
        alembic_config = Config(str(PROJECT_ROOT / "alembic.ini"))
        command.upgrade(alembic_config, "head")
        checks = asyncio.run(
            _exercise_checkpointer(test_database),
            loop_factory=_selector_loop_factory,
        )

        # command.check raises AutogenerateDiffsDetected if Alembic tries to drop or alter
        # saver-owned tables. No exception therefore proves the two migration owners coexist.
        command.check(alembic_config)
        alembic_ignores_checkpoint_tables = True
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
        raise RuntimeError("temporary checkpointer database cleanup failed")

    boolean_checks = tuple(value for value in checks.values() if isinstance(value, bool))
    ok = bool(boolean_checks) and all(boolean_checks) and alembic_ignores_checkpoint_tables and cleanup_ok
    return {
        "ok": ok,
        **checks,
        "alembic_ignores_checkpoint_tables": alembic_ignores_checkpoint_tables,
        "cleanup_ok": cleanup_ok,
        "total_elapsed_ms": _elapsed_ms(started_at),
    }


def main() -> int:
    """Print one sanitized JSON summary and return a process status code."""
    started_at = perf_counter()
    try:
        summary = _run_smoke()
    except Exception as error:
        # Database exceptions can include host/user details. Never print str(error).
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
