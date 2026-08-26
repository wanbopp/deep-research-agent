"""在真实 Embedding provider 和随机 PostgreSQL 数据库中验收长期记忆存储.

这不是 fake 单元测试，而是一条隔离的端到端 smoke：

1. 在已配置 PostgreSQL 实例中创建随机临时数据库；
2. 对临时库执行真实 Alembic ``upgrade head``，从而创建 ``vector`` 扩展和表；
3. 使用真实 provider 生成 1536 维向量；
4. 验证用户隔离、来源会话所有权、kind 过滤、语义排序和幂等删除；
5. 无论成功还是失败，都释放 Engine 并删除临时数据库。

脚本不会输出 API Key、数据库密码、用户 UUID、记忆正文或向量内容。最终 JSON
只包含模型名、计数、布尔验收项和耗时，适合保留为教学证据。
"""

import asyncio
import json
import os
import selectors
import sys
from collections.abc import Coroutine
from pathlib import Path
from time import perf_counter
from typing import Any
from uuid import uuid4

import psycopg
from alembic import command
from alembic.config import Config
from psycopg import sql
from psycopg.conninfo import make_conninfo
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine
from sqlmodel import select

from app.core.config import settings
from app.infrastructure.database import build_orm_database_url
from app.infrastructure.embeddings import OpenAITextEmbedder
from app.infrastructure.memory import PostgresMemoryStore
from app.models import ChatSession, Memory, User
from app.schemas.memory import MemoryCreate, MemoryKind, MemoryQuery
from app.services.memory import MemorySourceNotFoundError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONNECTION_TIMEOUT_SECONDS = 10

# 使用完全相同的文本作为查询与目标记忆，可以让真实 Embedding 排序验收稳定：
# 相同文本的向量距离应最小。常量只存在进程内，最终摘要不会输出正文。
RELEVANT_MEMORY = "用户偏好使用中文解释 Python 技术问题"
UNRELATED_MEMORY = "用户常用的代码编辑器是 PyCharm"


def _elapsed_ms(started_at: float) -> float:
    """返回保留两位小数的毫秒耗时."""
    return round((perf_counter() - started_at) * 1000, 2)


def _admin_conninfo(database: str) -> str:
    """构造只交给 psycopg 使用且绝不打印的管理连接参数."""
    return make_conninfo(
        host=settings.POSTGRES_HOST,
        port=settings.POSTGRES_PORT,
        dbname=database,
        user=settings.POSTGRES_USER,
        password=settings.POSTGRES_PASSWORD,
        connect_timeout=CONNECTION_TIMEOUT_SECONDS,
    )


def _temporary_database_url(database: str) -> str:
    """生成仅存于当前进程的临时数据库 URL."""
    return build_orm_database_url(settings).set(database=database).render_as_string(hide_password=False)


def _create_database(*, admin_database: str, test_database: str) -> None:
    """使用 autocommit 创建随机临时数据库.

    Args:
        admin_database: 仅作为 ``CREATE DATABASE`` 管理入口的现有数据库。
        test_database: 带随机后缀且只属于本次 smoke 的数据库名。
    """
    with psycopg.connect(
        _admin_conninfo(admin_database),
        autocommit=True,
    ) as connection:
        # 数据库名是 SQL 标识符，必须用 Identifier 引用，不能作为普通参数或
        # 字符串拼接，否则会引入 SQL 注入和特殊字符解析问题。
        connection.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(test_database)))


def _drop_database(*, admin_database: str, test_database: str) -> None:
    """终止临时库残留连接并删除且只删除本次随机数据库."""
    with psycopg.connect(
        _admin_conninfo(admin_database),
        autocommit=True,
    ) as connection:
        connection.execute(
            """
            SELECT pg_terminate_backend(pid)
            FROM pg_stat_activity
            WHERE datname = %s AND pid <> pg_backend_pid()
            """,
            (test_database,),
        )
        connection.execute(sql.SQL("DROP DATABASE IF EXISTS {}").format(sql.Identifier(test_database)))


def _vector_extension_exists(database: str) -> bool:
    """从 PostgreSQL catalog 验证真实 vector 扩展已经启用."""
    with psycopg.connect(_admin_conninfo(database)) as connection:
        row = connection.execute("SELECT EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'vector')").fetchone()
    return row == (True,)


def _create_memory_runtime(
    database: str,
) -> tuple[
    AsyncEngine,
    async_sessionmaker[AsyncSession],
    PostgresMemoryStore,
]:
    """为临时数据库构造 ORM 和真实 MemoryStore 运行时.

    Args:
        database: 本次 smoke 随机数据库名。

    Returns:
        需要由调用方关闭的 Engine、Session 工厂和待验收 store。
    """
    engine = create_async_engine(
        build_orm_database_url(settings).set(database=database),
        pool_size=2,
        max_overflow=0,
        pool_pre_ping=True,
        connect_args={"connect_timeout": CONNECTION_TIMEOUT_SECONDS},
    )
    session_factory = async_sessionmaker(
        bind=engine,
        class_=AsyncSession,
        expire_on_commit=False,
        autoflush=False,
    )
    embedder = OpenAITextEmbedder.from_settings(settings)
    store = PostgresMemoryStore(
        session_factory=session_factory,
        embedder=embedder,
    )
    return engine, session_factory, store


async def _seed_users_and_sessions(
    session_factory: async_sessionmaker[AsyncSession],
) -> tuple[User, ChatSession, User, ChatSession]:
    """在一个事务中创建两个用户及各自的来源会话."""
    async with session_factory() as session:
        async with session.begin():
            first_user = User(
                email=f"memory-a-{uuid4().hex[:8]}@example.com",
                # 临时库不会执行认证；这里只满足不可空数据库列，值不会输出。
                password_hash="smoke-only-password-hash",
            )
            second_user = User(
                email=f"memory-b-{uuid4().hex[:8]}@example.com",
                password_hash="smoke-only-password-hash",
            )
            session.add_all((first_user, second_user))
            await session.flush()

            first_session = ChatSession(
                user_id=first_user.id,
                title="Memory smoke A",
            )
            second_session = ChatSession(
                user_id=second_user.id,
                title="Memory smoke B",
            )
            session.add_all((first_session, second_session))
            await session.flush()

    return first_user, first_session, second_user, second_session


async def _exercise_memory_store(database: str) -> dict[str, object]:
    """对真实 provider 和临时数据库执行 MemoryStore 行为验收."""
    started_at = perf_counter()
    engine, session_factory, store = _create_memory_runtime(database)

    try:
        first_user, first_session, second_user, second_session = await _seed_users_and_sessions(session_factory)

        # 三次 add 会产生真实 Embedding 请求。第二个用户保存与查询完全相同的
        # 文本，用来证明 user_id 过滤发生在数据库内部，而不是结果返回后。
        first_relevant = await store.add(
            user_id=first_user.id,
            memory=MemoryCreate(
                content=RELEVANT_MEMORY,
                kind=MemoryKind.PREFERENCE,
                source_thread_id=first_session.id,
            ),
        )
        await store.add(
            user_id=first_user.id,
            memory=MemoryCreate(
                content=UNRELATED_MEMORY,
                kind=MemoryKind.FACT,
                source_thread_id=first_session.id,
            ),
        )
        second_relevant = await store.add(
            user_id=second_user.id,
            memory=MemoryCreate(
                content=RELEVANT_MEMORY,
                kind=MemoryKind.PREFERENCE,
                source_thread_id=second_session.id,
            ),
        )

        first_results = await store.search(
            user_id=first_user.id,
            query=MemoryQuery(text=RELEVANT_MEMORY, limit=3),
        )
        owner_scope_matches = bool(first_results) and all(item.user_id == first_user.id for item in first_results)
        semantic_order_matches = bool(first_results) and first_results[0].id == first_relevant.id

        fact_results = await store.search(
            user_id=first_user.id,
            query=MemoryQuery(
                text=RELEVANT_MEMORY,
                kinds=frozenset({MemoryKind.FACT}),
                limit=3,
            ),
        )
        kind_filter_matches = bool(fact_results) and all(item.kind is MemoryKind.FACT for item in fact_results)

        try:
            await store.add(
                user_id=first_user.id,
                memory=MemoryCreate(
                    content="不应写入的跨用户来源",
                    kind=MemoryKind.CONSTRAINT,
                    source_thread_id=second_session.id,
                ),
            )
        except MemorySourceNotFoundError:
            cross_user_source_rejected = True
        else:
            cross_user_source_rejected = False

        # 第二个用户不能删除第一个用户的记忆。随后所有者删除两次都成功，证明
        # delete 同时满足 owner-scoped 和幂等语义。
        await store.delete(
            user_id=second_user.id,
            memory_id=first_relevant.id,
        )
        # 这里直接读取临时库，避免为了验证 DELETE 再消耗一次真实模型请求。
        # 读取结果只用于布尔断言，记忆正文和向量仍不会进入最终输出。
        async with session_factory() as session:
            after_cross_delete = (
                await session.execute(select(Memory).where(Memory.id == first_relevant.id))
            ).scalar_one_or_none()
        cross_user_delete_isolated = after_cross_delete is not None

        await store.delete(
            user_id=first_user.id,
            memory_id=first_relevant.id,
        )
        await store.delete(
            user_id=first_user.id,
            memory_id=first_relevant.id,
        )

        # 直接读取临时库中的内部行，只验证维度和剩余数量，不输出向量内容。
        async with session_factory() as session:
            rows = (await session.execute(select(Memory))).scalars().all()
        second_user_unchanged = any(row.id == second_relevant.id for row in rows)
        vector_dimensions_match = bool(rows) and all(
            len(row.embedding) == settings.EMBEDDING_DIMENSIONS for row in rows
        )
        owner_delete_completed = all(row.id != first_relevant.id for row in rows)

        checks = (
            owner_scope_matches,
            semantic_order_matches,
            kind_filter_matches,
            cross_user_source_rejected,
            cross_user_delete_isolated,
            owner_delete_completed,
            second_user_unchanged,
            vector_dimensions_match,
        )
        return {
            "ok": all(checks),
            "model": settings.EMBEDDING_MODEL,
            "dimensions": settings.EMBEDDING_DIMENSIONS,
            "stored_count": len(rows),
            "owner_scope_matches": owner_scope_matches,
            "semantic_order_matches": semantic_order_matches,
            "kind_filter_matches": kind_filter_matches,
            "cross_user_source_rejected": cross_user_source_rejected,
            "cross_user_delete_isolated": cross_user_delete_isolated,
            "owner_delete_completed": owner_delete_completed,
            "second_user_unchanged": second_user_unchanged,
            "vector_dimensions_match": vector_dimensions_match,
            "elapsed_ms": _elapsed_ms(started_at),
        }
    finally:
        # Engine 持有连接池；先 dispose，外层才能删除临时数据库。
        await engine.dispose()


def _run_async(
    coroutine: Coroutine[Any, Any, dict[str, object]],
) -> dict[str, object]:
    """运行异步 smoke，并兼容 Windows 上 psycopg 的 Selector event loop."""
    if sys.platform == "win32":
        with asyncio.Runner(loop_factory=lambda: asyncio.SelectorEventLoop(selectors.SelectSelector())) as runner:
            return runner.run(coroutine)
    return asyncio.run(coroutine)


def main() -> int:
    """编排临时库、migration、真实 store 验收和异常安全清理."""
    started_at = perf_counter()
    admin_database = settings.POSTGRES_DB
    test_database = f"deep_research_memory_{uuid4().hex[:12]}"
    previous_override = os.environ.get("ALEMBIC_DATABASE_URL")
    database_created = False
    cleanup_ok = False

    try:
        _create_database(
            admin_database=admin_database,
            test_database=test_database,
        )
        database_created = True

        os.environ["ALEMBIC_DATABASE_URL"] = _temporary_database_url(test_database)
        alembic_config = Config(str(PROJECT_ROOT / "alembic.ini"))
        command.upgrade(alembic_config, "head")

        summary = _run_async(_exercise_memory_store(test_database))
        summary["vector_extension_exists"] = _vector_extension_exists(test_database)
        summary["ok"] = bool(summary["ok"] and summary["vector_extension_exists"])
    except Exception as exc:
        summary = {
            "ok": False,
            "error_type": type(exc).__name__,
            "elapsed_ms": _elapsed_ms(started_at),
        }
    finally:
        if previous_override is None:
            os.environ.pop("ALEMBIC_DATABASE_URL", None)
        else:
            os.environ["ALEMBIC_DATABASE_URL"] = previous_override

        if database_created:
            try:
                _drop_database(
                    admin_database=admin_database,
                    test_database=test_database,
                )
            except Exception:
                cleanup_ok = False
            else:
                cleanup_ok = True

    summary["cleanup_ok"] = cleanup_ok
    summary["ok"] = bool(summary["ok"] and cleanup_ok)
    print(json.dumps(summary, ensure_ascii=False))
    return 0 if summary["ok"] is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
