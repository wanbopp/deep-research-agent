"""使用真实 Redis、PostgreSQL、pgvector 和 Embedding 验收 MemoryService.

本脚本在随机临时数据库中执行 Alembic migration，并通过真实 provider 生成向量。
Redis 使用当前配置的真实实例，但只写入带随机用户身份哈希的短 TTL key；脚本会
精确记录并删除自己访问过的 key，不会执行 ``FLUSHDB`` 或扫描其他应用数据。

最终摘要只输出布尔证据、计数、模型名和耗时，不输出用户 UUID、查询正文、记忆
正文、向量、缓存 key、连接串或凭据。
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
from uuid import UUID, uuid4

import psycopg
from alembic import command
from alembic.config import Config
from psycopg import sql
from psycopg.conninfo import make_conninfo
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine
from sqlmodel import select

from app.core.config import settings
from app.infrastructure.cache import RedisCache
from app.infrastructure.database import build_orm_database_url
from app.infrastructure.embeddings import OpenAITextEmbedder
from app.infrastructure.memory import PostgresMemoryStore
from app.models import ChatSession, Memory, User
from app.schemas.memory import (
    MemoryCreate,
    MemoryItem,
    MemoryKind,
    MemoryQuery,
    MemorySearchResult,
    MemorySearchStatus,
)
from app.services.cache import Cache
from app.services.memory import MemoryStore
from app.services.memory_cache import (
    build_memory_generation_cache_key,
    build_memory_search_cache_key,
)
from app.services.memory_policy import MemoryRejectedError
from app.services.memory_service import MemoryService

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONNECTION_TIMEOUT_SECONDS = 10
UNAVAILABLE_PORT = 1

RELEVANT_MEMORY = "用户偏好使用中文解释 Python 技术问题"
SECOND_MEMORY = "用户要求技术回答保持简洁并给出关键步骤"
QUERY_TEXT = "请按照用户偏好解释 Python 技术问题"


class _TrackingCache:
    """委托真实 Cache，同时记录本次 smoke 需要精确清理的 key."""

    def __init__(self, delegate: Cache) -> None:
        """保存真实缓存适配器，不复制或解释缓存值."""
        self._delegate = delegate
        self.keys: set[str] = set()
        self.latest_search_write_key: str | None = None

    async def get(self, key: str) -> str | None:
        """记录并委托真实缓存读取."""
        self.keys.add(key)
        return await self._delegate.get(key)

    async def set(self, key: str, value: str, *, ttl_seconds: int) -> None:
        """记录并委托带 TTL 的真实缓存写入."""
        self.keys.add(key)
        if ":memory_search:" in key:
            self.latest_search_write_key = key
        await self._delegate.set(key, value, ttl_seconds=ttl_seconds)

    async def delete(self, key: str) -> None:
        """记录并委托真实幂等删除."""
        self.keys.add(key)
        await self._delegate.delete(key)

    async def cleanup(self) -> None:
        """只删除本次 smoke 记录过的哈希 key."""
        for key in tuple(self.keys):
            await self._delegate.delete(key)


class _CountingMemoryStore:
    """记录调用次数，同时把每次操作委托给真实 PostgresMemoryStore."""

    def __init__(self, delegate: MemoryStore) -> None:
        """保存真实存储协议并初始化计数器."""
        self._delegate = delegate
        self.search_calls = 0
        self.add_calls = 0
        self.delete_calls = 0

    async def search(
        self,
        *,
        user_id: UUID,
        query: MemoryQuery,
    ) -> tuple[MemoryItem, ...]:
        """计数后执行真实 Embedding 和 pgvector 检索."""
        self.search_calls += 1
        return await self._delegate.search(user_id=user_id, query=query)

    async def add(
        self,
        *,
        user_id: UUID,
        memory: MemoryCreate,
    ) -> MemoryItem:
        """计数后执行真实来源校验、Embedding 和 INSERT."""
        self.add_calls += 1
        return await self._delegate.add(user_id=user_id, memory=memory)

    async def delete(self, *, user_id: UUID, memory_id: UUID) -> None:
        """计数后执行真实 owner-scoped DELETE."""
        self.delete_calls += 1
        await self._delegate.delete(user_id=user_id, memory_id=memory_id)


def _elapsed_ms(started_at: float) -> float:
    """返回保留两位小数的毫秒耗时."""
    return round((perf_counter() - started_at) * 1000, 2)


def _conninfo(database: str) -> str:
    """构造只交给 psycopg 使用且绝不打印的连接参数."""
    return make_conninfo(
        host=settings.POSTGRES_HOST,
        port=settings.POSTGRES_PORT,
        dbname=database,
        user=settings.POSTGRES_USER,
        password=settings.POSTGRES_PASSWORD,
        connect_timeout=CONNECTION_TIMEOUT_SECONDS,
    )


def _temporary_database_url(database: str) -> str:
    """生成仅存于当前进程环境变量的 Alembic URL."""
    return build_orm_database_url(settings).set(database=database).render_as_string(hide_password=False)


def _create_database(*, admin_database: str, test_database: str) -> None:
    """使用安全标识符在 PostgreSQL 中创建随机临时数据库."""
    with psycopg.connect(_conninfo(admin_database), autocommit=True) as connection:
        connection.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(test_database)))


def _drop_database(*, admin_database: str, test_database: str) -> None:
    """终止临时连接并且只删除本次随机数据库."""
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


def _create_orm_runtime(
    database: str,
) -> tuple[AsyncEngine, async_sessionmaker[AsyncSession]]:
    """为随机临时数据库创建受限的 ORM Engine 和 Session 工厂."""
    engine = create_async_engine(
        build_orm_database_url(settings).set(database=database),
        pool_size=2,
        max_overflow=0,
        pool_pre_ping=True,
        connect_args={"connect_timeout": CONNECTION_TIMEOUT_SECONDS},
    )
    return engine, async_sessionmaker(
        bind=engine,
        class_=AsyncSession,
        expire_on_commit=False,
        autoflush=False,
    )


async def _seed_user_and_session(
    session_factory: async_sessionmaker[AsyncSession],
) -> tuple[User, ChatSession]:
    """在一个真实事务中创建 smoke 用户和 active 来源会话."""
    async with session_factory() as session:
        async with session.begin():
            user = User(
                email=f"memory-service-{uuid4().hex[:8]}@example.com",
                password_hash="smoke-only-password-hash",
            )
            session.add(user)
            await session.flush()
            chat_session = ChatSession(
                user_id=user.id,
                title="Memory service smoke",
            )
            session.add(chat_session)
            await session.flush()
    return user, chat_session


def _create_redis_client(*, port: int) -> Redis:
    """构造真实 Redis client；端口 1 用于安全制造连接失败."""
    return Redis(
        host=settings.REDIS_HOST,
        port=port,
        db=settings.REDIS_DB,
        password=settings.REDIS_PASSWORD or None,
        socket_connect_timeout=0.5,
        socket_timeout=0.5,
        decode_responses=True,
    )


async def _exercise_service(database: str) -> dict[str, object]:
    """执行真实缓存、写入、检索、降级和资源清理验收."""
    started_at = perf_counter()
    engine, session_factory = _create_orm_runtime(database)
    broken_engine: AsyncEngine | None = None
    redis_client = _create_redis_client(port=settings.REDIS_PORT)
    unavailable_redis = _create_redis_client(port=UNAVAILABLE_PORT)
    tracking_cache = _TrackingCache(RedisCache(redis_client))

    try:
        user, chat_session = await _seed_user_and_session(session_factory)
        embedder = OpenAITextEmbedder.from_settings(settings)
        real_store = PostgresMemoryStore(
            session_factory=session_factory,
            embedder=embedder,
        )
        counting_store = _CountingMemoryStore(real_store)
        service = MemoryService(
            counting_store,
            tracking_cache,
            search_cache_ttl_seconds=settings.MEMORY_SEARCH_CACHE_TTL_SECONDS,
            generation_ttl_seconds=settings.MEMORY_CACHE_GENERATION_TTL_SECONDS,
        )

        sensitive_before_calls = counting_store.add_calls
        try:
            await service.add(
                user_id=user.id,
                memory=MemoryCreate(
                    content="api_key=sk-smoke-placeholder-1234567890",
                    kind=MemoryKind.FACT,
                    source_thread_id=chat_session.id,
                ),
            )
        except MemoryRejectedError:
            sensitive_rejected = True
        else:
            sensitive_rejected = False
        sensitive_skipped_store = counting_store.add_calls == sensitive_before_calls

        first_item = await service.add(
            user_id=user.id,
            memory=MemoryCreate(
                content=RELEVANT_MEMORY,
                kind=MemoryKind.PREFERENCE,
                source_thread_id=chat_session.id,
            ),
        )
        query = MemoryQuery(text=QUERY_TEXT, limit=5)
        first_result = await service.search(user_id=user.id, query=query)
        calls_after_miss = counting_store.search_calls
        second_result = await service.search(user_id=user.id, query=query)
        cache_hit_skipped_store = (
            calls_after_miss == 1 and counting_store.search_calls == calls_after_miss and second_result == first_result
        )

        # 把“结构合法但属于另一用户”的值写入当前真实 Redis key。Service 必须
        # 删除污染值并回源 PostgreSQL，证明哈希 key 不是授权边界。
        poisoned_key = tracking_cache.latest_search_write_key
        if poisoned_key is None:
            raise RuntimeError("Memory search cache key was not written")
        wrong_owner_item = first_item.model_copy(update={"user_id": uuid4()})
        await tracking_cache.set(
            poisoned_key,
            MemorySearchResult(items=(wrong_owner_item,)).model_dump_json(),
            ttl_seconds=settings.MEMORY_SEARCH_CACHE_TTL_SECONDS,
        )
        calls_before_repair = counting_store.search_calls
        repaired_result = await service.search(user_id=user.id, query=query)
        owner_poison_repaired = counting_store.search_calls == calls_before_repair + 1 and all(
            item.user_id == user.id for item in repaired_result.items
        )

        generation_key = build_memory_generation_cache_key(user.id)
        generation_before_add = await tracking_cache.get(generation_key)
        second_item = await service.add(
            user_id=user.id,
            memory=MemoryCreate(
                content=SECOND_MEMORY,
                kind=MemoryKind.CONSTRAINT,
                source_thread_id=chat_session.id,
            ),
        )
        generation_after_add = await tracking_cache.get(generation_key)
        calls_before_post_add_search = counting_store.search_calls
        post_add_result = await service.search(user_id=user.id, query=query)
        generation_invalidated_search = (
            generation_before_add is not None
            and generation_after_add is not None
            and generation_after_add != generation_before_add
            and counting_store.search_calls == calls_before_post_add_search + 1
            and second_item in post_add_result.items
        )

        # 使用真实但不可连接的 Redis 端口，证明 CacheUnavailableError 只导致回源，
        # 不会把成功的 PostgreSQL/Embedding 搜索错误标成 degraded。
        fail_open_service = MemoryService(
            counting_store,
            RedisCache(unavailable_redis),
            search_cache_ttl_seconds=60,
            generation_ttl_seconds=3600,
        )
        calls_before_fail_open = counting_store.search_calls
        fail_open_result = await fail_open_service.search(
            user_id=user.id,
            query=MemoryQuery(text="Redis unavailable but memory store healthy"),
        )
        redis_failure_fell_back = (
            fail_open_result.status is MemorySearchStatus.AVAILABLE
            and counting_store.search_calls == calls_before_fail_open + 1
        )

        # 构造真实但不可连接的 PostgreSQL Engine。PostgresMemoryStore 仍会先调用
        # 真实 Embedding provider，随后把驱动故障映射为 MemoryUnavailableError；
        # MemoryService 必须返回 degraded，且绝不能缓存这个空结果。
        broken_engine = create_async_engine(
            build_orm_database_url(settings).set(
                host="127.0.0.1",
                port=UNAVAILABLE_PORT,
                database=database,
            ),
            pool_size=1,
            max_overflow=0,
            pool_timeout=1,
            connect_args={"connect_timeout": 1},
        )
        broken_session_factory = async_sessionmaker(
            bind=broken_engine,
            class_=AsyncSession,
            expire_on_commit=False,
            autoflush=False,
        )
        broken_store = PostgresMemoryStore(
            session_factory=broken_session_factory,
            embedder=embedder,
        )
        degraded_user_id = uuid4()
        degraded_query = MemoryQuery(text="Memory backend degraded smoke")
        degraded_service = MemoryService(
            broken_store,
            tracking_cache,
            search_cache_ttl_seconds=60,
            generation_ttl_seconds=3600,
        )
        degraded_result = await degraded_service.search(
            user_id=degraded_user_id,
            query=degraded_query,
        )
        degraded_generation = await tracking_cache.get(build_memory_generation_cache_key(degraded_user_id))
        if degraded_generation is None:
            raise RuntimeError("Degraded search generation was not created")
        degraded_cache_key = build_memory_search_cache_key(
            user_id=degraded_user_id,
            generation=degraded_generation,
            query=degraded_query,
        )
        degraded_not_cached = await tracking_cache.get(degraded_cache_key) is None
        store_failure_is_degraded = (
            degraded_result.is_degraded and degraded_result.error_code == "MEMORY_UNAVAILABLE" and degraded_not_cached
        )

        generation_before_delete = await tracking_cache.get(generation_key)
        await service.delete(user_id=user.id, memory_id=first_item.id)
        generation_after_delete = await tracking_cache.get(generation_key)
        calls_before_post_delete_search = counting_store.search_calls
        post_delete_result = await service.search(user_id=user.id, query=query)
        delete_invalidated_search = (
            generation_before_delete is not None
            and generation_after_delete is not None
            and generation_after_delete != generation_before_delete
            and counting_store.search_calls == calls_before_post_delete_search + 1
            and all(item.id != first_item.id for item in post_delete_result.items)
        )

        async with session_factory() as session:
            rows = (await session.execute(select(Memory))).scalars().all()
        database_state_matches = len(rows) == 1 and rows[0].id == second_item.id
        cache_keys_hide_inputs = all(
            str(user.id) not in key and QUERY_TEXT not in key and RELEVANT_MEMORY not in key
            for key in tracking_cache.keys
        )

        checks = (
            sensitive_rejected,
            sensitive_skipped_store,
            cache_hit_skipped_store,
            owner_poison_repaired,
            generation_invalidated_search,
            redis_failure_fell_back,
            store_failure_is_degraded,
            delete_invalidated_search,
            database_state_matches,
            cache_keys_hide_inputs,
        )
        summary = {
            "ok": all(checks),
            "model": settings.EMBEDDING_MODEL,
            "dimensions": settings.EMBEDDING_DIMENSIONS,
            "sensitive_rejected": sensitive_rejected,
            "sensitive_skipped_store": sensitive_skipped_store,
            "cache_hit_skipped_store": cache_hit_skipped_store,
            "owner_poison_repaired": owner_poison_repaired,
            "generation_invalidated_search": generation_invalidated_search,
            "redis_failure_fell_back": redis_failure_fell_back,
            "store_failure_is_degraded": store_failure_is_degraded,
            "degraded_not_cached": degraded_not_cached,
            "delete_invalidated_search": delete_invalidated_search,
            "database_state_matches": database_state_matches,
            "cache_keys_hide_inputs": cache_keys_hide_inputs,
            "store_calls": {
                "search": counting_store.search_calls,
                "add": counting_store.add_calls,
                "delete": counting_store.delete_calls,
            },
            "elapsed_ms": _elapsed_ms(started_at),
        }
    finally:
        # 先清理本次 Redis key，再关闭客户端和数据库连接池。任何清理失败都让
        # smoke 失败，不能返回乐观的 ok=true。
        await tracking_cache.cleanup()
        await unavailable_redis.aclose()
        await redis_client.aclose()
        if broken_engine is not None:
            await broken_engine.dispose()
        await engine.dispose()

    summary["resource_cleanup_ok"] = True
    return summary


def _run_async(
    coroutine: Coroutine[Any, Any, dict[str, object]],
) -> dict[str, object]:
    """运行异步验收，并兼容 Windows psycopg 的 Selector event loop."""
    if sys.platform == "win32":
        with asyncio.Runner(loop_factory=lambda: asyncio.SelectorEventLoop(selectors.SelectSelector())) as runner:
            return runner.run(coroutine)
    return asyncio.run(coroutine)


def main() -> int:
    """编排随机数据库、migration、真实服务验收和异常安全清理."""
    started_at = perf_counter()
    admin_database = settings.POSTGRES_DB
    test_database = f"deep_research_memory_service_{uuid4().hex[:10]}"
    previous_override = os.environ.get("ALEMBIC_DATABASE_URL")
    database_created = False
    database_cleanup_ok = False

    try:
        _create_database(
            admin_database=admin_database,
            test_database=test_database,
        )
        database_created = True
        os.environ["ALEMBIC_DATABASE_URL"] = _temporary_database_url(test_database)
        command.upgrade(Config(str(PROJECT_ROOT / "alembic.ini")), "head")
        summary = _run_async(_exercise_service(test_database))
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
                database_cleanup_ok = False
            else:
                database_cleanup_ok = True

    summary["database_cleanup_ok"] = database_cleanup_ok
    summary["ok"] = bool(summary["ok"] and database_cleanup_ok)
    print(json.dumps(summary, ensure_ascii=False))
    return 0 if summary["ok"] is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
