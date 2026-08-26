"""使用真实 PostgreSQL 与 Redis 验收 ChatSession 列表 cache-aside.

脚本创建随机临时数据库并执行正式 Alembic migration，Redis 则使用当前受控
开发配置。它不调用 LLM、Graph 或工具，只验证业务缓存接线：

1. 首次列表从 PostgreSQL 回源并写入 Redis；
2. 第二次列表命中旧副本，证明读路径确实经过缓存；
3. service create 提交后主动失效，下一次读取看到最新数据库状态；
4. 损坏 JSON 被拒绝、删除并通过 PostgreSQL 修复；
5. Redis 不可用时读取仍能 fail-open；
6. 不同 user_id 使用不同摘要 key；
7. 删除阶段 1 提交 deleting 后立即失效列表缓存。

最终 JSON 只输出布尔值、数量和耗时，不输出邮箱、UUID、缓存 key、连接信息、
缓存内容或底层异常文本。
"""

import asyncio
import json
import os
import selectors
from pathlib import Path
from time import perf_counter
from uuid import UUID, uuid4

import psycopg
from alembic import command
from alembic.config import Config
from fastapi import FastAPI
from psycopg import sql
from psycopg.conninfo import make_conninfo
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.config import Settings, settings
from app.infrastructure.database import build_orm_database_url
from app.infrastructure.lifespan import (
    get_application_cache,
    get_application_chat_cleanup_service,
    get_application_resources,
    lifespan,
)
from app.repositories import ChatSessionRepository, UserRepository
from app.schemas.chat_session import (
    ChatSessionCreateRequest,
    ChatSessionListResponse,
)
from app.services.cache import Cache, CacheUnavailableError
from app.services.chat_session_cache import build_chat_session_list_cache_key
from app.services.chat_sessions import ChatSessionService

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONNECTION_TIMEOUT_SECONDS = 10
TOTAL_TIMEOUT_SECONDS = 30.0
TEMPORARY_DATABASE_PREFIX = "deep_research_chat_cache_"


class _UnavailableCache:
    """确定性模拟缓存不可用，但不替换真实 PostgreSQL 成功路径.

    该对象只用于 fail-open 分支。真实 hit、TTL、delete 和序列化仍由同一 smoke
    前面的 RedisCache 完成，因此这里不是模型、数据库或成功缓存路径的 fake。
    """

    async def get(self, key: str) -> str | None:
        """忽略 key 并报告稳定缓存故障."""
        raise CacheUnavailableError

    async def set(self, key: str, value: str, *, ttl_seconds: int) -> None:
        """忽略写入参数并报告稳定缓存故障."""
        raise CacheUnavailableError

    async def delete(self, key: str) -> None:
        """忽略失效 key 并报告稳定缓存故障."""
        raise CacheUnavailableError


def _elapsed_ms(started_at: float) -> float:
    """返回不含连接信息的毫秒耗时."""
    return round((perf_counter() - started_at) * 1000, 2)


def _selector_loop_factory() -> asyncio.AbstractEventLoop:
    """创建 psycopg 异步模式在 Windows 上需要的 Selector 事件循环."""
    return asyncio.SelectorEventLoop(selectors.SelectSelector())


def _conninfo(database: str) -> str:
    """构造只交给 psycopg 使用且绝不输出的连接字符串.

    Args:
        database: 已存在的管理数据库或本次随机测试数据库名称。

    Returns:
        使用当前 Git 忽略配置安全转义后的 PostgreSQL conninfo。
    """
    return make_conninfo(
        host=settings.POSTGRES_HOST,
        port=settings.POSTGRES_PORT,
        dbname=database,
        user=settings.POSTGRES_USER,
        password=settings.POSTGRES_PASSWORD,
        connect_timeout=CONNECTION_TIMEOUT_SECONDS,
    )


def _temporary_database_url(database: str) -> str:
    """构造仅供本次 Alembic migration 使用的临时 URL."""
    return build_orm_database_url(settings).set(database=database).render_as_string(hide_password=False)


def _create_database(admin_database: str, test_database: str) -> None:
    """在 autocommit 连接中创建随机隔离数据库.

    ``CREATE DATABASE`` 不能位于普通事务中；``sql.Identifier`` 确保随机名称
    始终被当作数据库标识符，而不是可执行 SQL 片段。
    """
    with psycopg.connect(_conninfo(admin_database), autocommit=True) as connection:
        connection.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(test_database)))


def _drop_database(admin_database: str, test_database: str) -> None:
    """终止残留连接并只删除本次随机测试数据库."""
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
    """创建只修改临时数据库名和连接预算的运行配置.

    Args:
        database: 已完成业务 Alembic migration 的随机数据库。

    Returns:
        仍使用真实 Redis、Neo4j 和其他应用配置的 Settings 副本。
    """
    config = Settings()
    config.POSTGRES_DB = database
    config.POSTGRES_PSYCOPG_POOL_SIZE = 2
    config.POSTGRES_ORM_POOL_SIZE = 2
    config.POSTGRES_ORM_MAX_OVERFLOW = 0
    return config


async def _create_user(
    session_factory: async_sessionmaker[AsyncSession],
    *,
    email: str,
) -> UUID:
    """在真实数据库中提交一个 smoke 用户.

    Args:
        session_factory: lifespan 创建的请求级 ORM Session 工厂。
        email: 只存在于临时数据库且不会进入输出的随机邮箱。

    Returns:
        已提交用户的 UUID，后续作为可信 service user_id 使用。
    """
    async with session_factory() as session:
        async with session.begin():
            user = await UserRepository(session).create(
                email=email,
                # 本 smoke 不测试登录或哈希算法，只需要满足数据库非空约束。
                password_hash="$argon2id$cache-smoke-non-credential",
            )
    return user.id


async def _create_through_service(
    session_factory: async_sessionmaker[AsyncSession],
    cache: Cache,
    *,
    user_id: UUID,
    title: str,
    ttl_seconds: int,
) -> UUID:
    """通过正式 ChatSessionService 创建并触发提交后失效."""
    async with session_factory() as session:
        service = ChatSessionService(
            session,
            cache=cache,
            cache_ttl_seconds=ttl_seconds,
        )
        response = await service.create(
            user_id=user_id,
            request=ChatSessionCreateRequest(title=title),
        )
    return response.thread_id


async def _create_directly_in_database(
    session_factory: async_sessionmaker[AsyncSession],
    *,
    user_id: UUID,
    title: str,
) -> UUID:
    """绕过 service 创建会话，用于证明第二次 list 确实命中旧缓存.

    这不是正式业务写入方式。它故意不执行缓存失效：如果下一次 list 仍只返回
    缓存中的旧数量，就能证明命中来自 Redis，而不是再次查询 PostgreSQL。
    """
    async with session_factory() as session:
        async with session.begin():
            chat_session = await ChatSessionRepository(session).create(
                user_id=user_id,
                title=title,
            )
    return chat_session.id


async def _list_through_service(
    session_factory: async_sessionmaker[AsyncSession],
    cache: Cache,
    *,
    user_id: UUID,
    ttl_seconds: int,
) -> ChatSessionListResponse:
    """使用全新请求级 Session 执行正式 cache-aside 列表读取."""
    async with session_factory() as session:
        service = ChatSessionService(
            session,
            cache=cache,
            cache_ttl_seconds=ttl_seconds,
        )
        return await service.list_owned(user_id=user_id)


async def _cleanup_users(
    session_factory: async_sessionmaker[AsyncSession],
    user_ids: tuple[UUID, ...],
) -> None:
    """先删除用户会话，再删除临时用户，避免依赖未声明的级联行为."""
    async with session_factory() as session:
        async with session.begin():
            users = UserRepository(session)
            chat_sessions = ChatSessionRepository(session)
            for user_id in user_ids:
                for chat_session in await chat_sessions.list_by_user(user_id):
                    await session.delete(chat_session)
                user = await users.get_by_id(user_id)
                if user is not None:
                    await session.delete(user)


async def _exercise_cache(database: str) -> dict[str, bool | float | int]:
    """在生产 lifespan 中执行完整 cache-aside 行为验收.

    Args:
        database: 已迁移的随机 PostgreSQL 数据库。

    Returns:
        只包含行为布尔值、数量和耗时的安全摘要。
    """
    started_at = perf_counter()
    app = FastAPI()
    config = _runtime_settings(database)
    created_user_ids: list[UUID] = []
    cache_keys: list[str] = []

    async with lifespan(app, config=config):
        resources = get_application_resources(app)
        cache = get_application_cache(app)
        cleanup_service = get_application_chat_cleanup_service(app)

        try:
            async with asyncio.timeout(TOTAL_TIMEOUT_SECONDS):
                suffix = uuid4().hex
                owner_id = await _create_user(
                    resources.orm_session_factory,
                    email=f"cache-owner-{suffix}@example.com",
                )
                other_id = await _create_user(
                    resources.orm_session_factory,
                    email=f"cache-other-{suffix}@example.com",
                )
                created_user_ids.extend((owner_id, other_id))

                owner_key = build_chat_session_list_cache_key(owner_id)
                other_key = build_chat_session_list_cache_key(other_id)
                cache_keys.extend((owner_key, other_key))

                first_id = await _create_through_service(
                    resources.orm_session_factory,
                    cache,
                    user_id=owner_id,
                    title="首次缓存会话",
                    ttl_seconds=config.CACHE_TTL_SECONDS,
                )
                first_list = await _list_through_service(
                    resources.orm_session_factory,
                    cache,
                    user_id=owner_id,
                    ttl_seconds=config.CACHE_TTL_SECONDS,
                )
                first_read_populated_cache = (
                    len(first_list.sessions) == 1
                    and first_list.sessions[0].thread_id == first_id
                    and await resources.redis_client.get(owner_key) is not None
                )

                direct_id = await _create_directly_in_database(
                    resources.orm_session_factory,
                    user_id=owner_id,
                    title="绕过失效的数据库会话",
                )
                cached_list = await _list_through_service(
                    resources.orm_session_factory,
                    cache,
                    user_id=owner_id,
                    ttl_seconds=config.CACHE_TTL_SECONDS,
                )
                second_read_hit_cache = (
                    len(cached_list.sessions) == 1 and cached_list.sessions[0].thread_id == first_id
                )

                service_created_id = await _create_through_service(
                    resources.orm_session_factory,
                    cache,
                    user_id=owner_id,
                    title="触发主动失效的会话",
                    ttl_seconds=config.CACHE_TTL_SECONDS,
                )
                refreshed_list = await _list_through_service(
                    resources.orm_session_factory,
                    cache,
                    user_id=owner_id,
                    ttl_seconds=config.CACHE_TTL_SECONDS,
                )
                refreshed_ids = {item.thread_id for item in refreshed_list.sessions}
                create_invalidated_cache = refreshed_ids == {
                    first_id,
                    direct_id,
                    service_created_id,
                }

                # 手工写入损坏 JSON，模拟旧 schema 或外部误写。service 必须拒绝
                # 该值、回源数据库，并把修复后的 Pydantic JSON 重新写回 Redis。
                await resources.redis_client.set(
                    owner_key,
                    "{invalid-cache-json",
                    ex=config.CACHE_TTL_SECONDS,
                )
                repaired_list = await _list_through_service(
                    resources.orm_session_factory,
                    cache,
                    user_id=owner_id,
                    ttl_seconds=config.CACHE_TTL_SECONDS,
                )
                repaired_raw = await resources.redis_client.get(owner_key)
                repaired_model = (
                    ChatSessionListResponse.model_validate_json(repaired_raw)
                    if isinstance(repaired_raw, str)
                    else None
                )
                invalid_payload_repaired = (
                    len(repaired_list.sessions) == 3
                    and repaired_model is not None
                    and len(repaired_model.sessions) == 3
                )

                # 失败 Cache 同时让 get 和 set 抛错。list_owned 仍使用真实 PostgreSQL
                # 返回结果，证明 fail-open 不依赖 Redis 成功路径或内存旧值。
                fallback_list = await _list_through_service(
                    resources.orm_session_factory,
                    _UnavailableCache(),
                    user_id=owner_id,
                    ttl_seconds=config.CACHE_TTL_SECONDS,
                )
                backend_failure_fell_back = len(fallback_list.sessions) == 3

                other_session_id = await _create_through_service(
                    resources.orm_session_factory,
                    cache,
                    user_id=other_id,
                    title="其他用户会话",
                    ttl_seconds=config.CACHE_TTL_SECONDS,
                )
                other_list = await _list_through_service(
                    resources.orm_session_factory,
                    cache,
                    user_id=other_id,
                    ttl_seconds=config.CACHE_TTL_SECONDS,
                )
                owner_after_other = await _list_through_service(
                    resources.orm_session_factory,
                    cache,
                    user_id=owner_id,
                    ttl_seconds=config.CACHE_TTL_SECONDS,
                )
                user_keys_are_isolated = (
                    owner_key != other_key
                    and len(other_list.sessions) == 1
                    and other_list.sessions[0].thread_id == other_session_id
                    and len(owner_after_other.sessions) == 3
                )

                deleting_id = await _create_through_service(
                    resources.orm_session_factory,
                    cache,
                    user_id=owner_id,
                    title="删除失效会话",
                    ttl_seconds=config.CACHE_TTL_SECONDS,
                )
                before_delete = await _list_through_service(
                    resources.orm_session_factory,
                    cache,
                    user_id=owner_id,
                    ttl_seconds=config.CACHE_TTL_SECONDS,
                )
                await cleanup_service.delete_owned(
                    session_id=deleting_id,
                    user_id=owner_id,
                )
                cache_absent_after_deleting_commit = await resources.redis_client.get(owner_key) is None
                after_delete = await _list_through_service(
                    resources.orm_session_factory,
                    cache,
                    user_id=owner_id,
                    ttl_seconds=config.CACHE_TTL_SECONDS,
                )
                after_delete_ids = {item.thread_id for item in after_delete.sessions}
                deleting_invalidated_cache = (
                    len(before_delete.sessions) == 4
                    and cache_absent_after_deleting_commit
                    and deleting_id not in after_delete_ids
                    and len(after_delete_ids) == 3
                )

            return {
                "first_read_populated_cache": first_read_populated_cache,
                "second_read_hit_cache": second_read_hit_cache,
                "create_invalidated_cache": create_invalidated_cache,
                "invalid_payload_repaired": invalid_payload_repaired,
                "backend_failure_fell_back": backend_failure_fell_back,
                "user_keys_are_isolated": user_keys_are_isolated,
                "deleting_invalidated_cache": deleting_invalidated_cache,
                "owner_session_count": len(after_delete_ids),
                "other_session_count": len(other_list.sessions),
                "within_total_budget": _elapsed_ms(started_at) <= TOTAL_TIMEOUT_SECONDS * 1000,
                "elapsed_ms": _elapsed_ms(started_at),
            }
        finally:
            # smoke 拥有临时业务行和缓存 key。先删除缓存，再清理数据库行；
            # lifespan 随后负责关闭共享 client 和连接池。
            cleanup_errors: list[Exception] = []
            for key in cache_keys:
                try:
                    await cache.delete(key)
                except Exception as error:
                    cleanup_errors.append(error)

            try:
                await _cleanup_users(
                    resources.orm_session_factory,
                    tuple(created_user_ids),
                )
            except Exception as error:
                cleanup_errors.append(error)

            if cleanup_errors:
                raise RuntimeError("ChatSession cache smoke resource cleanup failed")


def _run_smoke() -> dict[str, object]:
    """创建临时数据库、迁移、运行异步验收并保证最终删除数据库."""
    started_at = perf_counter()
    admin_database = settings.POSTGRES_DB
    test_database = f"{TEMPORARY_DATABASE_PREFIX}{uuid4().hex[:10]}"
    previous_override = os.environ.get("ALEMBIC_DATABASE_URL")
    database_created = False
    cleanup_ok = False
    checks: dict[str, bool | float | int] = {}

    try:
        _create_database(admin_database, test_database)
        database_created = True
        os.environ["ALEMBIC_DATABASE_URL"] = _temporary_database_url(test_database)
        command.upgrade(Config(str(PROJECT_ROOT / "alembic.ini")), "head")
        checks = asyncio.run(
            _exercise_cache(test_database),
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
        raise RuntimeError("temporary ChatSession cache database cleanup failed")

    boolean_checks = tuple(value for value in checks.values() if isinstance(value, bool))
    return {
        "ok": bool(boolean_checks) and all(boolean_checks) and cleanup_ok,
        **checks,
        "cleanup_ok": cleanup_ok,
        "total_elapsed_ms": _elapsed_ms(started_at),
    }


def main() -> int:
    """打印一行安全 JSON，并返回适合 PowerShell 与 CI 的退出码."""
    started_at = perf_counter()
    try:
        summary = _run_smoke()
    except Exception as error:
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
