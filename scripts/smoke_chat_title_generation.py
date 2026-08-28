"""使用真实 PostgreSQL 和真实 provider 验收会话自动命名.

脚本在随机临时数据库执行完整 Alembic migration，并用两个独立后台提交器模拟
两个 worker。它们同时为同一默认标题会话申请命名，只有 PostgreSQL 原子 claim
的赢家可以调用真实结构化模型。随后验证人工标题保护、过期租约恢复、缓存失效
和任务收敛。最后使用真实 provider transport timeout 验证失败任务会释放自己的租约，
使会话保持可重试状态。

最终摘要只输出布尔证据、调用计数、模型名和耗时；不输出用户 UUID、会话 UUID、
标题正文、Prompt、回复、缓存 key、连接串或凭据。
"""

import asyncio
import json
import os
import selectors
import sys
from collections.abc import Coroutine
from datetime import timedelta
from pathlib import Path
from time import perf_counter
from typing import Any
from uuid import UUID, uuid4

import psycopg
from alembic import command
from alembic.config import Config
from psycopg import sql
from psycopg.conninfo import make_conninfo
from pydantic import SecretStr
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine
from sqlmodel import select

from app.core.config import settings
from app.infrastructure.background_tasks import AsyncioBackgroundTaskSubmitter
from app.infrastructure.cache import InMemoryCache
from app.infrastructure.database import build_orm_database_url
from app.infrastructure.lifespan import _create_chat_title_generator
from app.models import DEFAULT_CHAT_SESSION_TITLE, ChatSession, User, utc_now
from app.repositories import ChatSessionRepository
from app.schemas.llm import ModelSpec
from app.services.chat_session_cache import build_chat_session_list_cache_key
from app.services.chat_title import (
    BackgroundChatTitleWriter,
    ChatTitleGenerator,
    LLMChatTitleGenerator,
)
from app.services.llm.factory import create_openai_chat_model
from app.services.llm.registry import LLMRegistry
from app.services.llm.service import LLMService

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONNECTION_TIMEOUT_SECONDS = 10
TITLE_TASK_TIMEOUT_SECONDS = 120.0
CLAIM_LEASE_SECONDS = 300.0
FAILURE_TOTAL_TIMEOUT_SECONDS = 5.0
FAILURE_TRANSPORT_TIMEOUT_SECONDS = 0.001
TEMPORARY_DATABASE_PREFIX = "deep_research_chat_title_"

USER_MESSAGE = "请详细讲解 PostgreSQL 原子更新如何避免并发重复执行。"
ASSISTANT_MESSAGE = "原子条件更新会把资格检查和状态修改放在同一条 SQL 中。"


class _CountingRealTitleGenerator:
    """记录调用次数，同时始终委托 production 的真实标题生成器."""

    def __init__(self, delegate: ChatTitleGenerator) -> None:
        """保存真实 delegate；本类不构造预设回复，也不替换 provider."""
        self._delegate = delegate
        self.calls = 0

    async def generate(
        self,
        *,
        user_message: str,
        assistant_message: str,
    ) -> str:
        """计数后执行真实 structured-output 请求."""
        self.calls += 1
        return await self._delegate.generate(
            user_message=user_message,
            assistant_message=assistant_message,
        )


class _ObservingFailingRealTitleGenerator:
    """观察一次真实失败请求，不构造预设结果或伪造 provider 异常."""

    def __init__(self, delegate: ChatTitleGenerator) -> None:
        """保存真实 delegate，并初始化不含异常正文的观察字段.

        Args:
            delegate: 使用真实 ChatOpenAI transport 的标题生成器。当前 wrapper 只
                记录调用次数和最终异常类型，然后原样重新抛出。
        """
        self._delegate = delegate
        self.calls = 0
        self.error_type: str | None = None

    async def generate(
        self,
        *,
        user_message: str,
        assistant_message: str,
    ) -> str:
        """执行真实请求，记录安全错误类型并保持原失败语义."""
        self.calls += 1
        try:
            return await self._delegate.generate(
                user_message=user_message,
                assistant_message=assistant_message,
            )
        except Exception as error:
            self.error_type = type(error).__name__
            raise


def _create_failing_real_generator() -> _ObservingFailingRealTitleGenerator:
    """创建 transport timeout 极短、但仍访问真实 provider 的标题生成器.

    Returns:
        记录调用和安全异常类型的 wrapper。请求不会使用 fake LLM；1 毫秒预算只是
        让真实网络稳定进入失败路径，并避免故障验收长时间占用模型配额。
    """
    spec = ModelSpec(
        alias="title-timeout",
        provider_model=settings.DEFAULT_LLM_MODEL,
        api_key=SecretStr(settings.OPENAI_API_KEY),
        base_url=settings.OPENAI_BASE_URL,
        temperature=0.0,
        max_tokens=32,
        request_timeout_seconds=FAILURE_TRANSPORT_TIMEOUT_SECONDS,
    )
    registry = LLMRegistry((spec,), create_openai_chat_model)
    service = LLMService(
        registry,
        max_attempts=1,
        retry_wait_multiplier=0,
        total_timeout_seconds=FAILURE_TOTAL_TIMEOUT_SECONDS,
    )
    return _ObservingFailingRealTitleGenerator(LLMChatTitleGenerator(service, aliases=(spec.alias,)))


def _elapsed_ms(started_at: float) -> float:
    """返回保留两位小数的执行耗时."""
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
    """生成仅保存在当前进程环境变量中的 Alembic URL."""
    return build_orm_database_url(settings).set(database=database).render_as_string(hide_password=False)


def _create_database(*, admin_database: str, test_database: str) -> None:
    """通过安全标识符创建本次 smoke 独享的随机数据库."""
    if not test_database.startswith(TEMPORARY_DATABASE_PREFIX):
        raise ValueError("refusing to create a database without the smoke prefix")
    with psycopg.connect(_conninfo(admin_database), autocommit=True) as connection:
        connection.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(test_database)))


def _drop_database(*, admin_database: str, test_database: str) -> None:
    """终止临时连接并只删除本次随机数据库."""
    if not test_database.startswith(TEMPORARY_DATABASE_PREFIX):
        raise ValueError("refusing to drop a database without the smoke prefix")
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
    """为临时数据库创建短生命周期 ORM Engine 和 Session 工厂."""
    engine = create_async_engine(
        build_orm_database_url(settings).set(database=database),
        pool_size=4,
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


async def _seed_sessions(
    session_factory: async_sessionmaker[AsyncSession],
) -> tuple[UUID, UUID, UUID, UUID, UUID]:
    """创建一个用户以及四种标题处理场景的独立会话."""
    async with session_factory() as session:
        async with session.begin():
            user = User(
                email=f"chat-title-{uuid4().hex[:8]}@example.com",
                password_hash="smoke-only-password-hash",
            )
            session.add(user)
            await session.flush()

            concurrent_session = ChatSession(user_id=user.id)
            manual_session = ChatSession(user_id=user.id, title="Manual title")
            stale_session = ChatSession(user_id=user.id)
            failure_session = ChatSession(user_id=user.id)
            session.add_all(
                (
                    concurrent_session,
                    manual_session,
                    stale_session,
                    failure_session,
                )
            )
            await session.flush()

    return (
        user.id,
        concurrent_session.id,
        manual_session.id,
        stale_session.id,
        failure_session.id,
    )


def _create_writer(
    *,
    generator: ChatTitleGenerator,
    session_factory: async_sessionmaker[AsyncSession],
    cache: InMemoryCache,
    submitter: AsyncioBackgroundTaskSubmitter,
) -> BackgroundChatTitleWriter:
    """使用与 production 相同的 writer 组合一份模拟 worker runtime."""
    return BackgroundChatTitleWriter(
        generator=generator,
        session_factory=session_factory,
        cache=cache,
        task_submitter=submitter,
        claim_lease_seconds=CLAIM_LEASE_SECONDS,
    )


async def _read_session(
    session_factory: async_sessionmaker[AsyncSession],
    session_id: UUID,
) -> ChatSession:
    """从权威数据库重新读取一条会话，不复用 identity map."""
    async with session_factory() as session:
        result = await session.execute(select(ChatSession).where(ChatSession.id == session_id))
        chat_session = result.scalar_one_or_none()
    if chat_session is None:
        raise RuntimeError("smoke chat session disappeared")
    return chat_session


async def _exercise_title_generation(database: str) -> dict[str, object]:
    """执行并发 claim、人工保护、过期恢复和资源收敛验收."""
    started_at = perf_counter()
    engine, session_factory = _create_orm_runtime(database)
    cache = InMemoryCache()

    try:
        (
            user_id,
            concurrent_id,
            manual_id,
            stale_id,
            failure_id,
        ) = await _seed_sessions(session_factory)
        counting_generator = _CountingRealTitleGenerator(_create_chat_title_generator(settings))

        # 先放入会话列表缓存。标题事务提交后必须主动删除，不能等 TTL 碰巧过期。
        cache_key = build_chat_session_list_cache_key(user_id)
        await cache.set(cache_key, "cached-list", ttl_seconds=60)

        # 两套 submitter/writer 代表两个互不共享 asyncio task set 的 worker。
        # 它们唯一共享的是 PostgreSQL，因此只有数据库 claim 能提供单赢家语义。
        first_submitter = AsyncioBackgroundTaskSubmitter()
        second_submitter = AsyncioBackgroundTaskSubmitter()
        first_writer = _create_writer(
            generator=counting_generator,
            session_factory=session_factory,
            cache=cache,
            submitter=first_submitter,
        )
        second_writer = _create_writer(
            generator=counting_generator,
            session_factory=session_factory,
            cache=cache,
            submitter=second_submitter,
        )
        for writer in (first_writer, second_writer):
            writer.submit_turn(
                user_id=user_id,
                source_thread_id=concurrent_id,
                user_message=USER_MESSAGE,
                assistant_message=ASSISTANT_MESSAGE,
            )

        await asyncio.gather(
            first_submitter.shutdown(timeout_seconds=TITLE_TASK_TIMEOUT_SECONDS),
            second_submitter.shutdown(timeout_seconds=TITLE_TASK_TIMEOUT_SECONDS),
        )
        concurrent_row = await _read_session(session_factory, concurrent_id)
        concurrent_single_model_call = counting_generator.calls == 1
        concurrent_title_completed = (
            concurrent_row.title != DEFAULT_CHAT_SESSION_TITLE
            and concurrent_row.title_generated_at is not None
            and concurrent_row.title_claim_token is None
            and concurrent_row.title_claimed_at is None
        )
        title_cache_invalidated = await cache.get(cache_key) is None

        # 自定义标题不满足 claim 的 WHERE 条件，因此连真实模型都不应调用。
        manual_submitter = AsyncioBackgroundTaskSubmitter()
        manual_writer = _create_writer(
            generator=counting_generator,
            session_factory=session_factory,
            cache=cache,
            submitter=manual_submitter,
        )
        manual_writer.submit_turn(
            user_id=user_id,
            source_thread_id=manual_id,
            user_message=USER_MESSAGE,
            assistant_message=ASSISTANT_MESSAGE,
        )
        await manual_submitter.shutdown(timeout_seconds=TITLE_TASK_TIMEOUT_SECONDS)
        manual_row = await _read_session(session_factory, manual_id)
        manual_title_preserved = manual_row.title == "Manual title" and counting_generator.calls == 1

        # 先写入一个已经过期的旧 token，模拟 worker 在 claim 后崩溃。新 writer
        # 应通过 claimed_at 判断可接管，并使用新 token 防止旧 worker 迟到提交。
        old_claimed_at = utc_now() - timedelta(seconds=CLAIM_LEASE_SECONDS * 2)
        async with session_factory() as session:
            async with session.begin():
                old_claimed = await ChatSessionRepository(session).claim_title_generation(
                    stale_id,
                    user_id=user_id,
                    claim_token=uuid4(),
                    claimed_at=old_claimed_at,
                    stale_before=old_claimed_at - timedelta(seconds=1),
                )

        recovery_submitter = AsyncioBackgroundTaskSubmitter()
        recovery_writer = _create_writer(
            generator=counting_generator,
            session_factory=session_factory,
            cache=cache,
            submitter=recovery_submitter,
        )
        recovery_writer.submit_turn(
            user_id=user_id,
            source_thread_id=stale_id,
            user_message=USER_MESSAGE,
            assistant_message=ASSISTANT_MESSAGE,
        )
        await recovery_submitter.shutdown(timeout_seconds=TITLE_TASK_TIMEOUT_SECONDS)
        recovered_row = await _read_session(session_factory, stale_id)
        stale_claim_recovered = (
            old_claimed
            and counting_generator.calls == 2
            and recovered_row.title != DEFAULT_CHAT_SESSION_TITLE
            and recovered_row.title_generated_at is not None
            and recovered_row.title_claim_token is None
        )

        # 最后验证真正的失败路径：writer 先成功 claim，然后通过真实 ChatOpenAI
        # transport 发起请求。1 毫秒 timeout 会让 LLMService 返回安全聚合异常；
        # BackgroundChatTitleWriter 捕获后必须释放当前 token。若 token 残留，后续
        # 正常轮次只能等待整段租约过期，用户会长时间看不到自动标题。
        failing_generator = _create_failing_real_generator()
        failure_submitter = AsyncioBackgroundTaskSubmitter()
        failure_writer = _create_writer(
            generator=failing_generator,
            session_factory=session_factory,
            cache=cache,
            submitter=failure_submitter,
        )
        failure_writer.submit_turn(
            user_id=user_id,
            source_thread_id=failure_id,
            user_message=USER_MESSAGE,
            assistant_message=ASSISTANT_MESSAGE,
        )
        await failure_submitter.shutdown(timeout_seconds=TITLE_TASK_TIMEOUT_SECONDS)
        failed_row = await _read_session(session_factory, failure_id)
        provider_failure_released_claim = (
            failing_generator.calls == 1
            and failing_generator.error_type is not None
            and failed_row.title == DEFAULT_CHAT_SESSION_TITLE
            and failed_row.title_generated_at is None
            and failed_row.title_claim_token is None
            and failed_row.title_claimed_at is None
        )

        all_submitters_idle = all(
            submitter.active_count == 0 and not submitter.accepting
            for submitter in (
                first_submitter,
                second_submitter,
                manual_submitter,
                recovery_submitter,
                failure_submitter,
            )
        )
        checks = (
            concurrent_single_model_call,
            concurrent_title_completed,
            title_cache_invalidated,
            manual_title_preserved,
            stale_claim_recovered,
            provider_failure_released_claim,
            all_submitters_idle,
        )
        return {
            "ok": all(checks),
            "model": settings.DEFAULT_LLM_MODEL,
            "concurrent_single_model_call": concurrent_single_model_call,
            "concurrent_title_completed": concurrent_title_completed,
            "title_cache_invalidated": title_cache_invalidated,
            "manual_title_preserved": manual_title_preserved,
            "stale_claim_recovered": stale_claim_recovered,
            "provider_failure_released_claim": provider_failure_released_claim,
            "provider_failure_type": failing_generator.error_type,
            "failed_real_model_call_count": failing_generator.calls,
            "real_model_call_count": counting_generator.calls,
            "all_submitters_idle": all_submitters_idle,
            "elapsed_ms": _elapsed_ms(started_at),
        }
    finally:
        await engine.dispose()


def _run_async(
    coroutine: Coroutine[Any, Any, dict[str, object]],
) -> dict[str, object]:
    """运行异步验收，并兼容 Windows psycopg 的 Selector event loop."""
    if sys.platform == "win32":
        with asyncio.Runner(loop_factory=lambda: asyncio.SelectorEventLoop(selectors.SelectSelector())) as runner:
            return runner.run(coroutine)
    return asyncio.run(coroutine)


def main() -> int:
    """编排临时数据库、migration、真实标题请求和异常安全清理."""
    started_at = perf_counter()
    admin_database = settings.POSTGRES_DB
    test_database = f"{TEMPORARY_DATABASE_PREFIX}{uuid4().hex[:10]}"
    previous_override = os.environ.get("ALEMBIC_DATABASE_URL")
    database_created = False
    cleanup_ok = False

    try:
        _create_database(admin_database=admin_database, test_database=test_database)
        database_created = True
        os.environ["ALEMBIC_DATABASE_URL"] = _temporary_database_url(test_database)
        command.upgrade(Config(str(PROJECT_ROOT / "alembic.ini")), "head")
        summary = _run_async(_exercise_title_generation(test_database))
    except Exception as error:
        summary = {
            "ok": False,
            "error_type": type(error).__name__,
            "elapsed_ms": _elapsed_ms(started_at),
        }
    finally:
        if previous_override is None:
            os.environ.pop("ALEMBIC_DATABASE_URL", None)
        else:
            os.environ["ALEMBIC_DATABASE_URL"] = previous_override

        if database_created:
            try:
                _drop_database(admin_database=admin_database, test_database=test_database)
            except Exception:
                cleanup_ok = False
            else:
                cleanup_ok = True

    summary["database_cleanup_ok"] = cleanup_ok
    summary["total_elapsed_ms"] = _elapsed_ms(started_at)
    summary["ok"] = bool(summary["ok"] and cleanup_ok)
    print(json.dumps(summary, ensure_ascii=False))
    return 0 if summary["ok"] is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
