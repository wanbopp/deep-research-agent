"""使用真实 provider 和基础设施验收 Chat Agent 的长期记忆闭环.

本脚本验证的不是同一 thread 的 checkpoint 历史，而是以下跨 thread 调用链：

1. 真实用户在来源会话中明确表达一条稳定学习偏好；
2. ChatService 返回最终回复后，后台 writer 调用真实结构化模型提取候选；
3. MemoryService 使用真实 Embedding 并把候选写入随机 PostgreSQL 数据库；
4. 同一用户的新会话通过 memory node 检索该记忆，真实模型据此回答；
5. 检索注入只存在于本次模型视图，不会被复制进 LangGraph checkpoint。

最终摘要只输出布尔证据、计数、模型名和耗时，不输出用户身份、Prompt、模型
回复、记忆正文、向量、缓存 key、JWT、连接串或凭据。
"""

import asyncio
import json
import logging
import os
import secrets
import selectors
from collections.abc import Callable
from pathlib import Path
from time import perf_counter
from unittest.mock import patch
from uuid import UUID, uuid4

# 必须在导入 app 模块前降低日志级别。真实 SDK 异常可能附带请求诊断；smoke 的
# 最外层只需要输出异常类型，不能把 Prompt、回复或凭据写入控制台。
os.environ["DEBUG"] = "false"
os.environ["LOG_LEVEL"] = "WARNING"

import psycopg  # noqa: E402
from alembic import command  # noqa: E402
from alembic.config import Config  # noqa: E402
from asgi_correlation_id import CorrelationIdMiddleware  # noqa: E402
from fastapi import FastAPI, status  # noqa: E402
from httpx import ASGITransport, AsyncClient, Response  # noqa: E402
from langchain_core.messages import AIMessage, HumanMessage  # noqa: E402
from langgraph.checkpoint.base import BaseCheckpointSaver  # noqa: E402
from psycopg import sql  # noqa: E402
from psycopg.conninfo import make_conninfo  # noqa: E402
from pydantic import SecretStr  # noqa: E402
from redis.asyncio import Redis  # noqa: E402
from sqlmodel import select  # noqa: E402

from app.agents.chat.graph import ChatGraph  # noqa: E402
from app.agents.chat.runtime import create_chat_runtime  # noqa: E402
from app.api.dependencies import (  # noqa: E402
    enforce_agent_rate_limit,
    enforce_auth_rate_limit,
    get_token_service,
)
from app.api.v1.auth import router as auth_router  # noqa: E402
from app.api.v1.chat import router as chat_router  # noqa: E402
from app.api.v1.chat_sessions import router as chat_sessions_router  # noqa: E402
from app.core.config import Settings, settings  # noqa: E402
from app.core.exception_handlers import register_exception_handlers  # noqa: E402
import app.infrastructure.background_tasks as background_tasks_module  # noqa: E402
from app.infrastructure.cache import RedisCache  # noqa: E402
from app.infrastructure.database import build_orm_database_url  # noqa: E402
import app.infrastructure.lifespan as lifespan_module  # noqa: E402
from app.infrastructure.lifespan import (  # noqa: E402
    get_application_background_task_submitter,
    get_application_resources,
    lifespan,
)
from app.models import DEFAULT_CHAT_SESSION_TITLE, ChatSession, Memory  # noqa: E402
from app.schemas.memory import (  # noqa: E402
    MemoryCreate,
    MemoryItem,
    MemoryQuery,
    MemorySearchResult,
)
from app.services.auth import TokenService  # noqa: E402
from app.services.chat import ChatService  # noqa: E402
from app.services.memory import MemoryUnavailableError  # noqa: E402
import app.services.memory_service as memory_service_module  # noqa: E402
from app.services.memory_service import MemoryService  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONNECTION_TIMEOUT_SECONDS = 10
TOTAL_TIMEOUT_SECONDS = 300.0
BACKGROUND_TASK_TIMEOUT_SECONDS = 120.0
HTTP_TIMEOUT_SECONDS = 120.0
TEMPORARY_DATABASE_PREFIX = "deep_research_chat_memory_"

# lifespan 会把 RedisCache 用于业务会话缓存和长期记忆缓存。smoke 不能扫描或清空
# 共享 Redis，因此用这个列表精确记录并删除本次运行触碰过的哈希 key。
_TRACKING_CACHES: list["_TrackingRedisCache"] = []


class _MemoryLogHandler(logging.Handler):
    """在内存中捕获结构化 LogRecord，用于事件和泄漏检查."""

    def __init__(self) -> None:
        """创建不会向控制台或文件额外写入内容的观察 handler."""
        super().__init__(level=logging.NOTSET)
        self.records: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        """保存防御性记录表示，不主动格式化可能含异常正文的消息.

        Args:
            record: 应用 logging/structlog 最终交给标准库的日志记录。

        Notes:
            handler 只用于当前进程内断言。若记录无法安全表示，则写入固定标记，
            让最终泄漏检查 fail-closed，而不是让日志观察器破坏业务请求。
        """
        try:
            self.records.append(repr(record.__dict__))
        except Exception:
            self.records.append("<unrepresentable-log-record>")


class _ControlledUnavailableMemoryStore:
    """在 Agent 层故障验收中提供可暂停的稳定 Store 失败.

    这个类不模拟 LLM。memory node 和后台 writer 仍分别调用真实 Chat 模型与真实
    structured extraction；只有 MemoryStore 协议边界被故意置为不可用，以验证
    上层的降级和异常隔离，而不依赖偶发的真实数据库中断。
    """

    def __init__(self) -> None:
        """创建调用计数和控制后台写入失败时机的两个事件."""
        self.search_calls = 0
        self.add_calls = 0
        self.add_started = asyncio.Event()
        self.release_add_failure = asyncio.Event()

    async def search(
        self,
        *,
        user_id: UUID,
        query: MemoryQuery,
    ) -> tuple[MemoryItem, ...]:
        """把一次真实 Agent 记忆查询映射为稳定的 Store 不可用错误."""
        _ = (user_id, query)
        self.search_calls += 1
        raise MemoryUnavailableError()

    async def add(
        self,
        *,
        user_id: UUID,
        memory: MemoryCreate,
    ) -> MemoryItem:
        """暂停后台写入，待 HTTP 完成证据取得后再抛稳定错误.

        Args:
            user_id: ChatService 后台 writer 绑定的可信用户身份。
            memory: 真实 structured extraction 生成并由服务端补齐来源的候选。

        Raises:
            MemoryUnavailableError: smoke 放行后始终抛出，模拟写入后端不可用。
        """
        _ = (user_id, memory)
        self.add_calls += 1
        self.add_started.set()
        await self.release_add_failure.wait()
        raise MemoryUnavailableError()

    async def delete(self, *, user_id: UUID, memory_id: UUID) -> None:
        """保持 MemoryStore 协议完整；本 smoke 不应调用删除."""
        _ = (user_id, memory_id)
        raise MemoryUnavailableError()


class _TrackingRedisCache(RedisCache):
    """委托真实 RedisCache，同时记录本次 smoke 自己访问过的 key."""

    def __init__(self, redis_client: Redis) -> None:
        """构造真实 adapter，并把当前实例登记到脚本级清理集合.

        Args:
            redis_client: lifespan 所有的真实异步 Redis client；本类只借用连接。
        """
        super().__init__(redis_client)
        self.keys: set[str] = set()
        _TRACKING_CACHES.append(self)

    async def get(self, key: str) -> str | None:
        """记录 key 后执行真实 Redis GET."""
        self.keys.add(key)
        return await super().get(key)

    async def set(self, key: str, value: str, *, ttl_seconds: int) -> None:
        """记录 key 后执行带 TTL 的真实 Redis SET."""
        self.keys.add(key)
        await super().set(key, value, ttl_seconds=ttl_seconds)

    async def delete(self, key: str) -> None:
        """记录 key 后执行真实幂等删除."""
        self.keys.add(key)
        await super().delete(key)

    async def cleanup(self) -> bool:
        """只删除当前 adapter 记录过的 key，并验证它们已经不存在.

        Returns:
            所有本次 smoke key 均已从真实 Redis 删除时返回 ``True``。
        """
        keys = tuple(self.keys)
        for key in keys:
            await super().delete(key)
        remaining_checks = [not bool(await self._redis_client.exists(key)) for key in keys]
        return all(remaining_checks)


def _elapsed_ms(started_at: float) -> float:
    """返回保留两位小数的单调时钟毫秒数."""
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
    """创建带固定安全前缀的随机隔离数据库.

    Args:
        admin_database: 已存在并允许创建数据库的管理库。
        test_database: 本次 smoke 独占的随机数据库名。

    Raises:
        ValueError: 数据库名不带固定前缀时拒绝执行。
    """
    if not test_database.startswith(TEMPORARY_DATABASE_PREFIX):
        raise ValueError("refusing to create a database without the smoke prefix")
    with psycopg.connect(_conninfo(admin_database), autocommit=True) as connection:
        connection.execute(
            sql.SQL("CREATE DATABASE {}").format(sql.Identifier(test_database)),
        )


def _drop_database(admin_database: str, test_database: str) -> None:
    """终止本次临时库的残留连接，然后只删除该随机数据库."""
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
        connection.execute(
            sql.SQL("DROP DATABASE IF EXISTS {}").format(
                sql.Identifier(test_database),
            )
        )


def _runtime_settings(database: str) -> Settings:
    """保留真实 provider/Redis 配置，只替换随机数据库和连接池大小."""
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


def _is_completed_chat(response: Response) -> bool:
    """检查公开 Chat 响应已经完成，不读取或输出正文."""
    if response.status_code != status.HTTP_200_OK:
        return False
    body = response.json()
    message = body.get("message")
    return body.get("status") == "completed" and isinstance(message, dict)


def _chat_content(response: Response) -> str:
    """从已完成响应读取模型正文，仅供进程内断言使用.

    Raises:
        RuntimeError: 响应不是稳定的 completed Chat 形状。
    """
    if not _is_completed_chat(response):
        raise RuntimeError("chat response must be completed")
    content = response.json()["message"].get("content")
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("completed chat response must contain text")
    return content


async def _register_user(
    client: AsyncClient,
    *,
    email: str,
    password: str,
) -> str:
    """通过正式注册 route 创建真实用户并返回进程内 token."""
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
    """通过正式 ChatSession route 创建属于当前用户的业务会话."""
    response = await client.post(
        "/api/v1/chat/sessions",
        headers=headers,
        json={"title": title},
    )
    response.raise_for_status()
    return UUID(str(response.json()["thread_id"]))


async def _wait_for_background_idle(active_count: Callable[[], int]) -> bool:
    """等待本 worker 的受管后台任务集合归零.

    Args:
        active_count: 每次调用返回当前受管任务数量的无参数函数。

    Returns:
        在预算内归零时返回 ``True``，否则返回 ``False``。

    Notes:
        这里等待是 smoke 的验收需要；正式 HTTP 路径仍然在提交任务后立即返回。
        如果 smoke 不等待，就可能在提取或 INSERT 尚未结束时读取数据库并误报失败。
    """
    loop = asyncio.get_running_loop()
    deadline = loop.time() + BACKGROUND_TASK_TIMEOUT_SECONDS
    while active_count() > 0:
        if loop.time() >= deadline:
            return False
        await asyncio.sleep(0.05)
    return True


async def _wait_for_event(event: asyncio.Event) -> bool:
    """在后台任务预算内等待一个确定性阶段信号.

    Args:
        event: 由受控 Store 在进入 ``add()`` 后设置的 asyncio 事件。

    Returns:
        事件在预算内出现时返回 ``True``，超时时返回 ``False``。
    """
    try:
        async with asyncio.timeout(BACKGROUND_TASK_TIMEOUT_SECONDS):
            await event.wait()
    except TimeoutError:
        return False
    return True


async def _exercise_smoke(database: str) -> dict[str, bool | int | float | str]:
    """运行真实长期记忆闭环并返回脱敏检查结果.

    Args:
        database: 已执行正式 Alembic migration 的随机数据库名。

    Returns:
        只含布尔值、计数、模型名与耗时的安全摘要。

    Raises:
        Exception: 基础设施、provider、协议或持久化失败会交给最外层安全处理。
    """
    started_at = perf_counter()
    _TRACKING_CACHES.clear()

    # 随机 JWT secret 只存在于当前进程，避免 smoke 签发的 token 与开发服务互信。
    token_service = TokenService(
        secret_key=SecretStr(secrets.token_urlsafe(48)),
    )
    captured_graphs: list[ChatGraph] = []
    captured_memory_services: list[MemoryService] = []

    def capture_runtime(
        *,
        checkpointer: BaseCheckpointSaver[str] | None = None,
        memory_service: MemoryService | None = None,
    ) -> ChatGraph:
        """捕获 production 依赖身份，同时仍构造真实 Agent Graph.

        Args:
            checkpointer: lifespan 已 setup 的 PostgreSQL saver。
            memory_service: lifespan 创建的真实 MemoryService。

        Returns:
            由正式 runtime factory 编译的完整 ChatGraph。
        """
        if memory_service is None:
            raise RuntimeError("memory smoke requires production MemoryService")
        graph = create_chat_runtime(
            checkpointer=checkpointer,
            memory_service=memory_service,
        )
        captured_graphs.append(graph)
        captured_memory_services.append(memory_service)
        return graph

    async def bypass_rate_limit() -> None:
        """隔离本 smoke 的成本配额；限流真实语义已由 Lab 15 独立验收."""

    app = FastAPI()
    register_exception_handlers(app)
    app.add_middleware(CorrelationIdMiddleware)
    app.include_router(auth_router, prefix="/api/v1")
    app.include_router(chat_router, prefix="/api/v1")
    app.include_router(chat_sessions_router, prefix="/api/v1")
    app.dependency_overrides[get_token_service] = lambda: token_service
    app.dependency_overrides[enforce_auth_rate_limit] = bypass_rate_limit
    app.dependency_overrides[enforce_agent_rate_limit] = bypass_rate_limit

    async with asyncio.timeout(TOTAL_TIMEOUT_SECONDS):
        # 两个 patch 都只做观察/清理：Graph、LLM、Embedding、Redis 和 PostgreSQL
        # 调用仍由 production 实现完成，没有 fake model 或 fake store。
        with (
            patch.object(lifespan_module, "create_chat_runtime", capture_runtime),
            patch.object(lifespan_module, "RedisCache", _TrackingRedisCache),
        ):
            async with lifespan(app, config=_runtime_settings(database)):
                resources = get_application_resources(app)
                submitter = get_application_background_task_submitter(app)
                if len(captured_graphs) != 1 or len(captured_memory_services) != 1:
                    raise RuntimeError("lifespan must create exactly one memory-enabled Graph")
                graph = captured_graphs[0]
                memory_service = captured_memory_services[0]

                transport = ASGITransport(app=app, raise_app_exceptions=False)
                log_handler = _MemoryLogHandler()
                root_logger = logging.getLogger()
                root_logger.addHandler(log_handler)
                try:
                    async with AsyncClient(
                        transport=transport,
                        base_url="http://testserver",
                        timeout=HTTP_TIMEOUT_SECONDS,
                    ) as client:
                        suffix = uuid4().hex
                        email = f"memory-smoke-{suffix}@example.com"
                        password = f"Memory-Smoke-{secrets.token_urlsafe(18)}!"

                        token = await _register_user(
                            client,
                            email=email,
                            password=password,
                        )
                        headers = _authorization_header(token)
                        user_id = token_service.decode_access_token(token).sub
                        source_thread_id = await _create_session(
                            client,
                            headers=headers,
                            # 默认标题让本次真实 HTTP Chat 同时经过 12E 自动命名。
                            # target/degraded 会话仍使用人工标题，验证它们不会额外调用
                            # 标题模型，也保持降级场景的任务数量断言稳定。
                            title=DEFAULT_CHAT_SESSION_TITLE,
                        )
                        target_thread_id = await _create_session(
                            client,
                            headers=headers,
                            title="Long-term memory target",
                        )

                        source_prompt = (
                            "这是需要长期保存的稳定学习偏好，不是验证码、密码或"
                            "临时任务：技术教学回答必须使用中文，并且先解释 Agent "
                            "状态变化。不要调用工具，"
                            "只需确认已经记录。"
                        )

                        source_response = await client.post(
                            "/api/v1/chat",
                            headers=headers,
                            json={
                                "thread_id": str(source_thread_id),
                                "message": source_prompt,
                            },
                        )
                        source_completed = _is_completed_chat(source_response)
                        source_tasks_drained = await _wait_for_background_idle(
                            lambda: submitter.active_count,
                        )

                        # 后台任务归零只代表“任务已结束”，并不代表写入成功。因此还要
                        # 从随机数据库读取真实行，验证 owner、来源会话和候选正文。
                        async with resources.orm_session_factory() as session:
                            result = await session.execute(select(Memory).where(Memory.user_id == user_id))
                            rows_after_source = tuple(result.scalars().all())
                            title_result = await session.execute(
                                select(ChatSession).where(ChatSession.id == source_thread_id)
                            )
                            source_session = title_result.scalar_one_or_none()

                        # 这条证据覆盖 production 接线，而不仅是 title writer 单独可用：
                        # HTTP Chat 完成 -> ChatService 完成边界 -> 共享 submitter ->
                        # PostgreSQL claim -> 真实 structured title -> 条件提交。
                        source_title_generated = (
                            source_session is not None
                            and source_session.title != DEFAULT_CHAT_SESSION_TITLE
                            and source_session.title_generated_at is not None
                            and source_session.title_claim_token is None
                            and source_session.title_claimed_at is None
                        )

                        matching_rows = tuple(row for row in rows_after_source if "中文" in row.content)
                        memory_persisted = len(matching_rows) == 1
                        memory_owner_and_source_match = (
                            memory_persisted
                            and matching_rows[0].user_id == user_id
                            and matching_rows[0].source_thread_id == source_thread_id
                        )

                        target_prompt = (
                            "根据我已经保存的长期偏好，技术教学回答应该使用什么语言？不要调用工具，只回复语言名称。"
                        )
                        target_search_results: list[MemorySearchResult] = []
                        original_memory_search = memory_service.search

                        async def observe_target_memory_search(
                            *,
                            user_id: UUID,
                            query: MemoryQuery,
                        ) -> MemorySearchResult:
                            """委托真实搜索，并记录 target invocation 的检索结果."""
                            result = await original_memory_search(
                                user_id=user_id,
                                query=query,
                            )
                            target_search_results.append(result)
                            return result

                        with patch.object(
                            memory_service,
                            "search",
                            observe_target_memory_search,
                        ):
                            target_response = await client.post(
                                "/api/v1/chat",
                                headers=headers,
                                json={
                                    "thread_id": str(target_thread_id),
                                    "message": target_prompt,
                                },
                            )
                        target_completed = _is_completed_chat(target_response)
                        target_content = _chat_content(target_response)
                        target_search_retrieved_source = (
                            len(target_search_results) == 1
                            and not target_search_results[0].is_degraded
                            and any(
                                item.source_thread_id == source_thread_id for item in target_search_results[0].items
                            )
                        )
                        target_answer_reflects_memory = (
                            "中文" in target_content or "chinese" in target_content.casefold()
                        )
                        target_used_cross_thread_memory = (
                            target_search_retrieved_source and target_answer_reflects_memory
                        )

                        # target 回答也会经过同一 completed 提交边界。即使提取器判断
                        # 该问答没有新候选，任务仍必须被观察并在 shutdown 前结束。
                        target_tasks_drained = await _wait_for_background_idle(
                            lambda: submitter.active_count,
                        )

                        target_snapshot = await graph.aget_state(
                            ChatService._build_config(
                                user_id=user_id,
                                public_thread_id=target_thread_id,
                            )
                        )
                        snapshot_values = dict(target_snapshot.values)
                        snapshot_messages = snapshot_values.get("messages")
                        if not isinstance(snapshot_messages, list):
                            raise RuntimeError("target checkpoint must contain messages")

                        # 目标 checkpoint 应只有“本轮真实 HumanMessage + 最终 AIMessage”。
                        # 如果临时记忆 HumanMessage 被错误写回 state，这里会出现第三条
                        # 消息，并把跨 thread 记忆永久复制到会话历史。
                        checkpoint_keeps_memory_context_temporary = (
                            len(snapshot_messages) == 2
                            and isinstance(snapshot_messages[0], HumanMessage)
                            and isinstance(snapshot_messages[1], AIMessage)
                            and snapshot_messages[0].content == target_prompt
                            and "memory_context" not in snapshot_values
                            and "memory_status" not in snapshot_values
                        )

                        # 正常闭环已完成后才替换 Store 协议边界。Graph 和 writer 持有
                        # 同一个 MemoryService，因此一次替换可以同时观察：
                        # 1. memory node 的 search 故障是否降级；
                        # 2. completed 后后台 add 故障是否被 submitter 消费。
                        controlled_store = _ControlledUnavailableMemoryStore()
                        memory_service._store = controlled_store

                        async with resources.orm_session_factory() as session:
                            baseline_result = await session.execute(select(Memory).where(Memory.user_id == user_id))
                            memory_count_before_failure = len(baseline_result.scalars().all())

                        degraded_thread_id = await _create_session(
                            client,
                            headers=headers,
                            title="Long-term memory degraded boundary",
                        )
                        degraded_marker = f"FAULT-{suffix[10:20].upper()}"
                        degraded_prompt = (
                            "这是稳定的长期学习偏好，不是凭据或临时任务："
                            f"我的故障演练教学风格名称是 {degraded_marker}，回答应使用"
                            "中文。即使长期记忆暂时不可用，也不要调用工具并正常确认。"
                        )
                        # spy 保留真实日志调用，只记录固定事件名和结构化字段是否出现。
                        # 它比读取已渲染日志文本稳定，也不会把异常或正文替换成 fake。
                        with (
                            patch.object(
                                memory_service_module.logger,
                                "warning",
                                wraps=memory_service_module.logger.warning,
                            ) as warning_spy,
                            patch.object(
                                background_tasks_module.logger,
                                "error",
                                wraps=background_tasks_module.logger.error,
                            ) as error_spy,
                        ):
                            degraded_response = await client.post(
                                "/api/v1/chat",
                                headers=headers,
                                json={
                                    "thread_id": str(degraded_thread_id),
                                    "message": degraded_prompt,
                                },
                            )
                            degraded_chat_completed = _is_completed_chat(degraded_response)

                            # HTTP 已经返回后，真实 structured extraction 才会推进到
                            # 受控 Store.add。任务仍活跃证明主响应没有等待写入结果。
                            background_add_started = await _wait_for_event(controlled_store.add_started)
                            response_returned_before_background_failure = (
                                degraded_chat_completed and background_add_started and submitter.active_count == 1
                            )

                            controlled_store.release_add_failure.set()
                            degraded_tasks_drained = await _wait_for_background_idle(
                                lambda: submitter.active_count,
                            )

                            degraded_search_logged = any(
                                call.args
                                and call.args[0] == "memory_search_degraded"
                                and call.kwargs.get("error_type") == "MemoryUnavailableError"
                                for call in warning_spy.call_args_list
                            )
                            background_failure_logged = any(
                                call.args
                                and call.args[0] == "background_task_failed"
                                and call.kwargs.get("error_type") == "MemoryUnavailableError"
                                for call in error_spy.call_args_list
                            )

                        async with resources.orm_session_factory() as session:
                            failed_result = await session.execute(select(Memory).where(Memory.user_id == user_id))
                            memory_count_after_failure = len(failed_result.scalars().all())

                        captured_log_text = "\n".join(log_handler.records)
                        failure_logs_hide_sensitive_data = all(
                            marker_text not in captured_log_text
                            for marker_text in (
                                degraded_marker,
                                email,
                                password,
                                token,
                                source_prompt,
                                target_prompt,
                                degraded_prompt,
                            )
                        )
                        degraded_store_calls_match = (
                            controlled_store.search_calls == 1 and controlled_store.add_calls == 1
                        )
                        failed_write_preserved_database = memory_count_after_failure == memory_count_before_failure
                finally:
                    # 正常路径已等待两次任务；异常路径也尽力等待，避免关闭 ORM/Redis
                    # 后还有受管任务继续访问资源。12E 会把这项能力移入 submitter 本身。
                    background_idle_before_shutdown = await _wait_for_background_idle(
                        lambda: submitter.active_count,
                    )
                    cleanup_results = [await cache.cleanup() for cache in _TRACKING_CACHES]
                    cache_cleanup_ok = all(cleanup_results)
                    root_logger.removeHandler(log_handler)

            state_removed_after_shutdown = (
                not hasattr(app.state, "resources")
                and not hasattr(app.state, "chat_service")
                and not hasattr(app.state, "chat_memory_writer")
                and not hasattr(app.state, "chat_title_writer")
                and not hasattr(app.state, "background_task_submitter")
            )
            postgres_pool_closed_after_shutdown = resources.postgres_pool.closed

    elapsed_ms = _elapsed_ms(started_at)
    return {
        "model": settings.DEFAULT_LLM_MODEL,
        "source_completed": source_completed,
        "source_tasks_drained": source_tasks_drained,
        "source_title_generated": source_title_generated,
        "memory_persisted": memory_persisted,
        "memory_owner_and_source_match": memory_owner_and_source_match,
        "memory_count_after_source": len(rows_after_source),
        "target_completed": target_completed,
        "target_search_retrieved_source": target_search_retrieved_source,
        "target_answer_reflects_memory": target_answer_reflects_memory,
        "target_used_cross_thread_memory": target_used_cross_thread_memory,
        "target_tasks_drained": target_tasks_drained,
        "checkpoint_keeps_memory_context_temporary": checkpoint_keeps_memory_context_temporary,
        "degraded_chat_completed": degraded_chat_completed,
        "background_add_started": background_add_started,
        "response_returned_before_background_failure": response_returned_before_background_failure,
        "degraded_tasks_drained": degraded_tasks_drained,
        "degraded_search_logged": degraded_search_logged,
        "background_failure_logged": background_failure_logged,
        "failure_logs_hide_sensitive_data": failure_logs_hide_sensitive_data,
        "degraded_store_calls_match": degraded_store_calls_match,
        "failed_write_preserved_database": failed_write_preserved_database,
        "background_idle_before_shutdown": background_idle_before_shutdown,
        "cache_cleanup_ok": cache_cleanup_ok,
        "state_removed_after_shutdown": state_removed_after_shutdown,
        "postgres_pool_closed_after_shutdown": postgres_pool_closed_after_shutdown,
        "within_total_budget": elapsed_ms <= TOTAL_TIMEOUT_SECONDS * 1000,
        "elapsed_ms": elapsed_ms,
    }


def _run_smoke() -> dict[str, object]:
    """迁移随机数据库、运行真实 smoke，并在 finally 中保证删除."""
    started_at = perf_counter()
    admin_database = settings.POSTGRES_DB
    test_database = f"{TEMPORARY_DATABASE_PREFIX}{uuid4().hex[:10]}"
    previous_override = os.environ.get("ALEMBIC_DATABASE_URL")
    database_created = False
    cleanup_ok = False
    checks: dict[str, bool | int | float | str] = {}

    try:
        _create_database(admin_database, test_database)
        database_created = True
        os.environ["ALEMBIC_DATABASE_URL"] = _temporary_database_url(test_database)
        command.upgrade(Config(str(PROJECT_ROOT / "alembic.ini")), "head")
        checks = asyncio.run(
            _exercise_smoke(test_database),
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
        raise RuntimeError("temporary long-term memory database cleanup failed")

    boolean_checks = tuple(value for value in checks.values() if isinstance(value, bool))
    ok = bool(boolean_checks) and all(boolean_checks) and cleanup_ok
    return {
        "ok": ok,
        **checks,
        "cleanup_ok": cleanup_ok,
        "total_elapsed_ms": _elapsed_ms(started_at),
    }


def main() -> int:
    """输出一条安全 JSON 摘要，并用进程退出码表示 smoke 结果."""
    started_at = perf_counter()
    try:
        summary = _run_smoke()
    except Exception as error:
        # provider/数据库异常文本可能包含连接或请求诊断，只公开异常类型。
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
