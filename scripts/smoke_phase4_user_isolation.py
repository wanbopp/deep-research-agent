"""使用真实服务验证 Phase 4 的双用户纵深隔离边界.

本 smoke 不只检查一个内部 key 是否不同，而是让两个真实注册用户依次经过正式
FastAPI route、JWT、请求级 ORM Session、ChatService、Redis guard、PostgreSQL
checkpointer、MemoryService、标题 claim 和 Redis 缓存。

处理顺序：

1. 创建随机 PostgreSQL 数据库并执行全部正式 Alembic migration；
2. 通过正式注册 API 创建用户 A/B，并分别创建业务会话；
3. 两个用户分别调用真实 Chat 模型，后台任务通过真实模型提取长期记忆；
4. 验证会话列表、checkpoint、记忆、标题 claim 和缓存均按可信 user_id 隔离；
5. 让用户 B 猜测用户 A 的公开会话 UUID，确认在 Graph/模型前安全返回 404；
6. shutdown 后清理本次 Redis key，并删除随机数据库。

最终摘要只输出布尔证据、数量、模型别名和耗时，不输出用户 ID、邮箱、JWT、Prompt、
模型回复、记忆正文、checkpoint 内容、缓存 key、数据库地址或任何凭据。
"""

import asyncio
import json
import os
import secrets
import selectors
from collections.abc import Callable
from datetime import timedelta
from pathlib import Path
from time import perf_counter
from unittest.mock import patch
from uuid import UUID, uuid4

# Settings 在导入 app 模块时初始化。真实 provider 和基础设施配置仍只从 Git 忽略的
# 本地环境文件读取；这里仅降低日志暴露面，并为真实网络调用保留足够总预算。
os.environ["DEBUG"] = "false"
os.environ["LOG_LEVEL"] = "WARNING"
os.environ["LLM_TOTAL_TIMEOUT"] = "180"

import psycopg  # noqa: E402
from alembic import command  # noqa: E402
from alembic.config import Config  # noqa: E402
from asgi_correlation_id import CorrelationIdMiddleware  # noqa: E402
from fastapi import FastAPI, status  # noqa: E402
from httpx import ASGITransport, AsyncClient, Response  # noqa: E402
from langchain_core.messages import BaseMessage  # noqa: E402
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
from app.infrastructure.cache import RedisCache  # noqa: E402
from app.infrastructure.database import build_orm_database_url  # noqa: E402
import app.infrastructure.lifespan as lifespan_module  # noqa: E402
from app.infrastructure.lifespan import (  # noqa: E402
    get_application_background_task_submitter,
    get_application_memory_service,
    get_application_resources,
    lifespan,
)
from app.models import (  # noqa: E402
    DEFAULT_CHAT_SESSION_TITLE,
    ChatSession,
    Memory,
    utc_now,
)
from app.repositories import ChatSessionRepository  # noqa: E402
from app.schemas.memory import MemoryQuery  # noqa: E402
from app.services.auth import TokenService  # noqa: E402
from app.services.chat import ChatService  # noqa: E402
from app.services.chat_session_cache import (  # noqa: E402
    build_chat_session_list_cache_key,
)
from app.services.memory_service import MemoryService  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEMPORARY_DATABASE_PREFIX = "deep_research_phase4_isolation_"
CONNECTION_TIMEOUT_SECONDS = 10
TOTAL_TIMEOUT_SECONDS = 360.0
HTTP_TIMEOUT_SECONDS = 180.0
BACKGROUND_TASK_TIMEOUT_SECONDS = 180.0

# lifespan 会创建多个 RedisCache 实例的使用者，但当前实现共享同一个 adapter。
# 这里替换的子类仍执行真实 Redis 命令，只额外记录本 smoke 触碰过的缓存 key，便于
# finally 精确清理；绝不扫描或 FLUSH 共享 Redis。
_TRACKING_CACHES: list["_TrackingRedisCache"] = []


class _TrackingRedisCache(RedisCache):
    """委托真实 RedisCache，并记录当前 smoke 自己访问过的 key."""

    def __init__(self, redis_client: Redis) -> None:
        """保存真实 Redis client，并登记当前追踪实例.

        Args:
            redis_client: lifespan 创建并拥有的真实异步 Redis client。当前类只借用，
                不负责关闭连接。
        """
        super().__init__(redis_client)
        self.keys: set[str] = set()
        _TRACKING_CACHES.append(self)

    async def get(self, key: str) -> str | None:
        """记录 key 后执行真实 Redis GET."""
        self.keys.add(key)
        return await super().get(key)

    async def set(self, key: str, value: str, *, ttl_seconds: int) -> None:
        """记录 key 后执行真实 Redis SET."""
        self.keys.add(key)
        await super().set(key, value, ttl_seconds=ttl_seconds)

    async def delete(self, key: str) -> None:
        """记录 key 后执行真实 Redis DELETE."""
        self.keys.add(key)
        await super().delete(key)

    async def cleanup(self) -> bool:
        """只删除本实例观察到的 key，并确认它们已经不存在."""
        keys = tuple(self.keys)
        for key in keys:
            await super().delete(key)
        remaining_checks = [not bool(await self._redis_client.exists(key)) for key in keys]
        return all(remaining_checks)


def _elapsed_ms(started_at: float) -> float:
    """返回保留两位小数的单调时钟耗时."""
    return round((perf_counter() - started_at) * 1000, 2)


def _selector_loop_factory() -> asyncio.AbstractEventLoop:
    """创建 Windows psycopg 异步驱动要求的 Selector event loop."""
    return asyncio.SelectorEventLoop(selectors.SelectSelector())


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
    """创建带固定前缀的随机隔离数据库."""
    if not test_database.startswith(TEMPORARY_DATABASE_PREFIX):
        raise ValueError("refusing to create a database without the smoke prefix")
    with psycopg.connect(_conninfo(admin_database), autocommit=True) as connection:
        connection.execute(
            sql.SQL("CREATE DATABASE {}").format(sql.Identifier(test_database)),
        )


def _drop_database(admin_database: str, test_database: str) -> None:
    """终止随机数据库残留连接，并且只删除本次 smoke 数据库."""
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
    """保留真实 provider/Redis 配置，只替换随机数据库和小型连接池."""
    config = Settings()
    config.POSTGRES_DB = database
    config.POSTGRES_PSYCOPG_POOL_SIZE = 4
    config.POSTGRES_ORM_POOL_SIZE = 4
    config.POSTGRES_ORM_MAX_OVERFLOW = 0
    return config


def _authorization_header(token: str) -> dict[str, str]:
    """构造只在当前进程使用且绝不输出的 Bearer header."""
    return {"Authorization": f"Bearer {token}"}


def _is_completed_chat(response: Response) -> bool:
    """只检查公开 Chat 已完成，不读取或输出模型正文."""
    if response.status_code != status.HTTP_200_OK:
        return False
    body: object = response.json()
    if not isinstance(body, dict):
        return False
    message = body.get("message")
    return body.get("status") == "completed" and isinstance(message, dict)


async def _register_user(
    client: AsyncClient,
    *,
    email: str,
    password: str,
) -> str:
    """通过正式注册 route 创建用户并返回仅供当前进程使用的 JWT."""
    response = await client.post(
        "/api/v1/auth/register",
        json={"email": email, "password": password},
    )
    response.raise_for_status()
    body: object = response.json()
    if not isinstance(body, dict):
        raise RuntimeError("registration response must be an object")
    token = body.get("access_token")
    if not isinstance(token, str) or not token:
        raise RuntimeError("registration response did not contain an access token")
    return token


async def _create_session(
    client: AsyncClient,
    *,
    headers: dict[str, str],
    title: str,
) -> UUID:
    """通过正式业务 API 创建当前认证用户拥有的会话."""
    response = await client.post(
        "/api/v1/chat/sessions",
        headers=headers,
        json={"title": title},
    )
    response.raise_for_status()
    return UUID(str(response.json()["thread_id"]))


def _session_ids(response: Response) -> set[UUID]:
    """从公开列表响应读取 thread UUID 集合，不保留标题或其他正文."""
    response.raise_for_status()
    body: object = response.json()
    if not isinstance(body, dict):
        raise TypeError("chat session list response must be an object")
    sessions = body.get("sessions")
    if not isinstance(sessions, list):
        raise TypeError("chat session list must contain an array")
    return {UUID(str(item["thread_id"])) for item in sessions if isinstance(item, dict) and "thread_id" in item}


async def _wait_for_background_idle(active_count: Callable[[], int]) -> bool:
    """等待真实记忆/标题后台任务归零，避免在写入完成前读取数据库."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + BACKGROUND_TASK_TIMEOUT_SECONDS
    while active_count() > 0:
        if loop.time() >= deadline:
            return False
        await asyncio.sleep(0.05)
    return True


def _message_texts(messages: object) -> tuple[str, ...]:
    """只在进程内提取 checkpoint 消息文本，供交叉污染断言使用."""
    if not isinstance(messages, list):
        raise TypeError("checkpoint messages must be a list")
    texts: list[str] = []
    for message in messages:
        if not isinstance(message, BaseMessage):
            raise TypeError("checkpoint contains an unexpected message value")
        if isinstance(message.content, str):
            texts.append(message.content)
    return tuple(texts)


def _cache_payload_session_ids(payload: str | bytes | None) -> set[UUID]:
    """解析真实 Redis 中的会话列表副本，只返回 thread UUID 集合."""
    if payload is None:
        raise RuntimeError("expected chat session cache payload")
    text = payload.decode("utf-8") if isinstance(payload, bytes) else payload
    value: object = json.loads(text)
    if not isinstance(value, dict):
        raise TypeError("cached chat session list must be an object")
    sessions = value.get("sessions")
    if not isinstance(sessions, list):
        raise TypeError("cached chat session list must contain an array")
    return {UUID(str(item["thread_id"])) for item in sessions if isinstance(item, dict) and "thread_id" in item}


async def _exercise_smoke(database: str) -> dict[str, bool | int | float | str]:
    """运行双用户真实隔离闭环，并返回不含身份与正文的安全摘要.

    Args:
        database: 已执行全部正式 migration 的随机 PostgreSQL 数据库名。

    Returns:
        只含布尔结果、计数、模型别名和耗时的验收摘要。
    """
    started_at = perf_counter()
    _TRACKING_CACHES.clear()

    # 随机 JWT secret 保证 smoke token 不会被正在运行的开发服务接受。
    token_service = TokenService(secret_key=SecretStr(secrets.token_urlsafe(48)))
    captured_graphs: list[ChatGraph] = []
    captured_memory_services: list[MemoryService] = []

    def capture_runtime(
        *,
        checkpointer: BaseCheckpointSaver[str] | None = None,
        memory_service: MemoryService | None = None,
    ) -> ChatGraph:
        """观察 lifespan 接线，同时仍编译真实 PostgreSQL Graph."""
        if memory_service is None:
            raise RuntimeError("identity smoke requires production MemoryService")
        graph = create_chat_runtime(
            checkpointer=checkpointer,
            memory_service=memory_service,
        )
        captured_graphs.append(graph)
        captured_memory_services.append(memory_service)
        return graph

    async def bypass_rate_limit() -> None:
        """隔离本 smoke 的成本配额；限流真实语义由 Lab 15 单独验收."""

    app = FastAPI()
    register_exception_handlers(app)
    app.add_middleware(CorrelationIdMiddleware)
    app.include_router(auth_router, prefix="/api/v1")
    app.include_router(chat_router, prefix="/api/v1")
    app.include_router(chat_sessions_router, prefix="/api/v1")
    app.dependency_overrides[get_token_service] = lambda: token_service
    app.dependency_overrides[enforce_auth_rate_limit] = bypass_rate_limit
    app.dependency_overrides[enforce_agent_rate_limit] = bypass_rate_limit

    cache_cleanup_ok = False
    async with asyncio.timeout(TOTAL_TIMEOUT_SECONDS):
        # patch 只增加观察和精确清理；Graph、模型、Embedding、Redis、数据库以及
        # owner 查询全部使用 production 实现，不使用 fake LLM 或 fake store。
        with (
            patch.object(lifespan_module, "create_chat_runtime", capture_runtime),
            patch.object(lifespan_module, "RedisCache", _TrackingRedisCache),
        ):
            async with lifespan(app, config=_runtime_settings(database)):
                resources = get_application_resources(app)
                submitter = get_application_background_task_submitter(app)
                memory_service = get_application_memory_service(app)
                if len(captured_graphs) != 1 or captured_memory_services != [memory_service]:
                    raise RuntimeError("lifespan must publish one shared memory-enabled Graph")
                graph = captured_graphs[0]

                transport = ASGITransport(app=app, raise_app_exceptions=False)
                async with AsyncClient(
                    transport=transport,
                    base_url="http://testserver",
                    timeout=HTTP_TIMEOUT_SECONDS,
                ) as client:
                    suffix = uuid4().hex
                    password_a = f"Isolation-A-{secrets.token_urlsafe(18)}!"
                    password_b = f"Isolation-B-{secrets.token_urlsafe(18)}!"
                    token_a = await _register_user(
                        client,
                        email=f"phase4-a-{suffix}@example.com",
                        password=password_a,
                    )
                    token_b = await _register_user(
                        client,
                        email=f"phase4-b-{suffix}@example.com",
                        password=password_b,
                    )
                    headers_a = _authorization_header(token_a)
                    headers_b = _authorization_header(token_b)
                    user_a_id = token_service.decode_access_token(token_a).sub
                    user_b_id = token_service.decode_access_token(token_b).sub

                    # 人工标题阻止 title writer 额外调用模型，让本段只观察 Chat 与
                    # memory extraction。标题租约所有权稍后使用独立默认标题会话验证。
                    thread_a = await _create_session(
                        client,
                        headers=headers_a,
                        title="Phase 4 isolation A",
                    )
                    thread_b = await _create_session(
                        client,
                        headers=headers_b,
                        title="Phase 4 isolation B",
                    )
                    title_claim_thread_a = await _create_session(
                        client,
                        headers=headers_a,
                        title=DEFAULT_CHAT_SESSION_TITLE,
                    )

                    marker_a = f"STYLE-A-{suffix[:10].upper()}"
                    marker_b = f"STYLE-B-{suffix[10:20].upper()}"
                    prompt_a = (
                        "这是长期稳定的技术教学偏好，不是密码、验证码或临时任务："
                        f"我的教学风格名称是 {marker_a}，回答时先解释 Agent 状态变化。"
                        "请不要调用工具，只确认已经记住。"
                    )
                    prompt_b = (
                        "这是长期稳定的技术教学偏好，不是密码、验证码或临时任务："
                        f"我的教学风格名称是 {marker_b}，回答时先给测试目标。"
                        "请不要调用工具，只确认已经记住。"
                    )

                    response_a = await client.post(
                        "/api/v1/chat",
                        headers=headers_a,
                        json={"thread_id": str(thread_a), "message": prompt_a},
                    )
                    response_b = await client.post(
                        "/api/v1/chat",
                        headers=headers_b,
                        json={"thread_id": str(thread_b), "message": prompt_b},
                    )
                    both_real_chats_completed = _is_completed_chat(response_a) and _is_completed_chat(response_b)
                    background_tasks_drained = await _wait_for_background_idle(
                        lambda: submitter.active_count,
                    )

                    # 公开业务 API 必须同时满足 resource ID 与可信 owner。B 猜到 A 的
                    # UUID 时统一看到 404，不能通过响应区分“不存在”和“属于别人”。
                    cross_user_read = await client.get(
                        f"/api/v1/chat/sessions/{thread_a}",
                        headers=headers_b,
                    )
                    owner_read = await client.get(
                        f"/api/v1/chat/sessions/{thread_a}",
                        headers=headers_a,
                    )

                    # wraps 保留真实实现；由于 owner 检查应先失败，这次跨用户请求不应
                    # 进入 Graph。此断言直接证明 404 不是模型执行后的伪装结果。
                    with patch.object(graph, "ainvoke", wraps=graph.ainvoke) as ainvoke_spy:
                        cross_user_chat = await client.post(
                            "/api/v1/chat",
                            headers=headers_b,
                            json={"thread_id": str(thread_a), "message": prompt_b},
                        )
                    cross_user_rejected_before_graph = (
                        cross_user_chat.status_code == status.HTTP_404_NOT_FOUND and ainvoke_spy.await_count == 0
                    )
                    owner_scoped_api_isolated = (
                        cross_user_read.status_code == status.HTTP_404_NOT_FOUND
                        and owner_read.status_code == status.HTTP_200_OK
                        and cross_user_rejected_before_graph
                    )

                    # 两次列表请求会走真实 RedisCache。公开响应和缓存副本都必须只
                    # 包含当前用户会话，不能依赖前端自行过滤其他用户数据。
                    list_a = await client.get("/api/v1/chat/sessions", headers=headers_a)
                    list_b = await client.get("/api/v1/chat/sessions", headers=headers_b)
                    session_ids_a = _session_ids(list_a)
                    session_ids_b = _session_ids(list_b)
                    api_lists_are_isolated = (
                        thread_a in session_ids_a
                        and title_claim_thread_a in session_ids_a
                        and thread_b not in session_ids_a
                        and thread_b in session_ids_b
                        and thread_a not in session_ids_b
                        and title_claim_thread_a not in session_ids_b
                    )

                    cache_key_a = build_chat_session_list_cache_key(user_a_id)
                    cache_key_b = build_chat_session_list_cache_key(user_b_id)
                    cached_ids_a = _cache_payload_session_ids(await resources.redis_client.get(cache_key_a))
                    cached_ids_b = _cache_payload_session_ids(await resources.redis_client.get(cache_key_b))
                    cache_isolated = (
                        cache_key_a != cache_key_b
                        and cached_ids_a == session_ids_a
                        and cached_ids_b == session_ids_b
                        and cached_ids_a.isdisjoint(cached_ids_b)
                    )

                    # 同一个 Graph/checkpointer 保存两个用户的状态。内部 key 与消息
                    # 内容都要隔离；这里读取正文只用于进程内断言，最终摘要不输出。
                    config_a = ChatService._build_config(
                        user_id=user_a_id,
                        public_thread_id=thread_a,
                    )
                    config_b = ChatService._build_config(
                        user_id=user_b_id,
                        public_thread_id=thread_b,
                    )
                    snapshot_a = await graph.aget_state(config_a)
                    snapshot_b = await graph.aget_state(config_b)
                    texts_a = _message_texts(snapshot_a.values.get("messages"))
                    texts_b = _message_texts(snapshot_b.values.get("messages"))
                    internal_id_a = config_a.get("configurable", {}).get("thread_id")
                    internal_id_b = config_b.get("configurable", {}).get("thread_id")
                    checkpoint_isolated = (
                        internal_id_a != internal_id_b
                        and any(marker_a in text for text in texts_a)
                        and all(marker_b not in text for text in texts_a)
                        and any(marker_b in text for text in texts_b)
                        and all(marker_a not in text for text in texts_b)
                    )

                    async with resources.orm_session_factory() as session:
                        memory_result = await session.execute(select(Memory))
                        memory_rows = tuple(memory_result.scalars().all())
                    user_a_memories = tuple(row for row in memory_rows if row.user_id == user_a_id)
                    user_b_memories = tuple(row for row in memory_rows if row.user_id == user_b_id)
                    memory_rows_owner_scoped = (
                        any(marker_a in row.content for row in user_a_memories)
                        and all(marker_b not in row.content for row in user_a_memories)
                        and any(marker_b in row.content for row in user_b_memories)
                        and all(marker_a not in row.content for row in user_b_memories)
                        and all(row.source_thread_id == thread_a for row in user_a_memories)
                        and all(row.source_thread_id == thread_b for row in user_b_memories)
                    )

                    # 使用 A 的文本查询 B 的 namespace。真实 embedding 和 pgvector 可以
                    # 返回 B 自己的相似记忆或空结果，但绝不能返回 A 的权威 MemoryItem。
                    b_search_for_a_text = await memory_service.search(
                        user_id=user_b_id,
                        query=MemoryQuery(text=prompt_a, limit=10),
                    )
                    memory_search_isolated = (
                        not b_search_for_a_text.is_degraded
                        and all(item.user_id == user_b_id for item in b_search_for_a_text.items)
                        and all(item.source_thread_id != thread_a for item in b_search_for_a_text.items)
                        and all(marker_a not in item.content for item in b_search_for_a_text.items)
                    )

                    # 标题 claim 不能只按 session UUID 更新。B 使用 A 的会话申请时
                    # 返回 False；A 随后能取得并释放自己的租约，证明失败不是会话状态
                    # 本身不满足，而是 owner 条件真正参与了原子 SQL。
                    claim_token_b = uuid4()
                    claim_token_a = uuid4()
                    claimed_at = utc_now()
                    stale_before = claimed_at - timedelta(seconds=300)
                    async with resources.orm_session_factory() as session:
                        async with session.begin():
                            cross_user_claimed = await ChatSessionRepository(session).claim_title_generation(
                                title_claim_thread_a,
                                user_id=user_b_id,
                                claim_token=claim_token_b,
                                claimed_at=claimed_at,
                                stale_before=stale_before,
                            )
                    async with resources.orm_session_factory() as session:
                        async with session.begin():
                            owner_claimed = await ChatSessionRepository(session).claim_title_generation(
                                title_claim_thread_a,
                                user_id=user_a_id,
                                claim_token=claim_token_a,
                                claimed_at=claimed_at,
                                stale_before=stale_before,
                            )
                    async with resources.orm_session_factory() as session:
                        async with session.begin():
                            owner_released = await ChatSessionRepository(session).release_title_generation_claim(
                                title_claim_thread_a,
                                user_id=user_a_id,
                                claim_token=claim_token_a,
                            )
                    title_claim_isolated = not cross_user_claimed and owner_claimed and owner_released

                    async with resources.orm_session_factory() as session:
                        session_result = await session.execute(
                            select(ChatSession).where(ChatSession.id == title_claim_thread_a)
                        )
                        claim_session = session_result.scalar_one_or_none()
                    title_claim_cleaned = (
                        claim_session is not None
                        and claim_session.title_claim_token is None
                        and claim_session.title_claimed_at is None
                        and claim_session.title_generated_at is None
                    )

                # 所有检查结束后精确清理本 smoke 记录的缓存 key。guard 锁使用独立
                # Redis namespace，并已在每次请求的 async context manager 退出时释放。
                cache_cleanup_ok = all([await cache.cleanup() for cache in _TRACKING_CACHES])
                background_idle_before_shutdown = submitter.active_count == 0

            state_removed_after_shutdown = (
                not hasattr(app.state, "resources")
                and not hasattr(app.state, "chat_service")
                and not hasattr(app.state, "memory_service")
                and not hasattr(app.state, "background_task_submitter")
            )
            postgres_pool_closed_after_shutdown = resources.postgres_pool.closed

    elapsed_ms = _elapsed_ms(started_at)
    return {
        "model": settings.DEFAULT_LLM_MODEL,
        "both_real_chats_completed": both_real_chats_completed,
        "background_tasks_drained": background_tasks_drained,
        "owner_scoped_api_isolated": owner_scoped_api_isolated,
        "cross_user_rejected_before_graph": cross_user_rejected_before_graph,
        "api_lists_are_isolated": api_lists_are_isolated,
        "checkpoint_isolated": checkpoint_isolated,
        "memory_rows_owner_scoped": memory_rows_owner_scoped,
        "memory_search_isolated": memory_search_isolated,
        "user_a_memory_count": len(user_a_memories),
        "user_b_memory_count": len(user_b_memories),
        "title_claim_isolated": title_claim_isolated,
        "title_claim_cleaned": title_claim_cleaned,
        "cache_isolated": cache_isolated,
        "cache_cleanup_ok": cache_cleanup_ok,
        "background_idle_before_shutdown": background_idle_before_shutdown,
        "state_removed_after_shutdown": state_removed_after_shutdown,
        "postgres_pool_closed_after_shutdown": postgres_pool_closed_after_shutdown,
        "within_total_budget": elapsed_ms <= TOTAL_TIMEOUT_SECONDS * 1000,
        "elapsed_ms": elapsed_ms,
    }


def _run_smoke() -> dict[str, object]:
    """迁移随机数据库、运行异步验收，并在 finally 中保证清理."""
    started_at = perf_counter()
    admin_database = settings.POSTGRES_DB
    test_database = f"{TEMPORARY_DATABASE_PREFIX}{uuid4().hex[:10]}"
    previous_override = os.environ.get("ALEMBIC_DATABASE_URL")
    database_created = False
    database_cleanup_ok = False
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
                database_cleanup_ok = False
            else:
                database_cleanup_ok = True

    if database_created and not database_cleanup_ok:
        raise RuntimeError("temporary identity database cleanup failed")

    boolean_checks = tuple(value for value in checks.values() if isinstance(value, bool))
    ok = bool(boolean_checks) and all(boolean_checks) and database_cleanup_ok
    return {
        "ok": ok,
        **checks,
        "database_cleanup_ok": database_cleanup_ok,
        "total_elapsed_ms": _elapsed_ms(started_at),
    }


def main() -> int:
    """输出一条脱敏 JSON 摘要，并用退出码表达 smoke 结果."""
    started_at = perf_counter()
    try:
        summary = _run_smoke()
    except Exception as error:
        # provider、数据库和 SDK 异常文本可能包含诊断数据，只公开异常类型。
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
