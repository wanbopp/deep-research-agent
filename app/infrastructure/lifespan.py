"""在 FastAPI lifespan 中创建、发布并关闭共享基础设施与应用服务."""

import asyncio
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager
from typing import cast

from fastapi import FastAPI
from pydantic import SecretStr

from app.agents.chat.runtime import create_chat_runtime
from app.core.config import Settings, settings
from app.core.logging import logger
from app.infrastructure.background_tasks import AsyncioBackgroundTaskSubmitter
from app.infrastructure.chat_session_ownership import (
    PostgresChatSessionOwnershipVerifier,
)
from app.infrastructure.chat_guard import RedisChatExecutionGuard
from app.infrastructure.cache import RedisCache
from app.infrastructure.embeddings import OpenAITextEmbedder
from app.infrastructure.factory import create_application_resources
from app.infrastructure.memory import PostgresMemoryStore
from app.infrastructure.neo4j import probe_neo4j
from app.infrastructure.postgres import probe_postgres
from app.infrastructure.probes import DependencyName, DependencyProbeResult
from app.infrastructure.rate_limit import RedisRateLimiter
from app.infrastructure.redis import probe_redis
from app.infrastructure.resources import ApplicationResources
from app.schemas.llm import ModelSpec
from app.services.chat import ChatService
from app.services.cache import Cache
from app.services.chat_session_cleanup import ChatSessionCleanupService
from app.services.llm.factory import create_openai_chat_model
from app.services.llm.registry import LLMRegistry
from app.services.llm.service import LLMService
from app.services.memory_extraction import (
    BackgroundChatMemoryWriter,
    ChatMemoryWriter,
    LLMMemoryExtractor,
)
from app.services.memory_service import MemoryService
from app.services.rate_limit import RateLimiter, RateLimitPolicies, RateLimitPolicy

STARTUP_PROBE_TIMEOUT_SECONDS = 5.0
STARTUP_TOTAL_TIMEOUT_SECONDS = 8.0
CHAT_GRAPH_TIMEOUT_SECONDS = 90.0
# Lease 必须长于 Graph 总超时。额外 30 秒用于事件循环调度和 finally 中的 Redis
# 释放；进程崩溃时 Redis 仍会在 120 秒内自动清除遗留锁。
CHAT_GUARD_LEASE_SECONDS = 120.0
MEMORY_EXTRACTION_MODEL_ALIAS = "memory-extraction"
REQUIRED_DEPENDENCIES = frozenset({DependencyName.POSTGRES})


def get_application_cache(app: FastAPI) -> Cache:
    """读取 lifespan 已创建的共享缓存适配器.

    Args:
        app: 当前 FastAPI 应用实例。

    Returns:
        借用 ``ApplicationResources.redis_client`` 的 RedisCache。返回类型使用
        应用层 Cache Protocol，调用方不依赖 redis-py。

    Raises:
        RuntimeError: startup 尚未发布缓存，或 shutdown 已经撤下缓存引用。
    """
    try:
        cache = app.state.cache
    except AttributeError as exc:
        raise RuntimeError("Application cache is not initialized") from exc
    return cast(Cache, cache)


def get_application_chat_service(app: FastAPI) -> ChatService:
    """读取 lifespan 已创建的共享 ChatService."""
    try:
        chat_service = app.state.chat_service
    except AttributeError as exc:
        raise RuntimeError("Application chat service is not initialized") from exc
    # cast 只帮助静态类型检查，不会在运行时转换或复制 ChatService。
    return cast(ChatService, chat_service)


def get_application_memory_service(app: FastAPI) -> MemoryService:
    """读取 lifespan 已创建的共享长期记忆应用服务.

    Args:
        app: 当前 FastAPI 应用实例。

    Returns:
        与 production Graph 共用的无请求状态 ``MemoryService``。它内部只保存
        Store、Cache 和策略对象，不保存当前 user_id、query 或检索结果。

    Raises:
        RuntimeError: startup 尚未发布服务，或 shutdown 已撤下服务引用。

    Notes:
        严格 getter 不创建 fallback service。若生命周期边界外偷偷构造另一个
        MemoryService，Graph 和后台写入可能使用不同缓存 generation 与资源配置。
    """
    try:
        memory_service = app.state.memory_service
    except AttributeError as exc:
        raise RuntimeError("Application memory service is not initialized") from exc
    return cast(MemoryService, memory_service)


def get_application_chat_memory_writer(app: FastAPI) -> ChatMemoryWriter:
    """读取 lifespan 已创建并注入 ChatService 的后台记忆写入边界.

    Args:
        app: 当前 FastAPI 应用实例。

    Returns:
        与 production ChatService 共用的无请求状态 writer。

    Raises:
        RuntimeError: startup 尚未发布 writer，或 shutdown 已撤下引用。
    """
    try:
        writer = app.state.chat_memory_writer
    except AttributeError as exc:
        raise RuntimeError("Application chat memory writer is not initialized") from exc
    return cast(ChatMemoryWriter, writer)


def get_application_background_task_submitter(
    app: FastAPI,
) -> AsyncioBackgroundTaskSubmitter:
    """读取当前 worker 的后台任务提交器.

    12D 暴露该对象是为了验证 production 只使用一个任务集合；12E 会在同一对象上
    增加 shutdown drain/cancel，而不是再创建第二套无法统一收敛的任务跟踪器。
    """
    try:
        submitter = app.state.background_task_submitter
    except AttributeError as exc:
        raise RuntimeError("Application background task submitter is not initialized") from exc
    return cast(AsyncioBackgroundTaskSubmitter, submitter)


def get_application_resources(app: FastAPI) -> ApplicationResources:
    """读取已经由 lifespan 初始化的共享资源."""
    try:
        resources = app.state.resources
    except AttributeError as exc:
        raise RuntimeError("Application resources are not initialized") from exc
    return cast(ApplicationResources, resources)  # 不做任何转换 声明返回值是 ApplicationResources


def get_application_chat_cleanup_service(
    app: FastAPI,
) -> ChatSessionCleanupService:
    """读取 lifespan 已创建的共享会话清理协调器.

    Args:
        app: 当前 FastAPI 应用实例。

    Returns:
        与 ChatService 共用 guard、key 映射和 checkpointer 的无状态协调器。

    Raises:
        RuntimeError: startup 尚未完成或 shutdown 已撤下应用服务。
    """
    try:
        cleanup_service = app.state.chat_session_cleanup_service
    except AttributeError as exc:
        raise RuntimeError("Application chat cleanup service is not initialized") from exc
    return cast(ChatSessionCleanupService, cleanup_service)


def get_application_rate_limiter(app: FastAPI) -> RateLimiter:
    """读取 lifespan 创建的共享限流适配器.

    Args:
        app: 当前 FastAPI 应用实例。

    Returns:
        借用 ``ApplicationResources.redis_client`` 的应用层 RateLimiter。

    Raises:
        RuntimeError: startup 尚未发布限流器，或 shutdown 已撤下引用。
    """
    try:
        limiter = app.state.rate_limiter
    except AttributeError as exc:
        raise RuntimeError("Application rate limiter is not initialized") from exc
    return cast(RateLimiter, limiter)


def get_application_rate_limit_policies(app: FastAPI) -> RateLimitPolicies:
    """读取 startup 已校验并发布的固定限流策略.

    Args:
        app: 当前 FastAPI 应用实例。

    Returns:
        认证入口与 Agent 入口各自的不可变策略。

    Raises:
        RuntimeError: 应用不在可处理请求的 lifespan 阶段。
    """
    try:
        policies = app.state.rate_limit_policies
    except AttributeError as exc:
        raise RuntimeError("Application rate limit policies are not initialized") from exc
    return cast(RateLimitPolicies, policies)


def _create_memory_extractor(config: Settings) -> LLMMemoryExtractor:
    """根据当前环境配置组合长期记忆结构化提取器.

    Args:
        config: lifespan 本次启动使用的配置快照。不能读取另一个全局 Settings，
            否则临时数据库 smoke 或多环境启动可能出现两套 provider 配置。

    Returns:
        只保存 LLMService 与 alias 快照的无请求状态提取器。

    Raises:
        RuntimeError: 没有配置模型 API key。
        ValidationError: 模型名称、token 或 timeout 配置违反 ModelSpec 边界。

    Notes:
        Chat Agent 与记忆提取使用相同 provider 配置，但分别拥有 Registry/LLMService。
        两个服务都不保存当前请求状态；独立 alias 让后续可以单独调整提取模型、温度
        和输出预算。构造 ChatOpenAI 所需配置不会发送网络请求，只有后台真正执行
        ``call_structured()`` 时才会访问 provider。
    """
    if not config.OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is required to create memory extractor")

    extraction_spec = ModelSpec(
        alias=MEMORY_EXTRACTION_MODEL_ALIAS,
        provider_model=config.DEFAULT_LLM_MODEL,
        api_key=SecretStr(config.OPENAI_API_KEY),
        base_url=config.OPENAI_BASE_URL,
        temperature=0.0,
        # 结构化结果至多包含一条短记忆，无需占用普通聊天的全部输出预算。
        max_tokens=min(config.MAX_TOKENS, 800),
        request_timeout_seconds=max(
            1.0,
            config.LLM_TOTAL_TIMEOUT * 0.75,
        ),
    )
    extraction_registry = LLMRegistry(
        specs=(extraction_spec,),
        factory=create_openai_chat_model,
    )
    extraction_llm_service = LLMService(
        extraction_registry,
        max_attempts=config.MAX_LLM_CALL_RETRIES,
        retry_wait_multiplier=0.2,
        total_timeout_seconds=config.LLM_TOTAL_TIMEOUT,
    )
    return LLMMemoryExtractor(
        extraction_llm_service,
        aliases=(MEMORY_EXTRACTION_MODEL_ALIAS,),
    )


@asynccontextmanager
async def lifespan(
    app: FastAPI,
    *,
    config: Settings = settings,
) -> AsyncIterator[None]:
    """管理 FastAPI 应用级基础设施资源的完整生命周期.

    Args:
        app: 将在 startup 成功后接收共享资源的 FastAPI 应用。
        config: 用于构造本进程资源的配置。正式应用使用模块级 settings；真实
            隔离 smoke 可显式传入只修改数据库名的 Settings，避免污染开发库。

    Yields:
        startup 已完成且 app.state.resources 可用的应用运行阶段。

    Raises:
        RuntimeError: required dependency 不健康，或 checkpointer setup 失败。异常
            离开 AsyncExitStack 前仍会执行已经登记的资源清理回调。
    """
    # factory 会构造 AsyncPostgresSaver，因此必须在这里的运行中 event loop 内
    # 调用，不能提前到模块全局。此时仍没有连接数据库，也没有执行 DDL。
    resources = create_application_resources(config)

    async with AsyncExitStack() as stack:
        # 建议登记顺序：执行完成后逆向关闭
        # push_async_callback 只登记清理动作，此时不会执行 close。
        #
        # AsyncExitStack 按照后进先出（LIFO）执行：
        # 1. 最后登记 Redis，所以最先关闭 Redis；
        # 2. 然后关闭 Neo4j；
        # 3. 再释放 SQLAlchemy ORM engine 的连接池；
        # 4. 最后关闭原生 PostgreSQL pool。
        stack.push_async_callback(resources.postgres_pool.close)
        stack.push_async_callback(resources.orm_engine.dispose)
        stack.push_async_callback(resources.neo4j_driver.close)
        stack.push_async_callback(resources.redis_client.aclose)

        # factory 使用 open=False，因此必须在 startup 显式打开。
        await resources.postgres_pool.open()

        # 通过 asyncio.gather 并发执行三个 probe。
        # 总预算略大于单探针预算，避免两层 timeout 在同一时刻触发竞态。
        async with asyncio.timeout(STARTUP_TOTAL_TIMEOUT_SECONDS):
            results: tuple[DependencyProbeResult, ...] = await asyncio.gather(
                probe_postgres(resources.postgres_pool, timeout_seconds=STARTUP_PROBE_TIMEOUT_SECONDS),
                probe_neo4j(resources.neo4j_driver, timeout_seconds=STARTUP_PROBE_TIMEOUT_SECONDS),
                probe_redis(resources.redis_client, timeout_seconds=STARTUP_PROBE_TIMEOUT_SECONDS),
            )

        # 找出不健康的 required dependency。
        required_failures = [
            result for result in results if result.name in REQUIRED_DEPENDENCIES and not result.is_healthy
        ]

        # required_failures 非空时记录安全日志并抛出固定 RuntimeError。
        # 日志只能包含依赖名和 error_code。
        if required_failures:
            logger.error(
                "required_dependency_unhealthy",
                failures=[
                    {
                        "dependency": result.name.value,
                        "error_code": (result.error_code.value if result.error_code is not None else "unknown_error"),
                    }
                    for result in required_failures
                ],
            )
            raise RuntimeError(
                f"Required dependencies unhealthy: {', '.join(r.name.value for r in required_failures)}"
            )

        # PostgreSQL 已通过 required probe 后，才运行 LangGraph 自有 migration。
        # setup() 必须发生在 app.state 发布之前：如果它失败，FastAPI startup
        # 整体失败，任何 route 都看不到半初始化 checkpointer。
        try:
            await resources.checkpointer.setup()
        except Exception as error:
            # 保留服务端 traceback 便于定位 migration/权限问题；结构化字段只含
            # 异常类型，不主动记录 SQL、连接串或数据库凭据。
            logger.exception(
                "checkpointer_setup_failed",
                error_type=type(error).__name__,
            )
            raise RuntimeError("PostgreSQL checkpointer setup failed") from error

        logger.info(
            "checkpointer_initialized",
            backend="postgres",
        )

        # 3. 先组合长期记忆依赖，再编译 production Graph。
        #
        # RedisCache 只借用 resources.redis_client，不拥有连接；MemoryStore 只保存
        # sessionmaker，每次调用才创建独立 AsyncSession；Embedder 客户端也是无请求
        # 状态的惰性适配器，构造时不会发送真实 provider 请求。因此三者和
        # MemoryService 都可以跨请求共享，同时不会共享某个用户或数据库事务。
        cache = RedisCache(resources.redis_client)
        memory_embedder = OpenAITextEmbedder.from_settings(config)
        memory_store = PostgresMemoryStore(
            session_factory=resources.orm_session_factory,
            embedder=memory_embedder,
        )
        memory_service = MemoryService(
            memory_store,
            cache,
            search_cache_ttl_seconds=config.MEMORY_SEARCH_CACHE_TTL_SECONDS,
            generation_ttl_seconds=config.MEMORY_CACHE_GENERATION_TTL_SECONDS,
        )

        # 记忆写入链也在 startup 组合一次，但不绑定当前用户或会话：
        # - extractor 只把对话转换成不含身份的 content/kind 候选；
        # - writer 在每次 submit_turn() 中接收可信 user_id/source_thread_id；
        # - submitter 为本 worker 的任务保留强引用并消费异常。
        # 12D 的 submitter 尚不在 shutdown 等待任务；12E 会在同一实例上补齐
        # drain/cancel，避免应用退出时底层 Redis/ORM 已关闭而任务仍在使用它们。
        background_task_submitter = AsyncioBackgroundTaskSubmitter()
        memory_extractor = _create_memory_extractor(config)
        chat_memory_writer = BackgroundChatMemoryWriter(
            extractor=memory_extractor,
            memory_service=memory_service,
            task_submitter=background_task_submitter,
        )

        # Graph 编译时只把共享 MemoryService 放入 memory node 闭包。当前用户身份
        # 仍要等每次 ChatService 调用时通过 ChatRuntimeContext 注入，绝不能在
        # startup 阶段绑定到共享 Graph 或 MemoryService 实例。
        graph = create_chat_runtime(
            checkpointer=resources.checkpointer,
            memory_service=memory_service,
        )

        # 4. guard 复用 lifespan 的共享 Redis client，但每次 hold 创建独立 Lock 和
        # owner token。Redis 当前仍是 optional dependency；若运行期间不可用，guard
        # 会 fail-closed 为稳定 503/SSE error，绝不会绕过互斥继续调用模型。
        execution_guard = RedisChatExecutionGuard(
            resources.redis_client,
            lease_seconds=CHAT_GUARD_LEASE_SECONDS,
        )

        # 5. verifier 可以跨请求共享，因为它只保存 sessionmaker；每次授权检查才
        # 创建独立短生命周期 AsyncSession。不能把一个 Session 放入 app.state 或
        # ChatService，否则并发 Graph 会共享事务、identity map 和失败状态。
        ownership_verifier = PostgresChatSessionOwnershipVerifier(
            resources.orm_session_factory,
        )

        # RateLimiter 与 Cache 可以借用同一个 Redis client，但两者语义完全独立。
        # 缓存读取失败允许业务回源数据库；限流无法判断时，高成本 Agent 请求必须
        # fail-closed 为 503，不能把 Redis 故障当作“仍有额度”。
        rate_limiter = RedisRateLimiter(resources.redis_client)
        rate_limit_policies = RateLimitPolicies(
            auth=RateLimitPolicy(
                name="auth",
                limit=config.AUTH_RATE_LIMIT_REQUESTS,
                window_seconds=config.AUTH_RATE_LIMIT_WINDOW_SECONDS,
            ),
            agent=RateLimitPolicy(
                name="agent",
                limit=config.AGENT_RATE_LIMIT_REQUESTS,
                window_seconds=config.AGENT_RATE_LIMIT_WINDOW_SECONDS,
            ),
        )

        # 6. ChatService 持有共享 Graph、guard、无状态 verifier 和 memory writer。
        # 它们都不保存当前用户；身份、公开 thread ID 和本轮消息仍由每次请求传入。
        # writer.submit_turn() 只提交任务，所以已完成响应不等待第二次模型调用。
        chat_service = ChatService(
            graph,
            execution_guard=execution_guard,
            ownership_verifier=ownership_verifier,
            graph_timeout_seconds=CHAT_GRAPH_TIMEOUT_SECONDS,
            memory_writer=chat_memory_writer,
        )

        # 7. cleanup coordinator 与 ChatService 共用同一 guard，确保删除和 Graph
        # 落在同一个跨 worker 临界区。内部 key factory 也直接复用 ChatService
        # 当前实现，避免两处字符串拼接随时间漂移成两套 checkpoint 身份空间。
        chat_session_cleanup_service = ChatSessionCleanupService(
            session_factory=resources.orm_session_factory,
            checkpoint_store=resources.checkpointer,
            execution_guard=execution_guard,
            internal_thread_id_factory=ChatService._build_checkpoint_thread_id,
            cache=cache,
        )

        # 8. 所有 startup 步骤成功后，才同时发布底层资源和上层应用服务。
        # yield 前 FastAPI 尚未接收请求，因此请求不会看到只发布一半的状态。
        app.state.resources = resources
        app.state.cache = cache
        app.state.rate_limiter = rate_limiter
        app.state.rate_limit_policies = rate_limit_policies
        app.state.memory_service = memory_service
        app.state.background_task_submitter = background_task_submitter
        app.state.chat_memory_writer = chat_memory_writer
        app.state.chat_service = chat_service
        app.state.chat_session_cleanup_service = chat_session_cleanup_service

        logger.info(
            "infrastructure_initialized",
            dependencies={result.name.value: result.status.value for result in results},
        )

        try:
            # yield 之后，FastAPI 才开始对外处理 HTTP/Agent 请求。
            yield
        finally:
            # shutdown 先撤下依赖资源的上层 service，再撤下底层资源引用。
            # 真正的客户端和连接池随后由 AsyncExitStack 按逆序关闭。
            del app.state.chat_session_cleanup_service
            del app.state.chat_service
            # ChatService 先撤下，新的 HTTP 请求便无法再提交任务。writer 和
            # submitter 随后撤下公开入口；当前 12D 不主动取消已提交任务，12E 会在
            # 删除这些引用前执行有界 drain/cancel。
            del app.state.chat_memory_writer
            del app.state.background_task_submitter
            # 先撤下所有会调用 MemoryService 的上层对象，再撤下 service 本身。
            # MemoryService 没有 close 方法，因为 Redis client 和 ORM engine 的唯一
            # 所有者仍是 ApplicationResources，随后由 AsyncExitStack 统一关闭。
            del app.state.memory_service
            del app.state.rate_limit_policies
            del app.state.rate_limiter
            del app.state.cache
            del app.state.resources
            logger.info("infrastructure_shutdown_started")

    # 离开 AsyncExitStack 时，三个 close callback 已经执行完毕。
    logger.info("infrastructure_shutdown_completed")
