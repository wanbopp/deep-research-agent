"""Manage shared infrastructure resources across the FastAPI lifespan."""

import asyncio
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager
from typing import cast

from fastapi import FastAPI

from app.agents.chat.runtime import create_chat_runtime
from app.core.config import Settings, settings
from app.core.logging import logger
from app.infrastructure.chat_session_ownership import (
    PostgresChatSessionOwnershipVerifier,
)
from app.infrastructure.chat_guard import RedisChatExecutionGuard
from app.infrastructure.factory import create_application_resources
from app.infrastructure.neo4j import probe_neo4j
from app.infrastructure.postgres import probe_postgres
from app.infrastructure.probes import DependencyName, DependencyProbeResult
from app.infrastructure.redis import probe_redis
from app.infrastructure.resources import ApplicationResources
from app.services.chat import ChatService
from app.services.chat_session_cleanup import ChatSessionCleanupService

STARTUP_PROBE_TIMEOUT_SECONDS = 5.0
STARTUP_TOTAL_TIMEOUT_SECONDS = 8.0
CHAT_GRAPH_TIMEOUT_SECONDS = 90.0
# Lease 必须长于 Graph 总超时。额外 30 秒用于事件循环调度和 finally 中的 Redis
# 释放；进程崩溃时 Redis 仍会在 120 秒内自动清除遗留锁。
CHAT_GUARD_LEASE_SECONDS = 120.0
REQUIRED_DEPENDENCIES = frozenset({DependencyName.POSTGRES})


def get_application_chat_service(app: FastAPI) -> ChatService:
    """读取 lifespan 已创建的共享 ChatService."""
    try:
        chat_service = app.state.chat_service
    except AttributeError as exc:
        raise RuntimeError("Application chat service is not initialized") from exc
    # cast 只帮助静态类型检查，不会在运行时转换或复制 ChatService。
    return cast(ChatService, chat_service)


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

        # 3. 编译 production Graph。
        graph = create_chat_runtime(
            checkpointer=resources.checkpointer,
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

        # 6. ChatService 持有共享 Graph、guard 和无状态 verifier。三者都不保存
        # 当前用户；身份和公开 thread ID 仍由每次请求传入并构造独立内部 key。
        chat_service = ChatService(
            graph,
            execution_guard=execution_guard,
            ownership_verifier=ownership_verifier,
            graph_timeout_seconds=CHAT_GRAPH_TIMEOUT_SECONDS,
        )

        # 7. cleanup coordinator 与 ChatService 共用同一 guard，确保删除和 Graph
        # 落在同一个跨 worker 临界区。内部 key factory 也直接复用 ChatService
        # 当前实现，避免两处字符串拼接随时间漂移成两套 checkpoint 身份空间。
        chat_session_cleanup_service = ChatSessionCleanupService(
            session_factory=resources.orm_session_factory,
            checkpoint_store=resources.checkpointer,
            execution_guard=execution_guard,
            internal_thread_id_factory=ChatService._build_checkpoint_thread_id,
        )

        # 8. 所有 startup 步骤成功后，才同时发布底层资源和上层应用服务。
        # yield 前 FastAPI 尚未接收请求，因此请求不会看到只发布一半的状态。
        app.state.resources = resources
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
            del app.state.resources
            logger.info("infrastructure_shutdown_started")

    # 离开 AsyncExitStack 时，三个 close callback 已经执行完毕。
    logger.info("infrastructure_shutdown_completed")
