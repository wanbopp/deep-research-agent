"""Manage shared infrastructure resources across the FastAPI lifespan."""

import asyncio
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager
from typing import cast

from fastapi import FastAPI

from app.core.config import settings
from app.core.logging import logger
from app.infrastructure.factory import create_application_resources
from app.infrastructure.neo4j import probe_neo4j
from app.infrastructure.postgres import probe_postgres
from app.infrastructure.probes import DependencyName, DependencyProbeResult
from app.infrastructure.redis import probe_redis
from app.infrastructure.resources import ApplicationResources

STARTUP_PROBE_TIMEOUT_SECONDS = 5.0
STARTUP_TOTAL_TIMEOUT_SECONDS = 8.0
REQUIRED_DEPENDENCIES = frozenset({DependencyName.POSTGRES})


def get_application_resources(app: FastAPI) -> ApplicationResources:
    """读取已经由 lifespan 初始化的共享资源."""
    try:
        resources = app.state.resources
    except AttributeError as exc:
        raise RuntimeError("Application resources are not initialized") from exc
    return cast(ApplicationResources, resources)  # 不做任何转换 声明返回值是 ApplicationResources


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """管理 FastAPI 应用级基础设施资源的完整生命周期."""
    resources = create_application_resources(settings)

    async with AsyncExitStack() as stack:
        # 建议登记顺序：执行完成后逆向关闭
        # push_async_callback 只登记清理动作，此时不会执行 close。
        #
        # AsyncExitStack 按照后进先出（LIFO）执行：
        # 1. 最后登记 Redis，所以最先关闭 Redis；
        # 2. 然后关闭 Neo4j；
        # 3. 最后关闭 PostgreSQL pool。
        stack.push_async_callback(resources.postgres_pool.close)
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
        # startup 验收通过后才把 resources 保存到 app.state。
        app.state.resources = resources

        logger.info(
            "infrastructure_initialized",
            dependencies={result.name.value: result.status.value for result in results},
        )

        try:
            # yield 之后，FastAPI 才开始对外处理 HTTP/Agent 请求。
            yield
        finally:
            # shutdown 开始后先移除引用，避免关闭期间继续获取资源。
            del app.state.resources
            logger.info("infrastructure_shutdown_started")

    # 离开 AsyncExitStack 时，三个 close callback 已经执行完毕。
    logger.info("infrastructure_shutdown_completed")
