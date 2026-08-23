"""Health and root endpoints."""

import asyncio
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Literal

from fastapi import APIRouter, Request, Response, status

from app.core.config import settings
from app.core.logging import logger
from app.infrastructure.lifespan import REQUIRED_DEPENDENCIES, get_application_resources
from app.infrastructure.neo4j import probe_neo4j
from app.infrastructure.postgres import probe_postgres
from app.infrastructure.probes import DependencyProbeResult
from app.infrastructure.redis import probe_redis
from app.schemas.health import (
    DependencyHealthResponse,
    HealthResponse,
    LivenessResponse,
    ReadinessResponse,
)

router = APIRouter(tags=["health"])

READINESS_PROBE_TIMEOUT_SECONDS = 1.0


@router.get("/")
async def root() -> dict:
    """根路径：返回项目基本信息."""
    logger.info("root_endpoint_called")

    return {
        "name": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "environment": settings.ENVIRONMENT.value,
        "docs_url": "/docs",
    }


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """健康检查：先只检查 API 进程是否活着."""
    logger.info("health_check_called")

    return HealthResponse(
        status="healthy",
        version=settings.VERSION,
        environment=settings.ENVIRONMENT.value,
        timestamp=datetime.now(UTC),
    )


@router.get("/health/live", response_model=LivenessResponse)
async def liveness_check() -> LivenessResponse:
    """证明当前 API 进程仍然能够处理 HTTP 请求.

    Liveness 故意不读取 app.state，也不访问任何外部依赖。
    外部依赖状态属于 readiness 的职责。
    """
    logger.info("liveness_check_called")

    return LivenessResponse(
        status="alive",
        version=settings.VERSION,
        environment=settings.ENVIRONMENT.value,
        timestamp=datetime.now(UTC),
    )


type ReadinessStatus = Literal["ready", "degraded", "not_ready"]


def _to_dependency_health(
    result: DependencyProbeResult,
) -> DependencyHealthResponse:
    """把内部探针结果转换为允许通过 HTTP 公开的状态.

    model_validate() 会再次检查枚举字符串是否属于公开响应允许的范围。
    原始异常对象和异常文本不会进入 HTTP 响应。
    """
    return DependencyHealthResponse.model_validate(
        {
            "name": result.name.value,
            "required": result.name in REQUIRED_DEPENDENCIES,
            "status": result.status.value,
            "latency_ms": result.latency_ms,
            "error_code": (result.error_code.value if result.error_code is not None else None),
        }
    )


def _aggregate_readiness(
    results: Sequence[DependencyProbeResult],
) -> ReadinessStatus:
    """根据必需和可选依赖状态计算实例整体 readiness."""
    # 必需依赖失败意味着新业务请求无法被可靠处理。
    required_failed = any(result.name in REQUIRED_DEPENDENCIES and not result.is_healthy for result in results)
    if required_failed:
        return "not_ready"

    # 走到这里说明必需依赖健康。
    # 如果只有可选依赖失败，核心请求仍可接收，但能力有所下降。
    if any(not result.is_healthy for result in results):
        return "degraded"

    return "ready"


@router.get(
    "/health/ready",
    response_model=ReadinessResponse,
    responses={
        status.HTTP_503_SERVICE_UNAVAILABLE: {
            "model": ReadinessResponse,
            "description": "A required dependency is unavailable.",
        }
    },
)
async def readiness_check(
    request: Request,
    response: Response,
) -> ReadinessResponse:
    """检查当前实例是否适合接收新的业务和 Agent 请求."""
    # lifespan 是资源所有者；route 只取得并借用已有资源。
    resources = get_application_resources(request.app)

    # 三个探针互不依赖，因此并发执行。
    # gather 的结果顺序与传入协程的顺序保持一致。
    results: tuple[DependencyProbeResult, ...] = await asyncio.gather(
        probe_postgres(
            resources.postgres_pool,
            timeout_seconds=READINESS_PROBE_TIMEOUT_SECONDS,
        ),
        probe_neo4j(
            resources.neo4j_driver,
            timeout_seconds=READINESS_PROBE_TIMEOUT_SECONDS,
        ),
        probe_redis(
            resources.redis_client,
            timeout_seconds=READINESS_PROBE_TIMEOUT_SECONDS,
        ),
    )

    readiness_status = _aggregate_readiness(results)
    dependencies = [_to_dependency_health(result) for result in results]

    # 只有 required 依赖失败才把 HTTP 状态改成 503。
    # optional 依赖失败虽然是 degraded，但实例仍可以接收核心请求。
    if readiness_status == "not_ready":
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    logger.info(
        "readiness_check_completed",
        readiness_status=readiness_status,
        dependencies={dependency.name: dependency.status for dependency in dependencies},
    )

    return ReadinessResponse(
        status=readiness_status,
        version=settings.VERSION,
        environment=settings.ENVIRONMENT.value,
        timestamp=datetime.now(UTC),
        dependencies=dependencies,
    )
