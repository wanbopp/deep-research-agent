"""健康检查响应模型."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """API 进程健康响应检查."""

    status: Literal["healthy"]  # 唯一允许的值是 healthy
    version: str
    environment: str
    timestamp: datetime


class LivenessResponse(BaseModel):
    """表示 API 进程仍然能够处理 HTTP 请求."""

    status: Literal["alive"]
    version: str
    environment: str
    timestamp: datetime


class DependencyHealthResponse(BaseModel):
    """单个基础设施依赖对外公开的安全状态."""

    name: Literal["postgres", "neo4j", "redis"]
    required: bool
    status: Literal["healthy", "unhealthy"]
    latency_ms: float = Field(ge=0)
    error_code: (
        Literal[
            "timeout",
            "authentication_error",
            "connection_error",
            "unknown_error",
        ]
        | None
    ) = None


class ReadinessResponse(BaseModel):
    """表示当前实例是否具备继续接收业务流量的条件."""

    status: Literal["ready", "degraded", "not_ready"]
    version: str
    environment: str
    timestamp: datetime
    dependencies: list[DependencyHealthResponse]
