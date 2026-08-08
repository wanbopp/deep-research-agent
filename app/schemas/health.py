"""健康检查响应模型."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel


class HealthResponse(BaseModel):
    """API 进程健康响应检查."""

    status: Literal["healthy"]  # 唯一允许的值是 healthy
    version: str
    environment: str
    timestamp: datetime
