"""API v1 router aggregation."""

from typing import Any

from fastapi import APIRouter, status

from app.api.v1.auth import router as auth_router
from app.api.v1.chat import router as chat_router
from app.api.v1.health import router as health_router
from app.schemas.base import ErrorResponse

# 这里 OPENAPI 元数据，不负责处理真正的异常
# 实际的错误响应仍由 exception_handlers.py 生成.

COMMON_ERROR_RESPONSES: dict[int | str, dict[str, Any]] = {
    status.HTTP_404_NOT_FOUND: {
        "model": ErrorResponse,
        "description": "Resource not found",
    },
    status.HTTP_422_UNPROCESSABLE_CONTENT: {
        "model": ErrorResponse,
        "description": "Request validation failed",
    },
    status.HTTP_500_INTERNAL_SERVER_ERROR: {
        "model": ErrorResponse,
        "description": "Internal server error",
    },
}

# 放在聚合 router 上，使被纳入 API V1 的路由共享错误文档。
api_router = APIRouter(responses=COMMON_ERROR_RESPONSES)

api_router.include_router(health_router)

api_router.include_router(chat_router)

api_router.include_router(auth_router)
