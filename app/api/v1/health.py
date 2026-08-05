"""Health and root endpoints."""
from datetime import datetime

from fastapi import APIRouter

from app.core.config import settings
from app.core.logging import logger

router = APIRouter(tags=["health"])


@router.get("/")
async def root() -> dict:
    """根路径：返回项目基本信息。"""
    logger.info("root_endpoint_called")

    return {
        "name": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "environment": settings.ENVIRONMENT.value,
        "docs_url": "/docs",
    }


@router.get("/health")
async def health_check() -> dict:
    """健康检查：先只检查 API 进程是否活着。"""
    logger.info("health_check_called")

    return {
        "status": "healthy",
        "version": settings.VERSION,
        "environment": settings.ENVIRONMENT.value,
        "timestamp": datetime.now().isoformat(),
    }
