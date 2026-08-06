"""所有 API 测试共享的准备工作."""

from collections.abc import AsyncIterator

import pytest
from asgi_correlation_id import CorrelationIdMiddleware
from fastapi import FastAPI
from httpx import AsyncClient, ASGITransport

from app.core.exception_handlers import register_exception_handlers
from app.main import app


@pytest.fixture
def anyio_backend() -> str:
    """指定 AnyIO 只使用 Python 自带的 asyncio 后端."""
    return "asyncio"


@pytest.fixture  # 声明这个函数不是测试函数，而是测试依赖提供者
async def client(anyio_backend: str) -> AsyncIterator[AsyncClient]:
    """为每个需要他的测试创建一个 FastAPI 测试客户端."""
    # ASGITransport 把 HTTPX 请求直接交给 FastAPI 不使用真实端口
    transport = ASGITransport(app=app)

    # base_url 只用于补全相对 URL 不会真的访问 test server
    async with AsyncClient(
        transport=transport,
        base_url="http://testserver",
    ) as async_client:
        # pytest 把这对象注入测试；测试结束后回到这里关闭客户端
        yield async_client


@pytest.fixture
async def error_client(
    anyio_backend: str,
) -> AsyncIterator[AsyncClient]:
    """创建 只在测试进程中存在的异常场景客户端."""
    error_app = FastAPI()

    # 复用正式异常 handler 确保测试的不是另一套错误协议
    register_exception_handlers(error_app)

    # handler 从 correlation_id ContextVar 读取 ID；中间件同时写响应头。
    error_app.add_middleware(CorrelationIdMiddleware)

    @error_app.get("/validation")
    async def validation_probe(limit: int) -> dict[str, int]:
        """仅用于测试 FastAPI 的 query 参数校验."""
        return {"limit": limit}

    @error_app.get("/unhandled")
    async def unhandled_probe() -> None:
        """仅用于测试未处理异常的安全 500 响应."""
        raise RuntimeError("sensitive internal detail")

    transport = ASGITransport(
        app=error_app, raise_app_exceptions=False
    )  # 使用新的 FastAPI创建的,pytest 是否直接收到Starlette 抛出的异常
    async with AsyncClient(
        transport=transport,
        base_url="http://testserver",
    ) as async_client:
        yield async_client
