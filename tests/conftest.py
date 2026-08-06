"""所有 API 测试共享的准备工作."""

from collections.abc import AsyncIterator

import pytest
from httpx import AsyncClient, ASGITransport

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
