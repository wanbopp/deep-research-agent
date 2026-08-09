"""结构化日志上下文的隔离测试."""

import asyncio
import logging
from collections.abc import AsyncIterator
from uuid import uuid4

import pytest
from asgi_correlation_id import CorrelationIdMiddleware, correlation_id
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from app.core.middleware import RequestLoggingMiddleware
from app.core.logging import bind_context, clear_context, get_context, logger


# 真实的 HTTP 请求测试并发 HTTP 请求上下文隔离


@pytest.fixture
async def context_client(
    anyio_backend: str,
) -> AsyncIterator[AsyncClient]:
    """创建用于观察并发请求上下文测试客户端."""
    context_app = FastAPI()

    # 保持于正式应用一致的中间件添加顺序
    context_app.add_middleware(RequestLoggingMiddleware)
    context_app.add_middleware(CorrelationIdMiddleware)

    @context_app.get("/context/{label}")
    async def context_probe(
        label: str,
        delay_seconds: float = 0,
    ) -> dict[str, object]:
        """返回当前请求内部读取到的上下文字段."""
        bind_context(context_label=label)

        # 第一个请求会在这里等待，让第二个请求进入同一路由。
        await asyncio.sleep(delay_seconds)
        logger.info("context_probe_observed")
        return {
            "request_id": correlation_id.get(),
            "context_label": get_context().get("context_label"),
        }

    transport = ASGITransport(app=context_app)

    async with AsyncClient(
        transport=transport,
        base_url="http://testserver",
    ) as async_client:
        yield async_client


@pytest.mark.anyio
async def test_concurrent_requests_keep_context_isolated(
    context_client: AsyncClient,
) -> None:
    """两个重叠的 HTTP 请求应保留各自的 ID 和日志上下文."""
    first_request_id = uuid4().hex
    second_request_id = uuid4().hex

    assert first_request_id != second_request_id

    first_response, second_response = await asyncio.gather(
        context_client.get(
            "/context/first",
            params={"delay_seconds": 0.02},
            headers={"X-Request-ID": first_request_id},
        ),
        context_client.get(
            "/context/second",
            params={"delay_seconds": 0},
            headers={"X-Request-ID": second_request_id},
        ),
    )

    assert first_response.status_code == 200
    assert second_response.status_code == 200

    first_body = first_response.json()
    second_body = second_response.json()

    # 路由内部读取到的 correlation_id 必须属于当前请求。
    assert first_body["request_id"] == first_request_id
    assert second_body["request_id"] == second_request_id

    # 自定义日志上下文也不能在并发请求之间互相覆盖。
    assert first_body["context_label"] == "first"
    assert second_body["context_label"] == "second"

    # 响应头应携带同一个请求在路由内部看到的 ID。
    assert first_response.headers["X-Request-ID"] == first_request_id
    assert second_response.headers["X-Request-ID"] == second_request_id


@pytest.mark.anyio
async def test_logging_context_is_isolated_between_concurrent_tasks() -> None:
    """两个并发任务应该分别读取自己绑定的日志上下文."""

    async def observe_context(session_id: str) -> str:
        """绑定字段，让出执行权后再读取当前任务中的值."""
        bind_context(session_id=session_id)

        try:
            # 主动暂停当前任务，让另外一个任务有机会运行和绑定字段
            # 制造真正需要的 ContextVar 隔离的交错执行顺序
            await asyncio.sleep(0)

            observed_session_id = get_context()["session_id"]

            assert isinstance(observed_session_id, str)
            return observed_session_id
        finally:
            # 即使断言失败也要清理当前任务的上下文
            clear_context()

    # 确保并发任务前，父任务中没有遗留字段
    clear_context()

    first_result, second_result = await asyncio.gather(
        observe_context("session-a"),
        observe_context("session-b"),
    )

    assert first_result == "session-a"
    assert second_result == "session-b"

    # 子任务修改和清理自己的 ContextVar，不应该给当前父任务留下任何字段
    assert get_context() == {}


@pytest.mark.anyio
async def test_concurrent_requests_write_correlated_log_events(
    context_client: AsyncClient,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """并发请求日志应保留各自的 request_id 和上下文字段."""
    caplog.set_level(logging.INFO)

    first_request_id = uuid4().hex
    second_request_id = uuid4().hex

    first_response, second_response = await asyncio.gather(
        context_client.get(
            "/context/first",
            params={"delay_seconds": 0.02},
            headers={"X-Request-ID": first_request_id},
        ),
        context_client.get(
            "/context/second",
            params={"delay_seconds": 0},
            headers={"X-Request-ID": second_request_id},
        ),
    )

    assert first_response.status_code == 200
    assert second_response.status_code == 200

    observed_events: list[dict[str, object]] = []

    for record in caplog.records:
        # structlog 在进入最终 renderer 前，
        # 会把结构化事件字典放在 LogRecord.msg 中。
        if not isinstance(record.msg, dict):
            continue

        if record.msg.get("event") == "context_probe_observed":
            observed_events.append(record.msg)

    # 两个请求应该分别产生一条目标日志。
    assert len(observed_events) == 2

    first_event = next(event for event in observed_events if event.get("context_label") == "first")
    second_event = next(event for event in observed_events if event.get("context_label") == "second")

    assert first_event["request_id"] == first_request_id
    assert first_event["path"] == "/context/first"
    assert first_event["method"] == "GET"

    assert second_event["request_id"] == second_request_id
    assert second_event["path"] == "/context/second"
    assert second_event["method"] == "GET"
