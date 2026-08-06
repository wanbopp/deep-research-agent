"""统一错误响应的API 约束测试."""

import pytest
from httpx import AsyncClient


@pytest.mark.anyio
async def test_unknown_route_returns_404_contract(client: AsyncClient) -> None:
    """不存在的路由应返回可追踪的统一 404 响应."""
    response = await client.get("/api/v1/route-that-does-not-exist")

    # HTTP 层：保留404，而不是错误的统一成 200 或 500
    assert response.status_code == 404

    body = response.json()

    error = body["error"]

    # 错误协议：客户端使用稳定的 code 判断错误类别。
    assert error["code"] == "HTTP_ERROR"
    assert error["message"] == "Not Found"

    # 404 没有字段级补充信息，应该不应该输出 details：null
    # details 专门用于表达字段级错误
    assert "details" not in error

    # header 便于网关和通用客户端读取，body 便于业务客户端展示和上报。
    request_id = response.headers.get("X-Request-ID")
    assert request_id is not None
    assert request_id != ""
    assert body["request_id"] == request_id


@pytest.mark.anyio
async def test_invalid_query_returns_422_contract(
    error_client: AsyncClient,
) -> None:
    """非法 query 参数应返回安全、可定位的统一 422 响应."""
    response = await error_client.get(
        "/validation",
        params={"limit": "not-an-integer"},
    )

    assert response.status_code == 422

    body = response.json()
    error = body["error"]

    assert error["code"] == "VALIDATION_ERROR"
    assert error["message"] == "Request validation failed"

    details = error["details"]
    assert len(details) == 1

    detail = details[0]
    assert detail["field"] == "query -> limit"
    assert detail["type"] == "int_parsing"
    assert isinstance(detail["message"], str)
    assert detail["message"] != ""

    # handler 只公开 field/message/type，不回显用户原始输入。
    assert "input" not in detail

    request_id = response.headers.get("X-Request-ID")
    assert request_id is not None
    assert request_id != ""
    assert body["request_id"] == request_id


@pytest.mark.anyio
async def test_unhandled_exception_returns_safe_500_contract(
    error_client: AsyncClient,
) -> None:
    """未处理异常应返回安全、可追踪的统一 500 响应."""
    response = await error_client.get("/unhandled")

    assert response.status_code == 500

    body = response.json()
    error = body["error"]

    assert error["code"] == "INTERNAL_SERVER_ERROR"
    assert error["message"] == "Internal server error"
    assert "details" not in error

    # 内部异常文本只能进入服务端日志，不能泄露给调用方。
    assert "sensitive internal detail" not in response.text

    request_id = response.headers.get("X-Request-ID")
    assert request_id is not None
    assert request_id != ""
    assert body["request_id"] == request_id
