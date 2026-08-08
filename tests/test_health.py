"""健康检查接口的最小契约测试."""

from datetime import datetime

import pytest
from httpx import AsyncClient

from app.core.config import settings


# pytest 默认会寻找符合约定的名字：文件名：test_*.py 函数名：test_*


@pytest.mark.anyio  # 告诉 pytest 这是AnyIO异步测试
async def test_health_check_return_200(client: AsyncClient) -> None:
    """健康检查返回 HTTP 200 和 health 状态.

    pytest 会根据参数名client，寻找同名 fixture，并把fixture的值传进来.
    """
    # 模拟客户端发送 Get api/v1/health 请求
    response = await client.get("/api/v1/health")

    # 第一层契约 HTTP 状态码必须是200
    assert response.status_code == 200

    # 把JSON响应转换为Python字典，便于检查其中字段
    body = response.json()

    # 业务层：固定字段应与当前应用配置一致
    # 第二层契约：业务状态必须明确表示服务健康。
    assert body["status"] == "healthy"
    assert body["version"] == settings.VERSION
    assert body["environment"] == settings.ENVIRONMENT.value

    # 格式层：不比较当前时间只验证它是合法字符串
    parsed_time = datetime.fromisoformat(body["timestamp"])
    assert isinstance(parsed_time, datetime)

    # 可追踪性：成功响应也必须带上 request_id
    request_id = response.headers.get("X-Request-ID")
    assert request_id is not None
    assert request_id != ""


@pytest.mark.anyio
async def test_openapi_documents_health_response(client: AsyncClient) -> None:
    """OpenAPI 应使用 HealthResponse 描述健康检查的成功响应."""
    response = await client.get("/api/v1/openapi.json")

    assert response.status_code == 200

    openapi = response.json()

    # 按照 OpenAPI 的层级找到：
    # health 路径 -> GET 操作 -> 200 响应 -> JSON body schema。
    response_schema = openapi["paths"]["/api/v1/health"]["get"]["responses"]["200"]["content"]["application/json"][
        "schema"
    ]

    # $ref 表示这里不重复展开模型，而是引用 components 中的定义。
    assert response_schema["$ref"] == "#/components/schemas/HealthResponse"

    health_schema = openapi["components"]["schemas"]["HealthResponse"]

    # 四个字段都没有默认值，因此都应该是必填字段。
    assert set(health_schema["required"]) == {
        "status",
        "version",
        "environment",
        "timestamp",
    }
