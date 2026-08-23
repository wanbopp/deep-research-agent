"""Verify liveness and readiness through the real HTTP boundary.

本脚本连接一个已经运行的 FastAPI 开发服务器，同时请求：

1. ``/api/v1/health/live``，确认 API 进程仍然能够响应；
2. ``/api/v1/health/ready``，确认当前依赖状态符合指定场景。

脚本本身不创建、关闭或重启任何基础设施服务。服务启停属于独立的运维动作，
必须在执行前说明影响，并在验收后立即恢复。这样测试代码只拥有“观察”权限，
不会因为一个断言错误而意外修改基础设施状态。

示例：

    # 三项依赖全部健康。
    python scripts/smoke_health_readiness.py --expected ready

    # Redis 被短暂停止，PostgreSQL 仍健康。
    python scripts/smoke_health_readiness.py \
        --expected degraded \
        --expected-unhealthy redis

    # PostgreSQL 被短暂停止。
    python scripts/smoke_health_readiness.py \
        --expected not_ready \
        --expected-unhealthy postgres

输出只包含状态码、公开依赖状态和耗时，不包含地址、密码、连接串或异常原文。
"""

import argparse
import asyncio
import json
from collections.abc import Sequence
from time import perf_counter
from typing import Literal, cast

from httpx import AsyncClient

from app.schemas.health import LivenessResponse, ReadinessResponse

type ExpectedReadinessStatus = Literal["ready", "degraded", "not_ready"]
type ExpectedDependencyName = Literal["postgres", "neo4j", "redis"]

DEFAULT_BASE_URL = "http://127.0.0.1:8000"
HTTP_TIMEOUT_SECONDS = 5.0
EXPECTED_DEPENDENCY_ORDER: tuple[ExpectedDependencyName, ...] = (
    "postgres",
    "neo4j",
    "redis",
)
EXPECTED_REQUIRED_FLAGS = {
    "postgres": True,
    "neo4j": False,
    "redis": False,
}


def _elapsed_ms(started_at: float) -> float:
    """Return elapsed milliseconds for the safe smoke summary."""
    return round((perf_counter() - started_at) * 1000, 2)


def _build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser without reading global process arguments."""
    parser = argparse.ArgumentParser(
        description="Verify the running API liveness/readiness responses.",
    )
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help="Running API origin; defaults to the local development server.",
    )
    parser.add_argument(
        "--expected",
        required=True,
        choices=("ready", "degraded", "not_ready"),
        help="Expected aggregate readiness status.",
    )
    parser.add_argument(
        "--expected-unhealthy",
        action="append",
        default=[],
        choices=EXPECTED_DEPENDENCY_ORDER,
        help="Dependency expected to be unhealthy; repeat for multiple names.",
    )
    return parser


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the expected health scenario without accepting credentials."""
    return _build_parser().parse_args(argv)


def _scenario_is_consistent(
    expected_status: ExpectedReadinessStatus,
    expected_unhealthy: set[ExpectedDependencyName],
) -> bool:
    """Check that CLI expectations follow the application's readiness policy."""
    if expected_status == "ready":
        return not expected_unhealthy

    if expected_status == "degraded":
        # PostgreSQL is required, so its failure can never mean only degraded.
        return bool(expected_unhealthy) and "postgres" not in expected_unhealthy

    # not_ready currently means at least the required PostgreSQL probe failed.
    return "postgres" in expected_unhealthy


async def _run_smoke(
    *,
    base_url: str,
    expected_status: ExpectedReadinessStatus,
    expected_unhealthy: set[ExpectedDependencyName],
) -> dict[str, object]:
    """Request both endpoints and validate their public behavior."""
    started_at = perf_counter()

    if not _scenario_is_consistent(expected_status, expected_unhealthy):
        raise ValueError("Expected status and unhealthy dependency names are inconsistent")

    async with AsyncClient(
        base_url=base_url,
        timeout=HTTP_TIMEOUT_SECONDS,
        # 本 smoke 默认访问 localhost。忽略 HTTP(S)_PROXY，避免本地服务器
        # 未启动时由系统代理返回 HTML 错误页，掩盖真实的连接失败。
        trust_env=False,
    ) as client:
        # 两个请求都经过真实 HTTP server、middleware 和 route。
        # 顺序执行更适合教学断点：先证明进程活着，再观察流量门禁。
        live_response = await client.get("/api/v1/health/live")
        ready_response = await client.get("/api/v1/health/ready")

    # 使用生产 Pydantic schema 解析响应，避免脚本用松散字典重复猜测字段。
    live = LivenessResponse.model_validate(live_response.json())
    readiness = ReadinessResponse.model_validate(ready_response.json())

    expected_ready_code = 503 if expected_status == "not_ready" else 200
    dependency_order = tuple(item.name for item in readiness.dependencies)
    observed_unhealthy = {
        cast(ExpectedDependencyName, item.name) for item in readiness.dependencies if item.status == "unhealthy"
    }

    required_flags_match = all(item.required is EXPECTED_REQUIRED_FLAGS[item.name] for item in readiness.dependencies)
    error_codes_match = all(
        (item.error_code is None) if item.status == "healthy" else (item.error_code is not None)
        for item in readiness.dependencies
    )

    checks = {
        "live_status_code_matches": live_response.status_code == 200,
        "live_status_matches": live.status == "alive",
        "ready_status_code_matches": ready_response.status_code == expected_ready_code,
        "ready_status_matches": readiness.status == expected_status,
        "dependency_order_matches": dependency_order == EXPECTED_DEPENDENCY_ORDER,
        "required_flags_match": required_flags_match,
        "unhealthy_dependencies_match": observed_unhealthy == expected_unhealthy,
        "error_codes_match": error_codes_match,
    }

    return {
        "ok": all(checks.values()),
        "expected": expected_status,
        "live_status_code": live_response.status_code,
        "ready_status_code": ready_response.status_code,
        "readiness_status": readiness.status,
        "dependency_statuses": {item.name: item.status for item in readiness.dependencies},
        **checks,
        "elapsed_ms": _elapsed_ms(started_at),
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Run one expected readiness scenario and print a safe JSON summary."""
    args = _parse_args(argv)
    expected_status = cast(ExpectedReadinessStatus, args.expected)
    expected_unhealthy = set(cast(list[ExpectedDependencyName], args.expected_unhealthy))
    started_at = perf_counter()

    try:
        summary = asyncio.run(
            _run_smoke(
                base_url=cast(str, args.base_url).rstrip("/"),
                expected_status=expected_status,
                expected_unhealthy=expected_unhealthy,
            )
        )
    except Exception as exc:
        print(
            json.dumps(
                {
                    "ok": False,
                    "stage": "health_readiness_http",
                    "expected": expected_status,
                    "error_type": type(exc).__name__,
                    "elapsed_ms": _elapsed_ms(started_at),
                }
            )
        )
        return 1

    print(json.dumps(summary))
    return 0 if summary["ok"] is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
