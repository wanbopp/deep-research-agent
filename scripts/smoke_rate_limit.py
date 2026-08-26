"""使用真实 Redis 验收跨 client 限流与 FastAPI 429/503 边界.

本 smoke 不调用 Agent Graph 或模型。它验证的核心恰好是：额度耗尽或 Redis
故障时，请求会在 FastAPI dependency 阶段停止，受保护的 route body 不会执行，
因此也不可能产生真实模型费用。
"""

import asyncio
import json
import selectors
from time import perf_counter
from uuid import UUID, uuid4

from asgi_correlation_id import CorrelationIdMiddleware
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from redis.asyncio import Redis

from app.api.dependencies import (
    AgentRateLimitDependency,
    AnonymousAuthRateLimitDependency,
    get_current_user,
)
from app.core.config import settings
from app.core.exception_handlers import register_exception_handlers
from app.infrastructure.rate_limit import RedisRateLimiter
from app.schemas.auth import AuthenticatedUser
from app.services.rate_limit import RateLimitPolicies, RateLimitPolicy


WINDOW_SECONDS = 2
WINDOW_WAIT_SECONDS = 2.2
CONCURRENCY = 8
SHARED_LIMIT = 3
TOTAL_TIMEOUT_SECONDS = 12.0


def _elapsed_ms(started_at: float) -> float:
    """返回只用于安全摘要的毫秒耗时."""
    return round((perf_counter() - started_at) * 1000, 2)


def _selector_loop_factory() -> asyncio.AbstractEventLoop:
    """创建与项目其他 Windows 基础设施 smoke 一致的事件循环."""
    return asyncio.SelectorEventLoop(selectors.SelectSelector())


def _create_redis_client() -> Redis:
    """创建由本 smoke 独立拥有的真实 Redis client.

    Returns:
        使用当前 Git 忽略环境配置、并自动解码字符串响应的异步 client。
    """
    return Redis(
        host=settings.REDIS_HOST,
        port=settings.REDIS_PORT,
        db=settings.REDIS_DB,
        password=settings.REDIS_PASSWORD or None,
        socket_connect_timeout=3.0,
        socket_timeout=3.0,
        decode_responses=True,
    )


def _create_unavailable_redis_client() -> Redis:
    """创建指向本机关闭端口的 client，用真实适配器验证 503.

    Returns:
        只用于本 smoke 的失败连接。它不会替换生产代码，也不会模拟模型。
    """
    return Redis(
        host="127.0.0.1",
        port=1,
        db=0,
        socket_connect_timeout=0.2,
        socket_timeout=0.2,
        decode_responses=True,
    )


async def _delete_policy_keys(redis_client: Redis, policy_names: tuple[str, ...]) -> None:
    """删除本 smoke 随机策略名下创建的全部 key.

    Args:
        redis_client: 当前 smoke 拥有的可用真实 client。
        policy_names: 本次运行随机生成且只属于当前脚本的策略名。
    """
    for policy_name in policy_names:
        pattern = f"deep-research:rate-limit:v1:{policy_name}:*"
        keys = [key async for key in redis_client.scan_iter(match=pattern)]
        if keys:
            await redis_client.delete(*keys)


async def _exercise_real_rate_limit() -> dict[str, bool | float | int]:
    """执行跨 client 原子计数、HTTP 拒绝、窗口恢复和清理验收.

    Returns:
        只包含行为布尔值、计数和耗时的脱敏摘要。

    Raises:
        Exception: 任一真实 Redis 或 HTTP 行为不符合预期时传播；finally 仍尝试
            删除测试 key 并关闭三个由本脚本拥有的 client。
    """
    started_at = perf_counter()
    first_client = _create_redis_client()
    second_client = _create_redis_client()
    unavailable_client = _create_unavailable_redis_client()
    first_limiter = RedisRateLimiter(first_client)
    second_limiter = RedisRateLimiter(second_client)
    unavailable_limiter = RedisRateLimiter(unavailable_client)

    run_id = uuid4().hex[:12]
    shared_policy = RateLimitPolicy(
        name=f"smoke_shared_{run_id}",
        limit=SHARED_LIMIT,
        window_seconds=WINDOW_SECONDS,
    )
    auth_policy = RateLimitPolicy(
        name=f"smoke_auth_{run_id}",
        limit=1,
        window_seconds=WINDOW_SECONDS,
    )
    agent_policy = RateLimitPolicy(
        name=f"smoke_agent_{run_id}",
        limit=1,
        window_seconds=WINDOW_SECONDS,
    )
    policy_names = (
        shared_policy.name,
        auth_policy.name,
        agent_policy.name,
    )
    shared_identity = f"user:raw-sensitive-{uuid4().hex}"
    other_identity = f"user:other-sensitive-{uuid4().hex}"

    cleanup_ok = False
    try:
        async with asyncio.timeout(TOTAL_TIMEOUT_SECONDS):
            # 请求在两个独立 RedisRateLimiter/client 之间交错。若算法在 Python
            # 内存计数或使用非原子“先读后写”，高并发下允许数就可能突破 3。
            tasks = [
                asyncio.create_task(
                    (first_limiter if index % 2 == 0 else second_limiter).acquire(
                        policy=shared_policy,
                        identity=shared_identity,
                    )
                )
                for index in range(CONCURRENCY)
            ]
            decisions = await asyncio.gather(*tasks)
            allowed_count = sum(decision.allowed for decision in decisions)
            concurrent_limit_exact = allowed_count == SHARED_LIMIT

            # identity 参与摘要，因此另一个身份拥有独立窗口，不会被 shared_identity
            # 已经耗尽的额度误伤。
            other_decision = await second_limiter.acquire(
                policy=shared_policy,
                identity=other_identity,
            )
            different_identity_isolated = other_decision.allowed

            shared_pattern = f"deep-research:rate-limit:v1:{shared_policy.name}:*"
            shared_keys = [key async for key in first_client.scan_iter(match=shared_pattern)]
            raw_identity_absent_from_keys = bool(shared_keys) and all(
                shared_identity not in key and other_identity not in key for key in shared_keys
            )

            # 构造最小 FastAPI 应用，但使用正式 dependency、正式异常 handler 和
            # 真实 RedisRateLimiter。route body 的计数器代表昂贵业务/模型调用。
            app = FastAPI()
            register_exception_handlers(app)
            app.add_middleware(CorrelationIdMiddleware)
            app.state.rate_limiter = first_limiter
            app.state.rate_limit_policies = RateLimitPolicies(
                auth=auth_policy,
                agent=agent_policy,
            )
            protected_agent_calls = 0
            protected_auth_calls = 0
            fixed_user = AuthenticatedUser(
                user_id=UUID("77777777-7777-4777-8777-777777777777"),
                email="rate-limit-smoke@example.com",
            )

            async def override_current_user() -> AuthenticatedUser:
                """提供固定的已认证身份，只隔离 JWT/数据库而不替换限流器."""
                return fixed_user

            app.dependency_overrides[get_current_user] = override_current_user

            @app.post("/agent-probe")
            async def agent_probe(
                _rate_limit: AgentRateLimitDependency,
            ) -> dict[str, bool]:
                """代表只有 dependency 允许后才会执行的高成本操作."""
                nonlocal protected_agent_calls
                protected_agent_calls += 1
                return {"entered": True}

            @app.post("/auth-probe")
            async def auth_probe(
                _rate_limit: AnonymousAuthRateLimitDependency,
            ) -> dict[str, bool]:
                """代表只有客户端 IP 尚有额度时才会执行的认证操作."""
                nonlocal protected_auth_calls
                protected_auth_calls += 1
                return {"entered": True}

            transport = ASGITransport(
                app=app,
                raise_app_exceptions=False,
                client=("198.51.100.10", 43210),
            )
            async with AsyncClient(transport=transport, base_url="http://testserver") as client:
                first_agent_response = await client.post("/agent-probe")
                rejected_agent_response = await client.post("/agent-probe")
                first_auth_response = await client.post("/auth-probe")
                rejected_auth_response = await client.post("/auth-probe")

                agent_429_body = rejected_agent_response.json()
                auth_429_body = rejected_auth_response.json()
                stable_429_contract = (
                    first_agent_response.status_code == 200
                    and rejected_agent_response.status_code == 429
                    and agent_429_body["error"]["code"] == "RATE_LIMIT_EXCEEDED"
                    and int(rejected_agent_response.headers["retry-after"]) > 0
                    and first_auth_response.status_code == 200
                    and rejected_auth_response.status_code == 429
                    and auth_429_body["error"]["code"] == "RATE_LIMIT_EXCEEDED"
                    and int(rejected_auth_response.headers["retry-after"]) > 0
                )
                rejected_requests_skipped_operation = protected_agent_calls == 1 and protected_auth_calls == 1

                # 同一个正式 dependency 改为借用故障 client。Redis 无法判断额度时
                # 必须返回 503，且不能把失败伪装成 429 或进入受保护操作。
                app.state.rate_limiter = unavailable_limiter
                unavailable_response = await client.post("/agent-probe")
                unavailable_body = unavailable_response.json()
                stable_503_contract = (
                    unavailable_response.status_code == 503
                    and unavailable_body["error"]["code"] == "RATE_LIMIT_UNAVAILABLE"
                    and protected_agent_calls == 1
                )

            # Redis 服务端 TTL 到期后，原身份应自动获得新窗口。真实等待只用于
            # 验证服务端过期，不使用进程内手动时钟代替 Redis 行为。
            await asyncio.sleep(WINDOW_WAIT_SECONDS)
            recovered_decision = await first_limiter.acquire(
                policy=shared_policy,
                identity=shared_identity,
            )
            window_recovers_after_expiry = recovered_decision.allowed

        checks: dict[str, bool | float | int] = {
            "allowed_count": allowed_count,
            "concurrent_limit_exact": concurrent_limit_exact,
            "different_identity_isolated": different_identity_isolated,
            "raw_identity_absent_from_keys": raw_identity_absent_from_keys,
            "stable_429_contract": stable_429_contract,
            "rejected_requests_skipped_operation": rejected_requests_skipped_operation,
            "stable_503_contract": stable_503_contract,
            "window_recovers_after_expiry": window_recovers_after_expiry,
            "within_total_budget": _elapsed_ms(started_at) <= TOTAL_TIMEOUT_SECONDS * 1000,
            "elapsed_ms": _elapsed_ms(started_at),
        }
    finally:
        cleanup_errors: list[Exception] = []
        try:
            await _delete_policy_keys(first_client, policy_names)
        except Exception as error:
            cleanup_errors.append(error)

        for client in (first_client, second_client, unavailable_client):
            try:
                await client.aclose()
            except Exception as error:
                cleanup_errors.append(error)

        cleanup_ok = not cleanup_errors

    if not cleanup_ok:
        raise RuntimeError("rate limit smoke cleanup failed")

    return {**checks, "cleanup_ok": cleanup_ok}


def _run_smoke() -> dict[str, object]:
    """运行异步验收，并把所有布尔检查汇总为 ok."""
    checks = asyncio.run(
        _exercise_real_rate_limit(),
        loop_factory=_selector_loop_factory,
    )
    boolean_checks = tuple(value for value in checks.values() if isinstance(value, bool))
    return {
        "ok": bool(boolean_checks) and all(boolean_checks),
        **checks,
    }


def main() -> int:
    """打印一行脱敏 JSON，并返回适合终端或 CI 的退出码."""
    started_at = perf_counter()
    try:
        summary = _run_smoke()
    except Exception as error:
        print(
            json.dumps(
                {
                    "ok": False,
                    "error_type": type(error).__name__,
                    "elapsed_ms": _elapsed_ms(started_at),
                }
            )
        )
        return 1

    print(json.dumps(summary))
    return 0 if summary["ok"] is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
