"""Redis application-level dependency probe."""

from redis.asyncio import Redis
from redis.exceptions import (
    AuthenticationError,
    ConnectionError as RedisConnectionError,
    TimeoutError as RedisTimeoutError,
)

from app.infrastructure.probes import (
    DependencyName,
    DependencyProbeResult,
    ProbeErrorCode,
    run_probe,
)


def classify_redis_error(exc: Exception) -> ProbeErrorCode:
    """把 Redis 驱动异常映射为稳定且可公开的错误代码."""
    # AuthenticationError 继承自 Redis ConnectionError，必须先判断。
    if isinstance(exc, AuthenticationError):
        return ProbeErrorCode.AUTHENTICATION

    if isinstance(exc, RedisTimeoutError):
        return ProbeErrorCode.TIMEOUT

    if isinstance(exc, RedisConnectionError):
        return ProbeErrorCode.CONNECTION

    return ProbeErrorCode.UNKNOWN


async def probe_redis(
    client: Redis,
    *,
    timeout_seconds: float = 1.0,
) -> DependencyProbeResult:
    """使用已有异步 Redis client 执行最小 ``PING`` 探针."""

    async def operation() -> None:
        # client 由 lifespan 创建并复用；探针既不创建也不关闭它。
        pong = await client.ping()
        if pong is not True:
            # PING 正常应返回 True。异常响应不包含敏感信息，可以安全分类。
            raise RedisConnectionError("Redis PING returned an unexpected response")

    return await run_probe(
        name=DependencyName.REDIS,
        operation=operation,
        timeout_seconds=timeout_seconds,
        classify_error=classify_redis_error,
    )
