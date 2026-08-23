"""PostgreSQL application-level dependency probe."""

from typing import Any

from psycopg import AsyncConnection, OperationalError
from psycopg.errors import InvalidPassword
from psycopg_pool import AsyncConnectionPool, PoolClosed, PoolTimeout

from app.infrastructure.probes import (
    DependencyName,
    DependencyProbeResult,
    ProbeErrorCode,
    run_probe,
)


def classify_postgres_error(exc: Exception) -> ProbeErrorCode:
    """把 psycopg/pool 异常映射为不含连接细节的稳定错误代码."""
    # InvalidPassword 同时也是 OperationalError，因此认证异常必须先判断。
    if isinstance(exc, InvalidPassword):
        return ProbeErrorCode.AUTHENTICATION

    # 获取不到连接、连接池已关闭以及网络/数据库故障都属于连接类失败。
    if isinstance(exc, (PoolTimeout, PoolClosed, OperationalError)):
        return ProbeErrorCode.CONNECTION

    return ProbeErrorCode.UNKNOWN


async def probe_postgres(
    pool: AsyncConnectionPool[AsyncConnection[Any]],
    *,
    timeout_seconds: float = 1.0,
) -> DependencyProbeResult:
    """使用已有连接池执行最小 ``SELECT 1`` 探针."""

    async def operation() -> None:
        # pool 由 lifespan 创建并拥有；探针只临时借用一条连接。
        # 离开上下文后连接会归还连接池，而不是关闭整个连接池。
        async with pool.connection() as connection:
            await connection.execute("SELECT 1")

    return await run_probe(
        name=DependencyName.POSTGRES,
        operation=operation,
        timeout_seconds=timeout_seconds,
        classify_error=classify_postgres_error,
    )
