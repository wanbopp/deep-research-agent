"""Neo4j application-level dependency probe."""

from neo4j import READ_ACCESS, AsyncDriver
from neo4j.exceptions import (
    AuthError,
    ConnectionAcquisitionTimeoutError,
    ServiceUnavailable,
    SessionExpired,
)

from app.infrastructure.probes import (
    DependencyName,
    DependencyProbeResult,
    ProbeErrorCode,
    run_probe,
)


def classify_neo4j_error(exc: Exception) -> ProbeErrorCode:
    """把 Neo4j 驱动异常映射为稳定且可公开的错误代码."""
    if isinstance(exc, AuthError):
        return ProbeErrorCode.AUTHENTICATION

    if isinstance(exc, ConnectionAcquisitionTimeoutError):
        return ProbeErrorCode.TIMEOUT

    if isinstance(exc, (ServiceUnavailable, SessionExpired)):
        return ProbeErrorCode.CONNECTION

    return ProbeErrorCode.UNKNOWN


async def probe_neo4j(
    driver: AsyncDriver,
    *,
    timeout_seconds: float = 1.0,
) -> DependencyProbeResult:
    """使用已有异步 driver 执行最小 ``RETURN 1`` 探针."""

    async def operation() -> None:
        # execute_query 使用托管事务并可能自动重试。健康探针只需要一次即时快照，
        # 因此显式借用 session，执行单次 auto-commit query，并消费结果。
        # session 由此上下文关闭；共享 driver 仍由 lifespan 拥有和关闭。
        async with driver.session(default_access_mode=READ_ACCESS) as session:
            result = await session.run("RETURN 1 AS probe")
            await result.consume()

    return await run_probe(
        name=DependencyName.NEO4J,
        operation=operation,
        timeout_seconds=timeout_seconds,
        classify_error=classify_neo4j_error,
    )
