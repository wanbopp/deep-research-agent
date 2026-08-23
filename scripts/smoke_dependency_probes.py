"""Verify all application-level dependency probes against real services.

本脚本验证的是应用实际使用的驱动路径，而不是容器内部 healthcheck：

* PostgreSQL：从异步连接池借用连接并执行 ``SELECT 1``；
* Neo4j：通过异步 Driver 执行只读的 ``RETURN 1``；
* Redis：通过异步 Client 执行 ``PING``。

三个探针并发执行，并分别受到短超时保护。控制台只输出稳定状态、耗时和安全错误
代码，不输出地址、用户名、密码、连接串或底层异常文本。
"""

import asyncio
import json
import logging
import os
import selectors
from time import perf_counter
from typing import Any

from neo4j import AsyncGraphDatabase
from psycopg import AsyncConnection
from psycopg.conninfo import make_conninfo
from psycopg_pool import AsyncConnectionPool
from redis.asyncio import Redis

from app.core.config import settings
from app.infrastructure.neo4j import probe_neo4j
from app.infrastructure.postgres import probe_postgres
from app.infrastructure.probes import DependencyProbeResult
from app.infrastructure.redis import probe_redis

# 单个探针最多等待 5 秒。三个探针并发执行，因此正常情况下总耗时接近最慢的
# 单个探针，而不是三个超时之和。
PROBE_TIMEOUT_SECONDS = 5.0
TOTAL_TIMEOUT_SECONDS = 8.0
RESOURCE_CLOSE_TIMEOUT_SECONDS = 2.0

# 地址和端口可以继续使用 Settings 的 localhost 默认值；这些身份字段必须由本地
# 忽略文件显式提供，不能让 smoke 悄悄使用示例默认凭据发起真实连接。
REQUIRED_CONFIGURATION = (
    "POSTGRES_DB",
    "POSTGRES_USER",
    "POSTGRES_PASSWORD",
    "NEO4J_PASSWORD",
    "REDIS_PASSWORD",
)

# 部分驱动会在连接失败或内部重试时直接记录完整目标地址。smoke 的输出契约只允许
# 最终安全 JSON，因此禁止第三方库自行写日志；探针结果仍会保留稳定错误代码。
logging.disable(logging.CRITICAL)


def _elapsed_ms(started_at: float) -> float:
    """Return elapsed milliseconds without exposing connection information."""
    return round((perf_counter() - started_at) * 1000, 2)


def _safe_result(result: DependencyProbeResult) -> dict[str, object]:
    """Convert an internal result into a JSON-safe, redacted summary."""
    return {
        "name": result.name.value,
        "status": result.status.value,
        "latency_ms": result.latency_ms,
        "error_code": result.error_code.value if result.error_code is not None else None,
    }


def _selector_loop_factory() -> asyncio.AbstractEventLoop:
    """Create the selector event loop required by psycopg async on Windows."""
    # Windows 默认 ProactorEventLoop，但 psycopg 的异步文件描述符监听依赖
    # SelectorEventLoop。使用 asyncio.run(loop_factory=...) 只影响本 smoke，
    # 不修改操作系统或其他 Python 进程的全局事件循环策略。
    return asyncio.SelectorEventLoop(selectors.SelectSelector())


def _missing_configuration() -> tuple[str, ...]:
    """Return missing variable names without reading values into output."""
    return tuple(name for name in REQUIRED_CONFIGURATION if not os.getenv(name))


async def _run_smoke() -> dict[str, object]:
    """Create clients, run all real probes concurrently, then release resources."""
    started_at = perf_counter()

    missing_configuration = _missing_configuration()
    if missing_configuration:
        return {
            "ok": False,
            "stage": "configuration",
            "missing_variables": list(missing_configuration),
            "elapsed_ms": _elapsed_ms(started_at),
        }

    # make_conninfo 会正确转义特殊字符。连接串只保存在内存中，绝不能输出。
    postgres_conninfo = make_conninfo(
        host=settings.POSTGRES_HOST,
        port=settings.POSTGRES_PORT,
        dbname=settings.POSTGRES_DB,
        user=settings.POSTGRES_USER,
        password=settings.POSTGRES_PASSWORD,
        connect_timeout=int(PROBE_TIMEOUT_SECONDS),
    )

    # min_size=0 表示 open() 时不预先连接。第一次真实连接会发生在 SELECT 1
    # operation 内，因此连接失败也会经过 run_probe 的统一超时和错误映射。
    postgres_pool: AsyncConnectionPool[AsyncConnection[Any]] = AsyncConnectionPool(
        conninfo=postgres_conninfo,
        min_size=0,
        max_size=1,
        open=False,
        timeout=PROBE_TIMEOUT_SECONDS,
    )

    # Neo4j Driver 和 Redis Client 的构造都是惰性的；真正的网络请求分别发生在
    # RETURN 1 与 PING 中，正好落在各自探针的超时边界内。
    neo4j_driver = AsyncGraphDatabase.driver(
        settings.NEO4J_URI,
        auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD),
        connection_timeout=PROBE_TIMEOUT_SECONDS,
        connection_acquisition_timeout=PROBE_TIMEOUT_SECONDS,
        # 探针应快速反映当前状态，不能在驱动内部执行长时间事务重试。
        max_transaction_retry_time=0.0,
    )
    redis_client = Redis(
        host=settings.REDIS_HOST,
        port=settings.REDIS_PORT,
        db=settings.REDIS_DB,
        password=settings.REDIS_PASSWORD or None,
        socket_connect_timeout=PROBE_TIMEOUT_SECONDS,
        socket_timeout=PROBE_TIMEOUT_SECONDS,
        decode_responses=True,
    )

    results: tuple[DependencyProbeResult, ...]
    cleanup_ok = False

    try:
        # 打开连接池的后台管理任务，但不预热连接；资源预热属于后续 lifespan。
        await postgres_pool.open()

        # 外层总预算保护整个并发批次，内层 run_probe 仍为每个依赖单独计时和分类。
        async with asyncio.timeout(TOTAL_TIMEOUT_SECONDS):
            gathered = await asyncio.gather(
                probe_postgres(postgres_pool, timeout_seconds=PROBE_TIMEOUT_SECONDS),
                probe_neo4j(neo4j_driver, timeout_seconds=PROBE_TIMEOUT_SECONDS),
                probe_redis(redis_client, timeout_seconds=PROBE_TIMEOUT_SECONDS),
            )
        results = tuple(gathered)

    finally:
        # smoke 是这些资源的所有者，所以无论探针成功、失败还是被取消，都必须关闭。
        # return_exceptions=True 让某个 close 失败时，其他资源仍有机会完成清理。
        cleanup_results = await asyncio.gather(
            postgres_pool.close(timeout=RESOURCE_CLOSE_TIMEOUT_SECONDS),
            neo4j_driver.close(),
            redis_client.aclose(),
            return_exceptions=True,
        )
        cleanup_ok = not any(isinstance(item, BaseException) for item in cleanup_results)

    safe_results = [_safe_result(result) for result in results]
    probes_ok = all(result.is_healthy for result in results)

    return {
        "ok": probes_ok and cleanup_ok,
        "probe_count": len(results),
        "results": safe_results,
        "cleanup_ok": cleanup_ok,
        "within_total_budget": _elapsed_ms(started_at) <= TOTAL_TIMEOUT_SECONDS * 1000,
        "elapsed_ms": _elapsed_ms(started_at),
    }


def main() -> int:
    """Run the smoke and print exactly one safe JSON summary."""
    started_at = perf_counter()

    try:
        summary = asyncio.run(
            _run_smoke(),
            loop_factory=_selector_loop_factory,
        )
    except Exception as exc:
        # 这里只公开异常类型，禁止打印 str(exc)，避免驱动异常携带连接信息。
        print(
            json.dumps(
                {
                    "ok": False,
                    "stage": "dependency_probe_smoke",
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
