"""Application-level dependency probe contracts and execution helpers."""

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import StrEnum
from time import perf_counter


class DependencyName(StrEnum):
    """应用当前能够探测的基础设施依赖."""

    POSTGRES = "postgres"
    NEO4J = "neo4j"
    REDIS = "redis"


class ProbeStatus(StrEnum):
    """单个依赖探针的稳定状态."""

    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"


class ProbeErrorCode(StrEnum):
    """允许向应用其他层暴露的安全错误分类."""

    TIMEOUT = "timeout"
    AUTHENTICATION = "authentication_error"
    CONNECTION = "connection_error"
    UNKNOWN = "unknown_error"


@dataclass(frozen=True, slots=True)
class DependencyProbeResult:
    """一次依赖探测的统一结果，不包含底层异常原文."""

    name: DependencyName
    status: ProbeStatus
    latency_ms: float
    error_code: ProbeErrorCode | None = None

    @property
    def is_healthy(self) -> bool:
        """返回依赖是否通过了本次探测."""
        return self.status is ProbeStatus.HEALTHY


# 具体探针会把 SELECT 1、RETURN 1、PING 等命令，
# 包装成一个无参数异步函数交给 run_probe。
#
# run_probe 不关心命令属于哪种数据库，只负责公共控制流程。
type ProbeOperation = Callable[[], Awaitable[None]]

# 不同驱动拥有不同的异常类。
# 分类器负责把驱动异常转换成稳定、安全的错误代码。
type ErrorClassifier = Callable[[Exception], ProbeErrorCode]


def _elapsed_ms(started_at: float) -> float:
    """计算从 started_at 到当前时刻的单调耗时，单位为毫秒."""
    return round((perf_counter() - started_at) * 1000, 2)


async def run_probe(
    *,
    name: DependencyName,
    operation: ProbeOperation,
    timeout_seconds: float,
    classify_error: ErrorClassifier,
) -> DependencyProbeResult:
    """在统一的超时和错误边界内执行一个依赖探针.

    operation 由具体依赖适配器提供；本函数统一负责：

    1. 验证超 时配置；
    2. 记录执行耗时；
    3. 限制探针执行时间；
    4. 将结果转换为 DependencyProbeResult；
    5. 避免原始异常内容进入返回结果。

    本函数只捕获 Exception，不捕获 BaseException。
    因此任务取消仍会向上传播，不会被误报为依赖故障。
    """
    if timeout_seconds <= 0:
        # 非法超时是程序配置错误，不是依赖服务故障。
        raise ValueError("timeout_seconds must be greater than zero")

    started_at = perf_counter()

    try:
        # 超时后，asyncio.timeout 会在代码块外抛出 TimeoutError。
        async with asyncio.timeout(timeout_seconds):
            await operation()

    except TimeoutError:
        return DependencyProbeResult(
            name=name,
            status=ProbeStatus.UNHEALTHY,
            latency_ms=_elapsed_ms(started_at),
            error_code=ProbeErrorCode.TIMEOUT,
        )

    except Exception as exc:
        # 只保留分类器返回的安全代码。
        # 不调用 str(exc)，也不把异常对象保存在结果中。
        return DependencyProbeResult(
            name=name,
            status=ProbeStatus.UNHEALTHY,
            latency_ms=_elapsed_ms(started_at),
            error_code=classify_error(exc),
        )

    return DependencyProbeResult(
        name=name,
        status=ProbeStatus.HEALTHY,
        latency_ms=_elapsed_ms(started_at),
    )
