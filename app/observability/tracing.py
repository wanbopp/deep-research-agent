"""内容安全、失败开放的 tracing 运行时.

Trace 只接收代码定义的结构化字段；prompt、topic、证据、工具输入输出和用户身份
不在模型中，因此调用方无法误把正文上传到外部观测系统。
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager
from contextvars import ContextVar, Token
from dataclasses import asdict, dataclass, replace
from types import TracebackType
from typing import Protocol
from uuid import UUID

from app.core.logging import logger


_ALLOWED_SPAN_NAMES = frozenset({"research.run", "research.node", "retrieval", "llm"})


@dataclass(frozen=True, slots=True)
class ObservationContext:
    """允许进入外部 trace 的有限关联字段，不包含用户内容和身份."""

    research_id: UUID | None = None
    run_id: UUID | None = None
    attempt_no: int | None = None
    node_name: str | None = None
    model_alias: str | None = None
    prompt_name: str | None = None
    prompt_version: str | None = None
    prompt_hash: str | None = None
    retrieval_strategy: str | None = None

    def metadata(self) -> dict[str, str | int]:
        """转换成可序列化 metadata，并丢弃所有空值."""
        result: dict[str, str | int] = {}
        for key, value in asdict(self).items():
            if value is not None:
                result[key] = str(value) if isinstance(value, UUID) else value
        return result


class TraceSink(Protocol):
    """外部追踪适配器的最小同步上下文接口."""

    def span(self, name: str, metadata: dict[str, str | int]) -> AbstractContextManager[object]:
        """创建一个 span 上下文."""
        ...

    def close(self) -> None:
        """尽力刷新并关闭适配器."""
        ...


class _LangfuseClient(Protocol):
    """Langfuse SDK 在本适配器中使用的窄接口."""

    def start_as_current_observation(self, **kwargs: object) -> AbstractContextManager[object]:
        """创建 SDK observation 上下文."""
        ...


class NoopTraceSink:
    """追踪关闭时的零副作用实现."""

    @contextmanager
    def span(self, name: str, metadata: dict[str, str | int]) -> Iterator[None]:
        """接受同一接口但不发送任何数据."""
        del name, metadata
        yield

    def close(self) -> None:
        """Noop 没有资源需要关闭."""


class LangfuseTraceSink:
    """只把已脱敏 metadata 交给 Langfuse v3/v4 SDK."""

    def __init__(self, client: _LangfuseClient) -> None:
        """保存惰性 SDK 客户端；凭据永远不进入日志或 metadata."""
        self._client = client

    def span(self, name: str, metadata: dict[str, str | int]) -> AbstractContextManager[object]:
        """创建当前 observation；input/output 故意保持为空."""
        return self._client.start_as_current_observation(name=name, as_type="span", metadata=metadata)

    def close(self) -> None:
        """SDK 版本可能只提供 flush；能力不存在时安全跳过."""
        flush = getattr(self._client, "flush", None)
        if callable(flush):
            flush()


class _SafeSpan(AbstractContextManager[object]):
    """确保适配器 enter/exit 失败都不会改变业务结果."""

    def __init__(self, sink: TraceSink, name: str, metadata: dict[str, str | int]) -> None:
        self._sink = sink
        self._name = name
        self._metadata = metadata
        self._delegate: AbstractContextManager[object] | None = None

    def __enter__(self) -> object:
        try:
            self._delegate = self._sink.span(self._name, self._metadata)
            return self._delegate.__enter__()
        except Exception as error:
            self._delegate = None
            logger.warning("trace_sink_failed", phase="enter", error_type=type(error).__name__)
            return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        if self._delegate is None:
            return False
        try:
            self._delegate.__exit__(exc_type, exc_value, traceback)
            # 外部 SDK 不能通过 ContextManager 的 True 返回值吞掉业务异常。
            return False
        except Exception as error:
            logger.warning("trace_sink_failed", phase="exit", error_type=type(error).__name__)
            return False


class SafeTracingRuntime:
    """通过 ContextVar 传播研究关联信息，并把适配器故障与业务隔离."""

    def __init__(self, sink: TraceSink | None = None) -> None:
        """创建默认 Noop 或测试注入的独立追踪运行时."""
        self._sink: TraceSink = sink or NoopTraceSink()
        self._context: ContextVar[ObservationContext | None] = ContextVar(
            "observation_context",
            default=None,
        )

    def _current_context(self) -> ObservationContext:
        """惰性返回不可变空上下文，避免 ContextVar 共享可变默认值."""
        return self._context.get() or ObservationContext()

    def configure(self, sink: TraceSink) -> None:
        """在进程 startup 阶段替换 sink；请求处理中不得调用."""
        previous = self._sink
        self._sink = sink
        try:
            previous.close()
        except Exception as error:
            logger.warning("trace_sink_failed", phase="close", error_type=type(error).__name__)

    @contextmanager
    def bind(self, **changes: object) -> Iterator[ObservationContext]:
        """在当前异步调用链增量绑定有限字段，退出后自动恢复."""
        allowed = ObservationContext.__dataclass_fields__
        unexpected = set(changes) - set(allowed)
        if unexpected:
            raise ValueError("unsupported observation context field")
        token: Token[ObservationContext | None] = self._context.set(replace(self._current_context(), **changes))
        try:
            yield self._current_context()
        finally:
            self._context.reset(token)

    def span(self, name: str, **changes: object) -> AbstractContextManager[object]:
        """只允许有限 span 名称，并合并当前上下文与本次安全字段."""
        if name not in _ALLOWED_SPAN_NAMES:
            raise ValueError("unsupported trace span name")
        context = replace(self._current_context(), **changes)
        return _SafeSpan(self._sink, name, context.metadata())

    def close(self) -> None:
        """尽力刷新 sink，失败只记录异常类型."""
        try:
            self._sink.close()
        except Exception as error:
            logger.warning("trace_sink_failed", phase="close", error_type=type(error).__name__)


def build_trace_sink(config: object) -> TraceSink:
    """根据 Settings 构造 sink；关闭或缺凭据时明确回退 Noop."""
    enabled = bool(getattr(config, "LANGFUSE_TRACING_ENABLED", False))
    public_key = str(getattr(config, "LANGFUSE_PUBLIC_KEY", ""))
    secret_key = str(getattr(config, "LANGFUSE_SECRET_KEY", ""))
    if not enabled or not public_key or not secret_key:
        return NoopTraceSink()
    try:
        from langfuse import Langfuse

        client = Langfuse(
            public_key=public_key,
            secret_key=secret_key,
            host=str(getattr(config, "LANGFUSE_HOST", "https://cloud.langfuse.com")),
            tracing_enabled=True,
        )
        return LangfuseTraceSink(client)  # type: ignore[arg-type]
    except Exception as error:
        logger.warning("trace_sink_initialization_failed", error_type=type(error).__name__)
        return NoopTraceSink()


tracing = SafeTracingRuntime()


__all__ = [
    "LangfuseTraceSink",
    "NoopTraceSink",
    "ObservationContext",
    "SafeTracingRuntime",
    "TraceSink",
    "build_trace_sink",
    "tracing",
]
