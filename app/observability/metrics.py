"""低基数 Prometheus 指标运行时.

业务代码只通过本模块提供的方法记录指标，不直接持有 Prometheus collector。
这样 API、Worker 和测试可以各自拥有独立 Registry，避免多进程或重复导入时
把不同生命周期的指标错误注册到同一个全局 Registry。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, generate_latest


_KNOWN_HTTP_METHODS = frozenset({"GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"})


def _http_method(value: str) -> str:
    """把任意方法压缩到有限标签集合，防止恶意方法名制造高基数."""
    normalized = value.upper()
    return normalized if normalized in _KNOWN_HTTP_METHODS else "OTHER"


def _status_class(status_code: int) -> str:
    """返回稳定的 HTTP 状态码类别，不把每个状态码都变成标签值."""
    if 100 <= status_code <= 599:
        return f"{status_code // 100}xx"
    return "unknown"


@dataclass(frozen=True, slots=True)
class MetricsSnapshot:
    """测试和诊断使用的 Prometheus 文本快照."""

    content: bytes
    content_type: str = "text/plain; version=0.0.4; charset=utf-8"


class MetricsRuntime:
    """拥有一个进程内 Registry 和全部受控指标定义.

    标签只能使用方法、路由模板、状态类别、模型别名、操作、节点或检索策略等
    有限集合。用户、请求、任务、会话、原始 URL、query、prompt 和文档正文只能
    进入经过脱敏的日志或 trace 上下文，严禁作为 metric label。
    """

    def __init__(self, registry: CollectorRegistry | None = None) -> None:
        """在指定 Registry 中创建指标；测试传入新 Registry 可完全隔离状态."""
        self.registry = registry or CollectorRegistry(auto_describe=True)
        self.http_requests = Counter(
            "http_requests_total",
            "按路由模板统计的 HTTP 请求数。",
            ("method", "route", "status_class"),
            registry=self.registry,
        )
        self.http_duration = Histogram(
            "http_request_duration_seconds",
            "按路由模板统计的 HTTP 请求耗时。",
            ("method", "route"),
            buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
            registry=self.registry,
        )
        self.llm_calls = Counter(
            "llm_calls_total",
            "逻辑 LLM 调用结果；一次业务调用只记录一次。",
            ("operation", "outcome"),
            registry=self.registry,
        )
        self.llm_call_duration = Histogram(
            "llm_call_duration_seconds",
            "逻辑 LLM 调用耗时。",
            ("operation",),
            buckets=(0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0),
            registry=self.registry,
        )
        self.llm_attempts = Counter(
            "llm_attempts_total",
            "包含 retry/fallback 的 Provider 实际尝试次数。",
            ("model_alias", "outcome"),
            registry=self.registry,
        )
        self.llm_tokens = Counter(
            "llm_tokens_total",
            "Provider 返回的 token 数；缺失 usage 时不估算。",
            ("model_alias", "direction"),
            registry=self.registry,
        )
        self.retrieval_duration = Histogram(
            "retrieval_duration_seconds",
            "按有限策略统计的检索耗时。",
            ("strategy", "outcome"),
            buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0),
            registry=self.registry,
        )
        self.retrieval_candidates = Histogram(
            "retrieval_candidates",
            "一次检索返回的候选数量。",
            ("strategy",),
            buckets=(0, 1, 2, 5, 10, 20, 40, 80),
            registry=self.registry,
        )
        self.research_tasks = Counter(
            "research_tasks_total",
            "持久研究任务生命周期结果。",
            ("outcome",),
            registry=self.registry,
        )
        self.research_worker_inflight = Gauge(
            "research_worker_inflight",
            "当前进程正在执行的研究任务数。",
            registry=self.registry,
        )

    def observe_http(self, *, method: str, route: str, status_code: int, duration_seconds: float) -> None:
        """记录一次 HTTP 请求，route 必须是框架解析后的模板而不是原始 URL."""
        safe_method = _http_method(method)
        safe_route = route if route.startswith("/") else "unmatched"
        self.http_requests.labels(safe_method, safe_route, _status_class(status_code)).inc()
        self.http_duration.labels(safe_method, safe_route).observe(max(0.0, duration_seconds))

    def observe_llm_tokens(self, *, model_alias: str, input_tokens: int | None, output_tokens: int | None) -> None:
        """只记录 Provider 明确返回的非负 token，不从文本长度推测."""
        for direction, value in (("input", input_tokens), ("output", output_tokens)):
            if value is not None and value >= 0:
                self.llm_tokens.labels(model_alias, direction).inc(value)

    def observe_llm_call(self, *, operation: str, outcome: str, duration_seconds: float) -> None:
        """记录一次逻辑调用；operation 和 outcome 必须来自代码内固定枚举."""
        self.llm_calls.labels(operation, outcome).inc()
        self.llm_call_duration.labels(operation).observe(max(0.0, duration_seconds))

    def observe_llm_attempt(self, *, model_alias: str, outcome: str) -> None:
        """记录一次真实 Provider attempt，用于区分重试和逻辑调用量."""
        self.llm_attempts.labels(model_alias, outcome).inc()

    def observe_retrieval(
        self,
        *,
        strategy: str,
        outcome: str,
        duration_seconds: float,
        candidate_count: int | None,
    ) -> None:
        """记录一次检索；query、URL 和正文永远不进入标签."""
        self.retrieval_duration.labels(strategy, outcome).observe(max(0.0, duration_seconds))
        if candidate_count is not None and candidate_count >= 0:
            self.retrieval_candidates.labels(strategy).observe(candidate_count)

    def observe_research_task(self, *, outcome: str) -> None:
        """记录一个持久研究任务的最终处理结果."""
        self.research_tasks.labels(outcome).inc()

    def render(self) -> MetricsSnapshot:
        """生成 Prometheus exposition 文本的不可变快照."""
        return MetricsSnapshot(content=generate_latest(self.registry))

    @staticmethod
    def finite_label_values(values: Sequence[str], *, fallback: str = "other") -> tuple[str, ...]:
        """把空标签替换为固定值；调用方仍需从受控枚举提供非空值."""
        return tuple(value if value else fallback for value in values)


# API 进程使用模块级实例；独立 Worker 会在自己的进程里得到独立 Registry。
metrics = MetricsRuntime()


__all__ = ["MetricsRuntime", "MetricsSnapshot", "metrics"]
