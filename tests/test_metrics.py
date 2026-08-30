"""可观测性指标的低基数和隔离性测试."""

from prometheus_client import CollectorRegistry

from app.observability.metrics import MetricsRuntime


def test_metrics_runtime_uses_route_template_and_status_class() -> None:
    """动态业务 ID 不应出现在 HTTP 指标，状态码也只保留类别."""
    runtime = MetricsRuntime(CollectorRegistry())

    runtime.observe_http(
        method="GET",
        route="/api/v1/research/{research_id}",
        status_code=200,
        duration_seconds=0.125,
    )

    content = runtime.render().content.decode()
    assert 'method="GET",route="/api/v1/research/{research_id}",status_class="2xx"' in content
    assert "research_id" in content  # 模板字段名允许出现。
    assert "00000000-0000-0000-0000-000000000001" not in content


def test_metrics_runtime_normalizes_unknown_http_method_and_route() -> None:
    """非标准方法和未匹配路径使用固定兜底值，不能扩大标签集合."""
    runtime = MetricsRuntime(CollectorRegistry())

    runtime.observe_http(method="CUSTOM-USER-VALUE", route="", status_code=799, duration_seconds=-1)

    content = runtime.render().content.decode()
    assert 'method="OTHER",route="unmatched",status_class="unknown"' in content


def test_metrics_runtime_does_not_estimate_missing_token_usage() -> None:
    """缺失 usage 时保持没有样本，避免把字符数误当 token 数."""
    runtime = MetricsRuntime(CollectorRegistry())

    runtime.observe_llm_tokens(model_alias="primary", input_tokens=None, output_tokens=None)

    content = runtime.render().content.decode()
    assert "llm_tokens_total{" not in content


def test_metrics_runtime_separates_logical_calls_from_provider_attempts() -> None:
    """一次逻辑调用可以包含多个 attempt，两类指标不能混为一个计数器."""
    runtime = MetricsRuntime(CollectorRegistry())

    runtime.observe_llm_attempt(model_alias="primary", outcome="error")
    runtime.observe_llm_attempt(model_alias="fallback", outcome="success")
    runtime.observe_llm_call(operation="structured", outcome="success", duration_seconds=0.5)
    runtime.observe_llm_tokens(model_alias="fallback", input_tokens=12, output_tokens=7)

    content = runtime.render().content.decode()
    assert 'llm_calls_total{operation="structured",outcome="success"} 1.0' in content
    assert 'llm_attempts_total{model_alias="primary",outcome="error"} 1.0' in content
    assert 'llm_attempts_total{model_alias="fallback",outcome="success"} 1.0' in content
    assert 'llm_tokens_total{direction="input",model_alias="fallback"} 12.0' in content
