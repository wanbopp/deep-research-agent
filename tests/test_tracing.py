"""Safe tracing 的脱敏与失败开放测试."""

from contextlib import AbstractContextManager, nullcontext
from uuid import uuid4

import pytest

from app.observability.evaluation import EvalRunMetadata
from app.observability.tracing import NoopTraceSink, SafeTracingRuntime


class RecordingSink:
    """记录测试收到的 span，不接触外部网络."""

    def __init__(self) -> None:
        """初始化空记录列表."""
        self.records: list[tuple[str, dict[str, str | int]]] = []

    def span(self, name: str, metadata: dict[str, str | int]) -> AbstractContextManager[object]:
        """保存不可变副本，验证运行时没有上传内容字段."""
        self.records.append((name, dict(metadata)))
        return nullcontext()

    def close(self) -> None:
        """测试 sink 无资源."""


class FailingSink:
    """模拟外部 SDK 在创建 span 时故障."""

    def span(self, name: str, metadata: dict[str, str | int]) -> AbstractContextManager[object]:
        """始终抛错，业务代码仍必须继续."""
        del name, metadata
        raise RuntimeError("trace backend unavailable")

    def close(self) -> None:
        """关闭同样失败，用于验证 fail-open."""
        raise RuntimeError("flush failed")


def test_trace_metadata_only_contains_allowlisted_structured_fields() -> None:
    """研究关联 ID 可发送，但用户、prompt 和证据字段根本无法绑定."""
    sink = RecordingSink()
    runtime = SafeTracingRuntime(sink)
    research_id = uuid4()
    run_id = uuid4()

    with runtime.bind(research_id=research_id, run_id=run_id, attempt_no=2):
        with runtime.span("research.node", node_name="retrieve"):
            pass

    assert sink.records == [
        (
            "research.node",
            {
                "research_id": str(research_id),
                "run_id": str(run_id),
                "attempt_no": 2,
                "node_name": "retrieve",
            },
        )
    ]
    with pytest.raises(ValueError, match="unsupported observation context field"):
        with runtime.bind(prompt="secret"):
            pass


def test_trace_failure_does_not_change_business_result_or_exception() -> None:
    """追踪创建和刷新失败不能影响正常结果，也不能吞掉业务异常."""
    runtime = SafeTracingRuntime(FailingSink())
    with runtime.span("llm", model_alias="primary"):
        result = 42
    assert result == 42

    with pytest.raises(LookupError, match="business"):
        with runtime.span("retrieval", retrieval_strategy="web"):
            raise LookupError("business")
    runtime.close()


def test_noop_and_eval_metadata_contracts() -> None:
    """关闭 tracing 无副作用，评测元数据必须包含全部版本维度."""
    runtime = SafeTracingRuntime(NoopTraceSink())
    with runtime.span("llm"):
        pass
    metadata = EvalRunMetadata(
        code_revision="abc123",
        model_versions=("planner:gpt-5",),
        prompt_versions=("planner:v1",),
        dataset_version="golden:v2",
    )
    assert metadata.dataset_version == "golden:v2"
    with pytest.raises(ValueError, match="must not be empty"):
        EvalRunMetadata(
            code_revision="abc123",
            model_versions=(),
            prompt_versions=("planner:v1",),
            dataset_version="golden:v2",
        )
