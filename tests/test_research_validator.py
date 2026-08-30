"""ResearchValidator 对模型编造证据 ID 的纠正与清洗回归测试."""

from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any
from uuid import uuid4

import pytest

from app.agents.research.validator import ResearchValidator
from app.schemas.research import ValidatedFact, ValidationResult


class ScriptedLLMService:
    """按脚本顺序返回验证结果，并记录收到的消息序列."""

    def __init__(self, results: list[ValidationResult]) -> None:
        """保存按调用次序返回的结果脚本."""
        self._results = list(results)
        self.calls: list[tuple] = []

    async def call_structured(self, messages, *, response_model, aliases, overrides=None, prompt=None):
        """返回脚本中的下一个结果，并记录本轮消息."""
        assert prompt is not None
        self.calls.append(tuple(messages))
        return self._results.pop(0)


def make_evidence(evidence_id: str) -> dict:
    """构造满足 Evidence 校验的最小证据字典."""
    return {
        "evidence_id": evidence_id,
        "step_id": "step-1",
        "source_kind": "web",
        "source_key": f"https://example.com/{evidence_id}",
        "title": f"来源 {evidence_id}",
        "content": "证据正文",
        "score": 0.9,
        "retrieved_at": datetime.now(UTC),
        "provider": "duckduckgo",
    }


def make_state() -> dict:
    """构造包含单步计划与两条证据的最小验证输入."""
    return {
        "topic": "测试主题",
        "config": {
            "max_steps": 5,
            "max_iterations": 2,
            "max_evidence_per_step": 8,
            "max_total_evidence": 30,
            "timeout_seconds": 300.0,
            "require_independent_sources": 2,
        },
        "status": "validating",
        "current_iteration": 0,
        "evidence": [make_evidence("ev-1"), make_evidence("ev-2")],
        "retrieval_failures": [],
        "plan": {
            "topic": "测试主题",
            "steps": [
                {
                    "step_number": 1,
                    "objective": "收集基础资料",
                    "search_queries": ["测试主题 综述"],
                }
            ],
        },
    }


def make_runtime() -> SimpleNamespace:
    """构造只含 research_id 的最小运行时上下文."""
    return SimpleNamespace(context=SimpleNamespace(research_id=uuid4()))


def valid_result() -> ValidationResult:
    """只引用合法证据的充分验证结果."""
    return ValidationResult(
        sufficient=True,
        facts=(
            ValidatedFact(
                fact_id="f-1",
                statement="事实一",
                supporting_evidence_ids=("ev-1",),
                confidence=0.9,
            ),
        ),
        summary="验证通过",
    )


def bogus_result() -> ValidationResult:
    """引用不存在证据 ID 的验证结果."""
    return ValidationResult(
        sufficient=True,
        facts=(
            ValidatedFact(
                fact_id="f-bogus",
                statement="引用了编造的证据",
                supporting_evidence_ids=("ev-does-not-exist",),
                confidence=0.9,
            ),
        ),
        summary="验证通过",
    )


@pytest.fixture
def anyio_backend() -> str:
    """与 conftest 保持一致，只使用 asyncio 后端."""
    return "asyncio"


@pytest.mark.anyio
async def test_validator_corrects_unknown_ids_with_second_call(anyio_backend: str) -> None:
    """首次引用非法 ID 时携带合法集合纠正重试，第二次合法即通过."""
    llm = ScriptedLLMService([bogus_result(), valid_result()])
    validator = ResearchValidator(llm, aliases=("test-alias",))  # type: ignore[arg-type]

    update: dict[str, Any] = await validator(
        make_state(),  # type: ignore[arg-type]
        runtime=make_runtime(),  # type: ignore[arg-type]
    )

    assert len(llm.calls) == 2
    correction_text = llm.calls[1][-1].content
    assert "ev-does-not-exist" in correction_text
    assert "ev-1" in correction_text
    assert update["status"] == "writing"


@pytest.mark.anyio
async def test_validator_sanitizes_partial_invalid_facts(anyio_backend: str) -> None:
    """纠正失败时清洗非法引用：保留合法事实，丢弃失效冲突."""
    still_invalid = ValidationResult(
        sufficient=True,
        facts=(
            ValidatedFact(
                fact_id="f-good",
                statement="有合法支持证据的事实",
                supporting_evidence_ids=("ev-1", "ev-does-not-exist"),
                confidence=0.9,
            ),
            ValidatedFact(
                fact_id="f-lost",
                statement="支持证据全部非法的事实",
                supporting_evidence_ids=("ev-does-not-exist",),
                confidence=0.9,
            ),
        ),
        conflicts=(),
        summary="验证通过",
    )
    llm = ScriptedLLMService([bogus_result(), still_invalid])
    validator = ResearchValidator(llm, aliases=("test-alias",))  # type: ignore[arg-type]

    update: dict[str, Any] = await validator(
        make_state(),  # type: ignore[arg-type]
        runtime=make_runtime(),  # type: ignore[arg-type]
    )

    assert update["status"] == "writing"
    validation = ValidationResult.model_validate(update["validation"])
    assert len(validation.facts) == 1
    assert validation.facts[0].fact_id == "f-good"
    assert validation.facts[0].supporting_evidence_ids == ("ev-1",)


@pytest.mark.anyio
async def test_validator_degrades_to_insufficient_when_no_fact_survives(anyio_backend: str) -> None:
    """清洗后没有任何事实时降级为证据不足并进入补查."""
    llm = ScriptedLLMService([bogus_result(), bogus_result()])
    validator = ResearchValidator(llm, aliases=("test-alias",))  # type: ignore[arg-type]

    update: dict[str, Any] = await validator(
        make_state(),  # type: ignore[arg-type]
        runtime=make_runtime(),  # type: ignore[arg-type]
    )

    assert update["status"] == "researching"
    assert update["current_iteration"] == 1
    validation = ValidationResult.model_validate(update["validation"])
    assert not validation.sufficient
    assert validation.facts == ()
    assert len(validation.missing) == 1
    assert validation.missing[0].step_id == "step-1"
