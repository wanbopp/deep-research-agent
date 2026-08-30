"""统一预算与 Context Fragment 的确定性测试."""

import pytest

from app.runtime import (
    BudgetPolicy,
    ContextAllocator,
    ContextFragment,
    ContextKind,
    ContextSource,
    Sensitivity,
    TrustLevel,
)


def _fragment(content: str, *, source: ContextSource = ContextSource.SYSTEM) -> ContextFragment:
    """构造测试片段."""
    return ContextFragment(
        kind=ContextKind.INSTRUCTION,
        source=source,
        trust_level=TrustLevel.TRUSTED,
        sensitivity=Sensitivity.INTERNAL,
        content=content,
    )


def test_budget_tightening_uses_strictest_value_for_every_dimension() -> None:
    """请求放大任何维度都无效，主动降低的维度会生效."""
    hard = BudgetPolicy()
    requested = BudgetPolicy(
        total_timeout_seconds=600,
        max_input_tokens=4000,
        max_tool_output_tokens=4000,
        max_evidence=10,
        max_retrieval_candidates=100,
        max_parallel_operations=4,
    )

    effective = hard.tighten(requested)

    assert effective.total_timeout_seconds == hard.total_timeout_seconds
    assert effective.max_input_tokens == 4000
    assert effective.max_tool_output_tokens == hard.max_tool_output_tokens
    assert effective.max_evidence == 10
    assert effective.max_retrieval_candidates == hard.max_retrieval_candidates
    assert effective.max_parallel_operations == 4


def test_context_allocator_records_deterministic_truncation_metadata() -> None:
    """总输入不超预算，截断片段保留原始和实际 token 数."""
    result = ContextAllocator(max_tokens=7).allocate(
        (_fragment("system rule"), _fragment("one two three four five six", source=ContextSource.WEB))
    )

    assert result.token_count <= 7
    assert result.truncated_fragment_count == 1
    assert result.fragments[-1].truncated is True
    assert result.fragments[-1].original_tokens > result.fragments[-1].estimated_tokens
    assert result.fragments[-1].source is ContextSource.WEB


def test_budget_and_context_reject_unbounded_sentinel_values() -> None:
    """0 或负数不能被解释为无限预算."""
    with pytest.raises(ValueError, match="greater than zero"):
        BudgetPolicy(max_input_tokens=0)
    with pytest.raises(ValueError, match="greater than zero"):
        ContextAllocator(max_tokens=0)
