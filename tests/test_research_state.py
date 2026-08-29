"""研究状态和计划边界测试；不使用假模型替代真实 provider 调用."""

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from app.agents.research.state import merge_evidence
from app.agents.research.router import ResearchRouter
from app.schemas.research import (
    Evidence,
    EvidenceSourceKind,
    ResearchPlan,
    ResearchStep,
    RetrievalStrategy,
)


def test_research_plan_requires_contiguous_step_numbers() -> None:
    """步骤编号不能跳跃，否则恢复游标无法稳定指向下一步."""
    with pytest.raises(ValidationError, match="contiguous"):
        ResearchPlan(
            topic="比较两个方案",
            steps=(
                ResearchStep(step_number=1, objective="收集方案 A", search_queries=("方案 A",)),
                ResearchStep(step_number=3, objective="收集方案 B", search_queries=("方案 B",)),
            ),
        )


def test_parallel_evidence_is_deduplicated_by_stable_source_identity() -> None:
    """两条查找路径命中同一来源时只能形成一条报告证据."""
    source_key = "document:chunk-1"
    evidence_id = Evidence.stable_id(
        source_kind=EvidenceSourceKind.DOCUMENT,
        source_key=source_key,
    )
    first = Evidence(
        evidence_id=evidence_id,
        step_id="step-1",
        source_kind=EvidenceSourceKind.DOCUMENT,
        source_key=source_key,
        title="测试文档",
        content="可回查的原文",
        score=0.9,
        retrieved_at=datetime.now(UTC),
        provider="hybrid-rag",
    )

    assert merge_evidence((first,), (first,)) == (first,)


def test_router_adds_required_strategy_without_discarding_planner_choice() -> None:
    """确定性规则应补上时效检索，同时保留 Planner 的文档检索建议."""
    step = ResearchStep(
        step_number=1,
        objective="分析 2026 年最新趋势",
        search_queries=("2026 latest trend",),
        preferred_strategies=(RetrievalStrategy.HYBRID,),
    )

    decision = ResearchRouter().route(step)

    assert decision.strategies == (
        RetrievalStrategy.HYBRID,
        RetrievalStrategy.WEB,
        RetrievalStrategy.GRAPH_GLOBAL,
    )
