"""检查研究证据是否足够、是否冲突以及是否需要补查."""

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.runtime import Runtime

from app.agents.research.context import ResearchRuntimeContext
from app.agents.research.state import ResearchState, ResearchStateUpdate
from app.core.logging import logger
from app.schemas.research import (
    MissingEvidenceRequest,
    ResearchStatus,
    RetrievalStrategy,
    ValidationResult,
)
from app.services.llm.service import LLMService


class ResearchValidator:
    """先执行确定性检查，再让模型判断证据内容是否支持研究结论."""

    def __init__(self, llm_service: LLMService, *, aliases: tuple[str, ...]) -> None:
        """保存统一模型服务和按优先级排列的真实模型别名."""
        if not aliases:
            raise ValueError("validator aliases must not be empty")
        self._llm_service = llm_service
        self._aliases = aliases

    async def __call__(
        self,
        state: ResearchState,
        *,
        runtime: Runtime[ResearchRuntimeContext],
    ) -> ResearchStateUpdate:
        """返回结构化验证结果，并拒绝模型编造不存在的证据 ID."""
        plan = state.get("plan")
        if plan is None:
            raise ValueError("validation requires a research plan")

        evidence = state["evidence"]
        if not evidence:
            missing = tuple(
                MissingEvidenceRequest(
                    step_id=step.step_id,
                    objective=step.objective,
                    search_queries=step.search_queries,
                    preferred_strategies=(RetrievalStrategy.HYBRID, RetrievalStrategy.WEB),
                )
                for step in plan.steps
            )
            result = ValidationResult(
                sufficient=False,
                missing=missing,
                summary="没有取得可验证证据",
            )
            return self._next_update(state, result)

        rendered = "\n\n".join(f"[{item.evidence_id}] source={item.source_key}\n{item.content}" for item in evidence)
        result = await self._llm_service.call_structured(
            (
                SystemMessage(
                    content=(
                        "Validate the research evidence. Create only facts directly supported by the listed "
                        "evidence IDs. Preserve conflicts. If evidence is insufficient, request focused follow-up "
                        "queries. Never invent an evidence ID."
                    )
                ),
                HumanMessage(content=f"Topic: {state['topic']}\n\nEvidence:\n{rendered}"),
            ),
            response_model=ValidationResult,
            aliases=self._aliases,
            overrides={"temperature": 0.0},
        )

        allowed_ids = {item.evidence_id for item in evidence}
        referenced_ids = {
            evidence_id
            for fact in result.facts
            for evidence_id in (*fact.supporting_evidence_ids, *fact.contradicting_evidence_ids)
        } | {evidence_id for conflict in result.conflicts for evidence_id in conflict.evidence_ids}
        if not referenced_ids <= allowed_ids:
            raise ValueError("validator returned unknown evidence IDs")

        logger.info(
            "research_evidence_validated",
            research_id=str(runtime.context.research_id),
            sufficient=result.sufficient,
            fact_count=len(result.facts),
            conflict_count=len(result.conflicts),
            missing_count=len(result.missing),
        )
        return self._next_update(state, result)

    @staticmethod
    def _next_update(state: ResearchState, result: ValidationResult) -> ResearchStateUpdate:
        """把验证判断转换成下一阶段状态，并实施硬循环上限."""
        if result.sufficient:
            return {"validation": result, "status": ResearchStatus.WRITING}

        next_iteration = state["current_iteration"] + 1
        if next_iteration >= state["config"].max_iterations:
            return {
                "validation": result,
                "current_iteration": next_iteration,
                "status": ResearchStatus.INSUFFICIENT_EVIDENCE,
                "stop_reason": "maximum validation iterations reached",
            }
        return {
            "validation": result,
            "current_iteration": next_iteration,
            "status": ResearchStatus.RESEARCHING,
        }


__all__ = ["ResearchValidator"]
