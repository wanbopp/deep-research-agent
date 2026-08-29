"""检查研究证据是否足够、是否冲突以及是否需要补查."""

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.runtime import Runtime

from app.agents.research.context import ResearchRuntimeContext
from app.agents.research.state import ResearchState, ResearchStateUpdate
from app.core.logging import logger
from app.schemas.research import (
    Conflict,
    Evidence,
    MissingEvidenceRequest,
    ResearchConfig,
    ResearchPlan,
    ResearchStatus,
    RetrievalStrategy,
    ValidatedFact,
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
        """返回结构化验证结果，并约束模型编造的证据 ID.

        模型第一次引用不存在的证据 ID 时，会携带合法 ID 集合再给一次纠正机会；
        纠正后仍然非法的引用会被清洗（丢弃失去支持证据的事实/冲突），最坏情况
        降级为证据不足进入补查，而不是让整个研究任务失败。
        """
        plan_data = state.get("plan")
        if plan_data is None:
            raise ValueError("validation requires a research plan")
        plan = ResearchPlan.model_validate(plan_data)
        evidence = tuple(Evidence.model_validate(item) for item in state["evidence"])

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
        base_messages = (
            SystemMessage(
                content=(
                    "Validate the research evidence. Create only facts directly supported by the listed "
                    "evidence IDs. Preserve conflicts. If evidence is insufficient, request focused follow-up "
                    "queries. Never invent an evidence ID."
                )
            ),
            HumanMessage(content=f"Topic: {state['topic']}\n\nEvidence:\n{rendered}"),
        )
        result = await self._llm_service.call_structured(
            base_messages,
            response_model=ValidationResult,
            aliases=self._aliases,
            overrides={"temperature": 0.0},
        )

        allowed_ids = {item.evidence_id for item in evidence}
        unknown_ids = self._unknown_ids(result, allowed_ids)
        if unknown_ids:
            # 模型偶发编造 ID 时先给一次纠正机会：明确列出合法 ID 集合，
            # 要求重新输出。纠正调用仍带相同结构化约束与温度设置。
            logger.warning(
                "research_validator_unknown_evidence_ids",
                research_id=str(runtime.context.research_id),
                unknown_count=len(unknown_ids),
                stage="first_pass",
            )
            correction = HumanMessage(
                content=(
                    "Your validation referenced evidence IDs that do not exist: "
                    f"{', '.join(sorted(unknown_ids))}. Only these evidence IDs exist: "
                    f"{', '.join(sorted(allowed_ids))}. Produce the validation again using "
                    "only existing evidence IDs."
                )
            )
            result = await self._llm_service.call_structured(
                (*base_messages, correction),
                response_model=ValidationResult,
                aliases=self._aliases,
                overrides={"temperature": 0.0},
            )
            unknown_ids = self._unknown_ids(result, allowed_ids)

        if unknown_ids:
            # 纠正后仍引用非法 ID：不能因此失败整个任务。清洗引用非法 ID 的
            # 事实与冲突，全部不可信时降级为证据不足，交给补查循环或有界终止。
            logger.warning(
                "research_validator_unknown_evidence_ids",
                research_id=str(runtime.context.research_id),
                unknown_count=len(unknown_ids),
                stage="sanitized",
            )
            result = self._sanitize(result, allowed_ids=allowed_ids, plan=plan)

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
    def _unknown_ids(result: ValidationResult, allowed_ids: set[str]) -> set[str]:
        """收集验证结果引用但不存在的证据 ID."""
        referenced_ids = {
            evidence_id
            for fact in result.facts
            for evidence_id in (*fact.supporting_evidence_ids, *fact.contradicting_evidence_ids)
        } | {evidence_id for conflict in result.conflicts for evidence_id in conflict.evidence_ids}
        return referenced_ids - allowed_ids

    @staticmethod
    def _sanitize(
        result: ValidationResult,
        *,
        allowed_ids: set[str],
        plan: ResearchPlan,
    ) -> ValidationResult:
        """移除引用非法证据 ID 的事实与冲突，保持验证结果可用.

        事实失去全部支持证据后整体丢弃；冲突剩余证据不足两条时也丢弃（不满足
        冲突的最小证据要求）。若清洗后不再有任何事实，把结果降级为证据不足并
        按计划步骤请求补查，由迭代上限保证最终收敛。
        """
        sanitized_facts: list[ValidatedFact] = []
        for fact in result.facts:
            supporting = tuple(v for v in fact.supporting_evidence_ids if v in allowed_ids)
            if not supporting:
                continue
            sanitized_facts.append(
                ValidatedFact(
                    fact_id=fact.fact_id,
                    statement=fact.statement,
                    supporting_evidence_ids=supporting,
                    contradicting_evidence_ids=tuple(v for v in fact.contradicting_evidence_ids if v in allowed_ids),
                    confidence=fact.confidence,
                )
            )

        sanitized_conflicts: list[Conflict] = []
        for conflict in result.conflicts:
            remaining = tuple(v for v in conflict.evidence_ids if v in allowed_ids)
            if len(remaining) >= 2:
                sanitized_conflicts.append(Conflict(description=conflict.description, evidence_ids=remaining))

        if sanitized_facts:
            return ValidationResult(
                sufficient=result.sufficient,
                facts=tuple(sanitized_facts),
                conflicts=tuple(sanitized_conflicts),
                missing=result.missing,
                summary=result.summary,
            )

        return ValidationResult(
            sufficient=False,
            missing=tuple(
                MissingEvidenceRequest(
                    step_id=step.step_id,
                    objective=step.objective,
                    search_queries=step.search_queries,
                    preferred_strategies=(RetrievalStrategy.HYBRID, RetrievalStrategy.WEB),
                )
                for step in plan.steps
            ),
            summary="验证结果引用的证据 ID 不可信，需要重新检索证据",
        )

    @staticmethod
    def _next_update(state: ResearchState, result: ValidationResult) -> ResearchStateUpdate:
        """把验证判断转换成下一阶段状态，并实施硬循环上限."""
        if result.sufficient:
            return {
                "validation": result.model_dump(mode="json"),
                "status": ResearchStatus.WRITING.value,
            }

        next_iteration = state["current_iteration"] + 1
        config = ResearchConfig.model_validate(state["config"])
        if next_iteration >= config.max_iterations:
            return {
                "validation": result.model_dump(mode="json"),
                "current_iteration": next_iteration,
                "status": ResearchStatus.INSUFFICIENT_EVIDENCE.value,
                "stop_reason": "maximum validation iterations reached",
            }
        return {
            "validation": result.model_dump(mode="json"),
            "current_iteration": next_iteration,
            "status": ResearchStatus.RESEARCHING.value,
        }


__all__ = ["ResearchValidator"]
