"""把 Hybrid RAG、GraphRAG 和网页搜索统一为研究证据."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Protocol
from time import perf_counter
from uuid import UUID

from ddgs import DDGS
from langgraph.runtime import Runtime
from pydantic import HttpUrl

from app.agents.research.context import ResearchRuntimeContext
from app.agents.research.router import ResearchRouter
from app.agents.research.state import ResearchState, ResearchStateUpdate
from app.core.logging import logger
from app.graphrag.global_retriever import GlobalGraphRetriever
from app.graphrag.local_retriever import LocalGraphRetriever
from app.observability import metrics, tracing
from app.rag.hybrid import HybridRetriever
from app.schemas.research import (
    Evidence,
    EvidenceSourceKind,
    ResearchConfig,
    ResearchPlan,
    ResearchStep,
    ResearchStatus,
    RetrievalFailure,
    RetrievalStrategy,
    ValidationResult,
)
from app.runtime import BudgetPolicy


class EvidenceRetriever(Protocol):
    """所有资料来源对研究协调器暴露的统一调用形状."""

    async def search(
        self,
        *,
        user_id: UUID,
        step: ResearchStep,
        query: str,
        top_k: int,
    ) -> tuple[Evidence, ...]:
        """返回已经绑定 step、来源身份和取得时间的证据."""
        ...


class HybridEvidenceRetriever:
    """把用户文档的文字检索结果转换为统一 Evidence."""

    def __init__(self, retriever: HybridRetriever) -> None:
        """保存 Phase 5 已创建的用户隔离 Hybrid Retriever."""
        self._retriever = retriever

    async def search(self, *, user_id: UUID, step: ResearchStep, query: str, top_k: int) -> tuple[Evidence, ...]:
        """检索用户文档，并把每个 chunk 转换为可引用证据."""
        candidates = await self._retriever.search(
            user_id=user_id,
            query=query,
            candidate_k=max(top_k * 3, 10),
            final_k=top_k,
        )
        now = datetime.now(UTC)
        return tuple(
            Evidence(
                evidence_id=Evidence.stable_id(
                    source_kind=EvidenceSourceKind.DOCUMENT,
                    source_key=f"chunk:{candidate.chunk_id}",
                ),
                step_id=step.step_id,
                source_kind=EvidenceSourceKind.DOCUMENT,
                source_key=f"chunk:{candidate.chunk_id}",
                title=f"Document {candidate.document_id}",
                content=candidate.text,
                score=max(candidate.score, 0.0),
                document_id=candidate.document_id,
                chunk_id=candidate.chunk_id,
                retrieved_at=now,
                provider="hybrid-rag",
            )
            for candidate in candidates
        )


class LocalGraphEvidenceRetriever:
    """把实体邻域路径转换为仍可回查原始 chunk 的证据."""

    def __init__(self, retriever: LocalGraphRetriever) -> None:
        """保存 Phase 6 已创建的局部实体关系检索器."""
        self._retriever = retriever

    async def search(self, *, user_id: UUID, step: ResearchStep, query: str, top_k: int) -> tuple[Evidence, ...]:
        """围绕问题中的实体展开有限路径，不执行无限图遍历."""
        result = await self._retriever.search(user_id=user_id, query=query, limit=top_k)
        now = datetime.now(UTC)
        evidence: list[Evidence] = []
        for path in result.paths:
            source_key = "graph-path:" + ":".join(str(value) for value in path.source_chunk_ids)
            relations = " -> ".join(item.value for item in path.relation_types)
            content = f"Entities: {' -> '.join(path.entity_names)}\nRelations: {relations}\nEvidence: {' | '.join(path.evidence_texts)}"
            evidence.append(
                Evidence(
                    evidence_id=Evidence.stable_id(source_kind=EvidenceSourceKind.GRAPH, source_key=source_key),
                    step_id=step.step_id,
                    source_kind=EvidenceSourceKind.GRAPH,
                    source_key=source_key,
                    title="Entity relationship path",
                    content=content,
                    score=1.0,
                    chunk_id=path.source_chunk_ids[0],
                    retrieved_at=now,
                    provider="graphrag-local",
                )
            )
        return tuple(evidence)


class GlobalGraphEvidenceRetriever:
    """把跨文档社区摘要转换为统一证据，并保留原始 chunk 引用."""

    def __init__(self, retriever: GlobalGraphRetriever) -> None:
        """保存 Phase 6 已创建的跨文档社区检索器."""
        self._retriever = retriever

    async def search(self, *, user_id: UUID, step: ResearchStep, query: str, top_k: int) -> tuple[Evidence, ...]:
        """查找最相关社区摘要，并为每项绑定原始来源 chunk."""
        result = await self._retriever.search(user_id=user_id, query=query, top_k=top_k)
        now = datetime.now(UTC)
        return tuple(
            Evidence(
                evidence_id=Evidence.stable_id(
                    source_kind=EvidenceSourceKind.GRAPH,
                    source_key=f"community:{community.community_id}",
                ),
                step_id=step.step_id,
                source_kind=EvidenceSourceKind.GRAPH,
                source_key=f"community:{community.community_id}",
                title=community.title,
                content=community.summary,
                score=max(score, 0.0),
                chunk_id=community.source_chunk_ids[0],
                retrieved_at=now,
                provider="graphrag-global",
            )
            for community, score in zip(result.communities, result.scores, strict=True)
        )


class WebEvidenceRetriever:
    """在线查询公开网页摘要；网页内容仍是待验证证据，不是既定事实."""

    def __init__(self, *, timeout_seconds: int = 10) -> None:
        """设置单次公共网页搜索的网络超时."""
        self._timeout_seconds = timeout_seconds

    async def search(self, *, user_id: UUID, step: ResearchStep, query: str, top_k: int) -> tuple[Evidence, ...]:
        """搜索网页摘要并保留 URL，结果仍需经过 Validator."""
        del user_id  # 网页本身是公共来源，用户身份只用于上层任务所有权和限流。

        def fetch() -> list[dict[str, str]]:
            """把同步搜索库移到线程，避免阻塞运行 Agent 的事件循环."""
            return DDGS(timeout=self._timeout_seconds).text(query, max_results=top_k)

        rows = await asyncio.to_thread(fetch)
        now = datetime.now(UTC)
        evidence: list[Evidence] = []
        for row in rows:
            url = row.get("href", "").strip()
            title = row.get("title", "").strip()
            body = row.get("body", "").strip()
            if not url or not title or not body:
                continue
            evidence.append(
                Evidence(
                    evidence_id=Evidence.stable_id(source_kind=EvidenceSourceKind.WEB, source_key=url),
                    step_id=step.step_id,
                    source_kind=EvidenceSourceKind.WEB,
                    source_key=url,
                    title=title,
                    content=body,
                    score=max(0.0, 1.0 - len(evidence) * 0.05),
                    url=HttpUrl(url),
                    retrieved_at=now,
                    provider="duckduckgo",
                )
            )
        return tuple(evidence)


class ParallelResearchRetriever:
    """按步骤并行执行查找，并允许单条来源失败后保留其他结果."""

    def __init__(
        self,
        *,
        router: ResearchRouter,
        retrievers: Mapping[RetrievalStrategy, EvidenceRetriever],
        budget_policy: BudgetPolicy | None = None,
    ) -> None:
        """保存无请求状态 Router 和按策略注册的查找适配器."""
        self._router = router
        self._retrievers = dict(retrievers)
        self._budget = budget_policy or BudgetPolicy()

    async def __call__(
        self,
        state: ResearchState,
        *,
        runtime: Runtime[ResearchRuntimeContext],
    ) -> ResearchStateUpdate:
        """并行展开步骤、查询和来源，汇合为证据增量与安全失败摘要."""
        plan_data = state.get("plan")
        if plan_data is None:
            raise ValueError("retrieval requires a research plan")
        plan = ResearchPlan.model_validate(plan_data)
        config = ResearchConfig.model_validate(state["config"])

        steps = plan.steps
        validation_data = state.get("validation")
        validation = ValidationResult.model_validate(validation_data) if validation_data is not None else None
        if state["current_iteration"] > 0 and validation is not None and validation.missing:
            # 补查不能机械重复第一轮查询。Validator 已经指出具体缺口，这里把缺口
            # 转回对应步骤的临时执行视图；原计划仍保留在 state 中供审计和报告使用。
            original_by_id = {step.step_id: step for step in plan.steps}
            supplemental: list[ResearchStep] = []
            for request in validation.missing:
                original = original_by_id.get(request.step_id)
                if original is None:
                    raise ValueError("missing evidence request references unknown step")
                supplemental.append(
                    original.model_copy(
                        update={
                            "objective": request.objective,
                            "search_queries": request.search_queries,
                            "preferred_strategies": request.preferred_strategies,
                        }
                    )
                )
            steps = tuple(supplemental)

        work: list[tuple[ResearchStep, RetrievalStrategy, str]] = []
        for step in steps:
            decision = self._router.route(step)
            for strategy in decision.strategies:
                for query in step.search_queries:
                    work.append((step, strategy, query))

        async def run_one(
            step: ResearchStep,
            strategy: RetrievalStrategy,
            query: str,
        ) -> tuple[Evidence, ...] | RetrievalFailure:
            retriever = self._retrievers.get(strategy)
            if retriever is None:
                metrics.observe_retrieval(
                    strategy=strategy.value,
                    outcome="unavailable",
                    duration_seconds=0.0,
                    candidate_count=None,
                )
                return RetrievalFailure(step_id=step.step_id, strategy=strategy, error_type="UnavailableRetriever")
            started_at = perf_counter()
            try:
                async with (
                    semaphore,
                    asyncio.timeout(min(30.0, config.timeout_seconds, self._budget.total_timeout_seconds)),
                ):
                    # query 和证据正文不进入 trace；只关联有限检索策略与研究 ID。
                    with tracing.span(
                        "retrieval",
                        research_id=runtime.context.research_id,
                        retrieval_strategy=strategy.value,
                    ):
                        result = await retriever.search(
                            user_id=runtime.context.user_id,
                            step=step,
                            query=query,
                            top_k=min(config.max_evidence_per_step, self._budget.max_retrieval_candidates),
                        )
                    metrics.observe_retrieval(
                        strategy=strategy.value,
                        outcome="success",
                        duration_seconds=perf_counter() - started_at,
                        candidate_count=len(result),
                    )
                    return result
            except Exception as error:
                metrics.observe_retrieval(
                    strategy=strategy.value,
                    outcome="error",
                    duration_seconds=perf_counter() - started_at,
                    candidate_count=None,
                )
                logger.warning(
                    "research_retrieval_failed",
                    research_id=str(runtime.context.research_id),
                    step_id=step.step_id,
                    strategy=strategy.value,
                    error_type=type(error).__name__,
                )
                return RetrievalFailure(step_id=step.step_id, strategy=strategy, error_type=type(error).__name__)

        semaphore = asyncio.Semaphore(self._budget.max_parallel_operations)
        results = await asyncio.gather(*(run_one(*item) for item in work))
        evidence: list[Evidence] = []
        failures: list[RetrievalFailure] = []
        for result in results:
            if isinstance(result, RetrievalFailure):
                failures.append(result)
            else:
                evidence.extend(result)

        # 全局上限在合并前执行，避免一个宽泛查询压垮 checkpoint 和 Writer 输入。
        limited = tuple(sorted(evidence, key=lambda item: (-item.score, item.evidence_id)))[
            : min(config.max_total_evidence, self._budget.max_evidence)
        ]
        return {
            # 节点内部使用模型获得类型保护，跨节点边界则降为 JSON 数据交给
            # checkpointer。下一节点会重新校验，不会因此失去数据约束。
            "evidence": tuple(item.model_dump(mode="json") for item in limited),
            "retrieval_failures": tuple(item.model_dump(mode="json") for item in failures),
            "status": ResearchStatus.VALIDATING.value,
        }


__all__ = [
    "EvidenceRetriever",
    "GlobalGraphEvidenceRetriever",
    "HybridEvidenceRetriever",
    "LocalGraphEvidenceRetriever",
    "ParallelResearchRetriever",
    "WebEvidenceRetriever",
]
