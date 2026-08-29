"""深度研究计划、证据、验证结果与报告的公共数据结构."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from hashlib import sha256
from typing import Self
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, model_validator


class _StrictResearchModel(BaseModel):
    """研究流程共享的严格、不可变值对象.

    Agent 状态会被保存后再恢复。禁止额外字段可以尽早发现旧节点和新节点的
    数据格式不一致；不可变对象则避免某个并行分支原地修改另一个分支正在读取
    的值。节点需要更新数据时，应创建新对象并把增量交给 LangGraph 合并。
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        str_strip_whitespace=True,
        hide_input_in_errors=True,
    )


class ResearchStatus(StrEnum):
    """研究图对外可解释的执行结果，而不是某个 LangGraph 节点名称."""

    PLANNING = "planning"
    RESEARCHING = "researching"
    VALIDATING = "validating"
    WRITING = "writing"
    COMPLETED = "completed"
    NEEDS_CLARIFICATION = "needs_clarification"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    BUDGET_EXHAUSTED = "budget_exhausted"
    FAILED = "failed"
    CANCELLED = "cancelled"


class RetrievalStrategy(StrEnum):
    """一个研究步骤可以使用的资料查找方式."""

    HYBRID = "hybrid"
    GRAPH_LOCAL = "graph_local"
    GRAPH_GLOBAL = "graph_global"
    WEB = "web"


class ResearchConfig(_StrictResearchModel):
    """一次研究任务可以消耗的服务端预算.

    客户端可以请求更小的预算，但 API/service 必须使用服务端上限重新校验，不能
    让客户端通过传入很大的数字制造无限检索循环或不可控的模型费用。
    """

    max_steps: int = Field(default=5, ge=1, le=8)
    max_iterations: int = Field(default=2, ge=1, le=4)
    max_evidence_per_step: int = Field(default=8, ge=1, le=20)
    max_total_evidence: int = Field(default=30, ge=1, le=80)
    timeout_seconds: float = Field(default=300.0, ge=10.0, le=900.0)
    require_independent_sources: int = Field(default=2, ge=1, le=4)


class ResearchStep(_StrictResearchModel):
    """计划中一个能够单独执行和判断完成的研究步骤."""

    # 保留从 1 开始的公开编号，便于报告和进度事件按人类习惯显示。
    step_number: int = Field(ge=1)
    objective: str = Field(min_length=1, max_length=500)
    search_queries: tuple[str, ...] = Field(min_length=1, max_length=5)
    # 完成条件回答“搜到什么才算这一步做完”，防止 Planner 只输出宽泛标题。
    completion_criteria: str = Field(
        default="找到至少一条能够直接支持该目标的可引用证据",
        min_length=1,
        max_length=500,
    )
    # Planner 给出建议；Router 还会应用确定性规则，不能盲信模型选择。
    preferred_strategies: tuple[RetrievalStrategy, ...] = Field(
        default=(RetrievalStrategy.HYBRID,),
        min_length=1,
        max_length=3,
    )

    @property
    def step_id(self) -> str:
        """返回稳定步骤 ID，供并行结果归组，不依赖执行完成顺序."""
        return f"step-{self.step_number}"


class ResearchPlan(_StrictResearchModel):
    """Planner 输出的有序执行计划."""

    topic: str = Field(min_length=1, max_length=1000)
    steps: tuple[ResearchStep, ...] = Field(min_length=1, max_length=8)
    assumptions: tuple[str, ...] = Field(default=(), max_length=5)
    clarification_question: str | None = Field(default=None, max_length=500)

    @model_validator(mode="after")
    def validate_step_order(self) -> Self:
        """拒绝重复或跳号步骤，保证恢复时游标含义稳定."""
        expected = tuple(range(1, len(self.steps) + 1))
        actual = tuple(step.step_number for step in self.steps)
        if actual != expected:
            raise ValueError("research step numbers must be contiguous and start at one")
        return self


class EvidenceSourceKind(StrEnum):
    """证据来自用户文档、知识图还是外部网页."""

    DOCUMENT = "document"
    GRAPH = "graph"
    WEB = "web"


class Evidence(_StrictResearchModel):
    """所有检索器都必须返回的统一证据形状.

    Writer 只认识 Evidence，不需要知道 pgvector、Neo4j 或网页搜索库的私有返回
    类型。source_key 是去重与引用的稳定身份；content 是允许进入验证和写作的
    有界片段，不应放入完整网页或整篇文档。
    """

    evidence_id: str = Field(min_length=1, max_length=80)
    step_id: str = Field(pattern=r"^step-[1-9][0-9]*$")
    source_kind: EvidenceSourceKind
    source_key: str = Field(min_length=1, max_length=1000)
    title: str = Field(min_length=1, max_length=500)
    content: str = Field(min_length=1, max_length=8000)
    score: float = Field(ge=0.0)
    document_id: UUID | None = None
    chunk_id: UUID | None = None
    url: HttpUrl | None = None
    published_at: datetime | None = None
    retrieved_at: datetime
    provider: str = Field(min_length=1, max_length=100)

    @classmethod
    def stable_id(cls, *, source_kind: EvidenceSourceKind, source_key: str) -> str:
        """从可信来源身份生成短 ID，避免让模型自行发明引用编号."""
        digest = sha256(f"{source_kind.value}:{source_key}".encode()).hexdigest()[:16]
        return f"ev-{digest}"


class RetrievalFailure(_StrictResearchModel):
    """某条查找路径的安全失败摘要；不保存异常正文或第三方响应."""

    step_id: str
    strategy: RetrievalStrategy
    error_type: str = Field(min_length=1, max_length=200)


class RouteDecision(_StrictResearchModel):
    """Router 为一个步骤选择的查找方式及可记录理由."""

    step_id: str = Field(pattern=r"^step-[1-9][0-9]*$")
    strategies: tuple[RetrievalStrategy, ...] = Field(min_length=1, max_length=4)
    reason: str = Field(min_length=1, max_length=500)


class ValidatedFact(_StrictResearchModel):
    """经过验证、允许进入报告的事实及其支持证据."""

    fact_id: str = Field(min_length=1, max_length=80)
    statement: str = Field(min_length=1, max_length=2000)
    supporting_evidence_ids: tuple[str, ...] = Field(min_length=1)
    contradicting_evidence_ids: tuple[str, ...] = ()
    confidence: float = Field(ge=0.0, le=1.0)


class Conflict(_StrictResearchModel):
    """不能被系统静默裁决的相互冲突结论."""

    description: str = Field(min_length=1, max_length=2000)
    evidence_ids: tuple[str, ...] = Field(min_length=2)


class MissingEvidenceRequest(_StrictResearchModel):
    """Validator 请求下一轮补查的明确缺口."""

    step_id: str = Field(pattern=r"^step-[1-9][0-9]*$")
    objective: str = Field(min_length=1, max_length=500)
    search_queries: tuple[str, ...] = Field(min_length=1, max_length=5)
    preferred_strategies: tuple[RetrievalStrategy, ...] = Field(min_length=1, max_length=3)


class ValidationResult(_StrictResearchModel):
    """证据是否足以写报告，以及不足时还缺什么."""

    sufficient: bool
    facts: tuple[ValidatedFact, ...] = ()
    conflicts: tuple[Conflict, ...] = ()
    missing: tuple[MissingEvidenceRequest, ...] = ()
    summary: str = Field(min_length=1, max_length=2000)

    @model_validator(mode="after")
    def validate_decision(self) -> Self:
        """充分时必须有事实；不充分时必须说明缺口或冲突."""
        if self.sufficient and not self.facts:
            raise ValueError("sufficient validation must contain facts")
        if not self.sufficient and not (self.missing or self.conflicts):
            raise ValueError("insufficient validation must describe missing evidence or conflicts")
        return self


class ReportSection(_StrictResearchModel):
    """报告中的一个结构化章节."""

    heading: str = Field(min_length=1, max_length=200)
    body: str = Field(min_length=1, max_length=12000)
    citation_ids: tuple[str, ...] = ()


class Citation(_StrictResearchModel):
    """最终显示引用与内部证据之间的映射."""

    citation_id: str = Field(pattern=r"^R[1-9][0-9]*$")
    evidence_id: str = Field(min_length=1, max_length=80)
    title: str = Field(min_length=1, max_length=500)
    source: str = Field(min_length=1, max_length=2000)


class ResearchReport(_StrictResearchModel):
    """可以持久化并重新渲染的最终研究报告."""

    title: str = Field(min_length=1, max_length=300)
    executive_summary: str = Field(min_length=1, max_length=4000)
    sections: tuple[ReportSection, ...] = Field(min_length=1)
    limitations: tuple[str, ...] = ()
    citations: tuple[Citation, ...] = ()


__all__ = [
    "Citation",
    "Conflict",
    "Evidence",
    "EvidenceSourceKind",
    "MissingEvidenceRequest",
    "ReportSection",
    "ResearchConfig",
    "ResearchPlan",
    "ResearchReport",
    "ResearchStatus",
    "ResearchStep",
    "RouteDecision",
    "RetrievalFailure",
    "RetrievalStrategy",
    "ValidatedFact",
    "ValidationResult",
]
