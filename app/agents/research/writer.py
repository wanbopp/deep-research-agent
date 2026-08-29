"""只根据已验证事实生成报告，并由服务端绑定引用表."""

from pydantic import BaseModel, ConfigDict, Field
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.runtime import Runtime

from app.agents.research.context import ResearchRuntimeContext
from app.agents.research.state import ResearchState, ResearchStateUpdate
from app.core.logging import logger
from app.schemas.research import (
    Citation,
    Evidence,
    ReportSection,
    ResearchReport,
    ResearchStatus,
    ValidationResult,
)
from app.services.llm.service import LLMService


class _ReportDraft(BaseModel):
    """模型可以撰写的内容；正式 Citation 对象由服务端补充."""

    model_config = ConfigDict(frozen=True, extra="forbid", str_strip_whitespace=True, hide_input_in_errors=True)

    title: str = Field(min_length=1, max_length=300)
    executive_summary: str = Field(min_length=1, max_length=4000)
    sections: tuple[ReportSection, ...] = Field(min_length=1)
    limitations: tuple[str, ...] = ()


class ResearchWriter:
    """把验证后的事实写成可持久化报告，不接触未验证原始材料."""

    def __init__(self, llm_service: LLMService, *, aliases: tuple[str, ...]) -> None:
        """保存统一模型服务和真实报告模型别名."""
        if not aliases:
            raise ValueError("writer aliases must not be empty")
        self._llm_service = llm_service
        self._aliases = aliases

    async def __call__(
        self,
        state: ResearchState,
        *,
        runtime: Runtime[ResearchRuntimeContext],
    ) -> ResearchStateUpdate:
        """生成报告草稿，校验引用，并附加服务端构造的来源表."""
        validation_data = state.get("validation")
        if validation_data is None:
            raise ValueError("report writing requires a validation result")
        validation = ValidationResult.model_validate(validation_data)

        if not validation.facts:
            # 没有足够证据也要形成可查询结果，而不是把“不知道”伪装成系统异常。
            # 这里不用模型扩写，避免在无事实输入时诱发看似完整的编造报告。
            report = ResearchReport(
                title=f"受限研究结果：{state['topic'][:200]}",
                executive_summary=validation.summary,
                sections=(
                    ReportSection(
                        heading="当前结论",
                        body="现有资料不足以形成可靠结论。",
                    ),
                ),
                limitations=tuple(
                    dict.fromkeys(
                        [
                            state.get("stop_reason", "证据不足"),
                            *[item.objective for item in validation.missing],
                            *[item.description for item in validation.conflicts],
                        ]
                    )
                ),
            )
            return {
                "report": report.model_dump(mode="json"),
                "status": ResearchStatus.COMPLETED.value,
            }

        evidence = tuple(Evidence.model_validate(item) for item in state["evidence"])
        evidence_by_id = {item.evidence_id: item for item in evidence}
        # 事实与冲突结论引用的证据都必须分配引用编号：冲突可能引用没有任何事实
        # 使用的证据，遗漏它们会让下方查表抛出 KeyError 并使整个任务失败。
        used_evidence_ids = tuple(
            dict.fromkeys(
                [
                    *[
                        evidence_id
                        for fact in validation.facts
                        for evidence_id in (*fact.supporting_evidence_ids, *fact.contradicting_evidence_ids)
                    ],
                    *[evidence_id for conflict in validation.conflicts for evidence_id in conflict.evidence_ids],
                ]
            )
        )
        citation_id_by_evidence = {
            evidence_id: f"R{index}" for index, evidence_id in enumerate(used_evidence_ids, start=1)
        }
        citations = tuple(
            self._make_citation(
                citation_id=citation_id_by_evidence[evidence_id],
                evidence=evidence_by_id[evidence_id],
            )
            for evidence_id in used_evidence_ids
        )

        facts_text = "\n".join(
            (
                f"- {fact.statement} "
                f"citations={[citation_id_by_evidence[value] for value in fact.supporting_evidence_ids]} "
                f"conflicts={[citation_id_by_evidence[value] for value in fact.contradicting_evidence_ids]}"
            )
            for fact in validation.facts
        )
        conflicts_text = (
            "\n".join(
                f"- {item.description}: {[citation_id_by_evidence[value] for value in item.evidence_ids]}"
                for item in validation.conflicts
            )
            or "none"
        )

        draft = await self._llm_service.call_structured(
            (
                SystemMessage(
                    content=(
                        "Write a concise research report only from the validated facts. Use citation IDs exactly "
                        "as provided. Every factual section must list its citation_ids. Preserve conflicts and "
                        "limitations; never invent a citation or source."
                    )
                ),
                HumanMessage(
                    content=(
                        f"Topic: {state['topic']}\nValidated facts:\n{facts_text}\n\n"
                        f"Conflicts:\n{conflicts_text}\n\nValidation summary: {validation.summary}"
                    )
                ),
            ),
            response_model=_ReportDraft,
            aliases=self._aliases,
            overrides={"temperature": 0.0},
        )

        allowed_citations = set(citation_id_by_evidence.values())
        referenced = {citation_id for section in draft.sections for citation_id in section.citation_ids}
        if not referenced:
            raise ValueError("research report does not cite any validated evidence")
        if not referenced <= allowed_citations:
            raise ValueError("research report contains unknown citation IDs")

        report = ResearchReport(
            title=draft.title,
            executive_summary=draft.executive_summary,
            sections=draft.sections,
            limitations=draft.limitations,
            citations=citations,
        )
        logger.info(
            "research_report_created",
            research_id=str(runtime.context.research_id),
            section_count=len(report.sections),
            citation_count=len(report.citations),
        )
        return {
            "report": report.model_dump(mode="json"),
            "status": ResearchStatus.COMPLETED.value,
        }

    @staticmethod
    def _make_citation(*, citation_id: str, evidence: Evidence) -> Citation:
        """从受信 Evidence 构造公开引用；模型没有修改来源的机会."""
        if evidence.url is not None:
            source = str(evidence.url)
        elif evidence.chunk_id is not None:
            source = f"chunk:{evidence.chunk_id}"
        else:
            source = evidence.source_key
        return Citation(
            citation_id=citation_id,
            evidence_id=evidence.evidence_id,
            title=evidence.title,
            source=source,
        )


def render_report_markdown(report: ResearchReport) -> str:
    """把结构化报告确定性渲染为 Markdown，不再次调用模型."""
    parts = [f"# {report.title}", "", report.executive_summary]
    for section in report.sections:
        parts.extend(("", f"## {section.heading}", "", section.body))
        if section.citation_ids:
            parts.append("\nSources: " + " ".join(f"[{value}]" for value in section.citation_ids))
    if report.limitations:
        parts.extend(("", "## Limitations", "", *[f"- {item}" for item in report.limitations]))
    if report.citations:
        parts.extend(("", "## References", ""))
        parts.extend(f"- [{item.citation_id}] {item.title}: {item.source}" for item in report.citations)
    return "\n".join(parts).strip() + "\n"


__all__ = ["ResearchWriter", "render_report_markdown"]
