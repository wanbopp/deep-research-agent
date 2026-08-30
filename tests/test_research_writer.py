"""ResearchWriter 的引用编号分配回归测试."""

from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any
from uuid import uuid4

import pytest

from app.agents.research.writer import ResearchWriter
from app.schemas.research import ReportSection


class StubLLMService:
    """返回引用了 R1 的最小合法报告草稿."""

    def __init__(self) -> None:
        """记录收到的 prompt，供断言冲突文本已经生成."""
        self.received_contents: list[str] = []

    async def call_structured(self, messages, *, response_model, aliases, overrides=None, prompt=None):
        """按 writer 期望的关键字参数返回结构化草稿."""
        assert prompt is not None
        self.received_contents = [message.content for message in messages]
        return response_model(
            title="测试报告",
            executive_summary="摘要",
            sections=(ReportSection(heading="发现", body="正文", citation_ids=("R1",)),),
        )


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


def make_state_with_unused_conflict_evidence() -> dict:
    """冲突结论引用了没有任何事实使用的证据（曾触发 KeyError 的形状）."""
    return {
        "topic": "测试主题",
        "evidence": [make_evidence("ev-1"), make_evidence("ev-2"), make_evidence("ev-3")],
        "validation": {
            "sufficient": True,
            "facts": [
                {
                    "fact_id": "f-1",
                    "statement": "事实一",
                    "supporting_evidence_ids": ["ev-1"],
                    "confidence": 0.9,
                }
            ],
            "conflicts": [
                {
                    "description": "两个来源结论相反",
                    "evidence_ids": ["ev-2", "ev-3"],
                }
            ],
            "missing": [],
            "summary": "验证通过",
        },
    }


@pytest.fixture
def anyio_backend() -> str:
    """与 conftest 保持一致，只使用 asyncio 后端."""
    return "asyncio"


@pytest.mark.anyio
async def test_writer_assigns_citations_to_conflict_only_evidence(anyio_backend: str) -> None:
    """冲突引用但事实未使用的证据也必须获得引用编号，任务不再失败."""
    stub = StubLLMService()
    writer = ResearchWriter(stub, aliases=("test-alias",))  # type: ignore[arg-type]

    result: dict[str, Any] = await writer(
        make_state_with_unused_conflict_evidence(),  # type: ignore[arg-type]
        runtime=SimpleNamespace(context=SimpleNamespace(research_id=uuid4())),  # type: ignore[arg-type]
    )

    assert result["status"] == "completed"
    report = result["report"]
    cited_evidence = {citation["evidence_id"] for citation in report["citations"]}
    assert cited_evidence == {"ev-1", "ev-2", "ev-3"}

    citation_ids = [citation["citation_id"] for citation in report["citations"]]
    assert len(citation_ids) == len(set(citation_ids))

    # 冲突文本必须带着解析出的引用编号进入模型输入。
    prompt_text = "\n".join(stub.received_contents)
    assert "两个来源结论相反" in prompt_text
