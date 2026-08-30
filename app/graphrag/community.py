"""可替换社区检测与真实 LLM 社区摘要生成."""

from collections import defaultdict, deque
from collections.abc import Sequence
from hashlib import sha256
from typing import Protocol
from uuid import UUID

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, ConfigDict, Field

from app.agents.prompts.loader import load_prompt_artifact, render_prompt_input
from app.graphrag.schemas import CommunityRecord, GraphEdge
from app.services.llm.service import LLMService

CONNECTED_COMPONENTS_VERSION = "connected-components-v1"
COMMUNITY_SUMMARY_VERSION = "community-summary-v2"


class CommunityDraft:
    """尚未生成摘要的确定性社区成员集合."""

    def __init__(self, *, community_id: str, member_entity_ids: Sequence[UUID], edges: Sequence[GraphEdge]) -> None:
        """保存不可变排序快照，确保重跑得到稳定版本输入."""
        self.community_id = community_id
        self.member_entity_ids = tuple(sorted(set(member_entity_ids), key=str))
        self.edges = tuple(edges)


class CommunityDetector(Protocol):
    """隔离 GDS、Python fallback 或其他社区算法的应用接口."""

    @property
    def algorithm_version(self) -> str:
        """返回会影响成员划分的算法版本."""
        ...

    def detect(self, edges: Sequence[GraphEdge]) -> tuple[CommunityDraft, ...]:
        """从用户隔离图投影中产生社区成员集合."""
        ...


class ConnectedComponentsDetector:
    """无需 Neo4j GDS 许可的确定性连通分量 fallback."""

    @property
    def algorithm_version(self) -> str:
        """返回 fallback 算法版本，不能伪装成 Leiden."""
        return CONNECTED_COMPONENTS_VERSION

    def detect(self, edges: Sequence[GraphEdge]) -> tuple[CommunityDraft, ...]:
        """按无向连通关系划分社区，并为成员集合生成稳定 ID."""
        adjacency: dict[UUID, set[UUID]] = defaultdict(set)
        edges_by_entity: dict[UUID, list[GraphEdge]] = defaultdict(list)
        for edge in edges:
            adjacency[edge.source_entity_id].add(edge.target_entity_id)
            adjacency[edge.target_entity_id].add(edge.source_entity_id)
            edges_by_entity[edge.source_entity_id].append(edge)
            edges_by_entity[edge.target_entity_id].append(edge)

        visited: set[UUID] = set()
        drafts: list[CommunityDraft] = []
        for root in sorted(adjacency, key=str):
            if root in visited:
                continue
            queue = deque([root])
            members: set[UUID] = set()
            while queue:
                current = queue.popleft()
                if current in visited:
                    continue
                visited.add(current)
                members.add(current)
                queue.extend(sorted(adjacency[current] - visited, key=str))
            component_edges = tuple(
                edge for edge in edges if edge.source_entity_id in members and edge.target_entity_id in members
            )
            digest = sha256("|".join(sorted(str(value) for value in members)).encode()).hexdigest()[:16]
            drafts.append(
                CommunityDraft(
                    community_id=f"community-{digest}",
                    member_entity_ids=tuple(members),
                    edges=component_edges,
                )
            )
        return tuple(drafts)


class _CommunitySummaryPayload(BaseModel):
    """模型只负责标题和摘要，成员与引用仍由服务端绑定."""

    model_config = ConfigDict(frozen=True, extra="forbid", str_strip_whitespace=True, hide_input_in_errors=True)
    title: str = Field(min_length=1, max_length=200)
    summary: str = Field(min_length=1, max_length=4000)


class CommunitySummarizer(Protocol):
    """把一个社区的来源边归纳为可检索摘要."""

    @property
    def model_name(self) -> str:
        """返回生成摘要的可审计模型名称."""
        ...

    async def summarize(self, draft: CommunityDraft) -> _CommunitySummaryPayload:
        """生成摘要，但不能改变成员与来源集合."""
        ...


class LLMCommunitySummarizer:
    """通过真实 structured LLM 生成来源约束的社区摘要."""

    def __init__(self, llm_service: LLMService, *, aliases: Sequence[str], model_name: str) -> None:
        """保存无请求状态的模型服务与 alias 快照."""
        if isinstance(aliases, str) or not aliases:
            raise ValueError("aliases must contain at least one model alias")
        self._llm_service = llm_service
        self._aliases = tuple(aliases)
        self._model_name = model_name

    @property
    def model_name(self) -> str:
        """返回真实 provider 模型名称."""
        return self._model_name

    async def summarize(self, draft: CommunityDraft) -> _CommunitySummaryPayload:
        """只根据给定事实边摘要，不允许补充外部知识."""
        facts = "\n".join(
            f"- {edge.source_name} --{edge.relation_type.value}--> {edge.target_name}: {edge.evidence_text}"
            for edge in draft.edges
        )
        prompt = load_prompt_artifact("graphrag_community_summary")
        return await self._llm_service.call_structured(
            (
                SystemMessage(content=prompt.content),
                HumanMessage(content=render_prompt_input("graphrag_community_summary", facts=facts)),
            ),
            response_model=_CommunitySummaryPayload,
            aliases=self._aliases,
            overrides={"temperature": 0.0},
            prompt=prompt,
        )


class CommunityRepository(Protocol):
    """CommunityService 所需的图仓储能力."""

    async def fetch_edges(self, *, user_id: UUID) -> tuple[GraphEdge, ...]:
        """读取一个用户的来源绑定图边."""
        ...

    async def replace_communities(self, *, user_id: UUID, communities: Sequence[CommunityRecord]) -> None:
        """原子替换用户社区快照."""
        ...


class CommunityService:
    """协调图投影、社区检测、真实摘要和版本化持久化."""

    def __init__(
        self, *, repository: CommunityRepository, detector: CommunityDetector, summarizer: CommunitySummarizer
    ) -> None:
        """注入独立策略，避免业务服务依赖 GDS 或 provider SDK."""
        self._repository = repository
        self._detector = detector
        self._summarizer = summarizer

    async def rebuild(self, *, user_id: UUID) -> tuple[CommunityRecord, ...]:
        """重建用户社区；摘要失败时不覆盖上一版社区快照."""
        edges = await self._repository.fetch_edges(user_id=user_id)
        drafts = self._detector.detect(edges)
        records: list[CommunityRecord] = []
        for draft in drafts:
            payload = await self._summarizer.summarize(draft)
            records.append(
                CommunityRecord(
                    community_id=draft.community_id,
                    member_entity_ids=draft.member_entity_ids,
                    title=payload.title,
                    summary=payload.summary,
                    source_chunk_ids=tuple(dict.fromkeys(edge.source_chunk_id for edge in draft.edges)),
                    algorithm_version=self._detector.algorithm_version,
                    summary_version=COMMUNITY_SUMMARY_VERSION,
                    model_name=self._summarizer.model_name,
                )
            )
        await self._repository.replace_communities(user_id=user_id, communities=records)
        return tuple(records)


__all__ = [
    "CommunityDetector",
    "CommunityDraft",
    "CommunityService",
    "CommunitySummarizer",
    "ConnectedComponentsDetector",
    "LLMCommunitySummarizer",
]
