"""Global GraphRAG 的真实 LLM map-reduce 回答链."""

import asyncio
from collections.abc import Sequence

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, ConfigDict, Field

from app.graphrag.schemas import (
    CommunityMapResult,
    CommunityRecord,
    GlobalGraphAnswer,
    GlobalGraphResult,
)
from app.services.llm.service import LLMService


class _MapPayload(BaseModel):
    """模型只生成社区局部结论，引用由服务端绑定."""

    model_config = ConfigDict(frozen=True, extra="forbid", str_strip_whitespace=True, hide_input_in_errors=True)
    claim: str = Field(min_length=1, max_length=3000)


class _ReducePayload(BaseModel):
    """reduce 模型生成答案并选择它实际使用的 map 下标."""

    model_config = ConfigDict(frozen=True, extra="forbid", str_strip_whitespace=True, hide_input_in_errors=True)
    answer: str = Field(min_length=1, max_length=8000)
    used_map_indexes: tuple[int, ...] = Field(min_length=1)


class GlobalGraphAnswerer:
    """并发 map 相关社区，再以受控索引执行一次 reduce."""

    def __init__(self, llm_service: LLMService, *, aliases: Sequence[str], max_parallel_maps: int = 4) -> None:
        """保存真实模型边界和 map 并发预算."""
        if isinstance(aliases, str) or not aliases:
            raise ValueError("aliases must contain at least one model alias")
        if max_parallel_maps <= 0:
            raise ValueError("max_parallel_maps must be greater than zero")
        self._llm_service = llm_service
        self._aliases = tuple(aliases)
        self._semaphore = asyncio.Semaphore(max_parallel_maps)

    async def answer(self, *, query: str, retrieval: GlobalGraphResult) -> GlobalGraphAnswer:
        """只使用检索到的社区摘要回答，并从选中 map 结果聚合引用."""
        if not query.strip():
            raise ValueError("query must not be blank")
        if not retrieval.communities:
            raise ValueError("global retrieval must contain at least one community")
        map_results = tuple(
            await asyncio.gather(
                *(self._map_one(query=query, community=community) for community in retrieval.communities)
            )
        )
        rendered = "\n".join(f"[{index}] {item.claim}" for index, item in enumerate(map_results))
        reduced = await self._llm_service.call_structured(
            (
                SystemMessage(
                    content=(
                        "Answer only from the numbered map claims. Select every claim index used. "
                        "Do not invent sources or use outside knowledge."
                    )
                ),
                HumanMessage(content=f"Question: {query}\nClaims:\n{rendered}"),
            ),
            response_model=_ReducePayload,
            aliases=self._aliases,
            overrides={"temperature": 0.0},
        )
        indexes = tuple(dict.fromkeys(reduced.used_map_indexes))
        if any(index < 0 or index >= len(map_results) for index in indexes):
            raise ValueError("reduce selected an unknown map result")
        source_chunk_ids = tuple(
            dict.fromkeys(chunk_id for index in indexes for chunk_id in map_results[index].source_chunk_ids)
        )
        return GlobalGraphAnswer(
            answer=reduced.answer,
            source_chunk_ids=source_chunk_ids,
            map_results=map_results,
        )

    async def _map_one(self, *, query: str, community: CommunityRecord) -> CommunityMapResult:
        """在并发预算内生成一个社区局部结论并绑定真实来源."""
        async with self._semaphore:
            payload = await self._llm_service.call_structured(
                (
                    SystemMessage(
                        content=(
                            "Answer the question only from this community summary. If evidence is indirect, "
                            "state that limitation. Do not add citations, ids, or outside knowledge."
                        )
                    ),
                    HumanMessage(content=f"Question: {query}\nCommunity: {community.summary}"),
                ),
                response_model=_MapPayload,
                aliases=self._aliases,
                overrides={"temperature": 0.0},
            )
        return CommunityMapResult(
            community_id=community.community_id,
            claim=payload.claim,
            source_chunk_ids=community.source_chunk_ids,
        )


__all__ = ["GlobalGraphAnswerer"]
