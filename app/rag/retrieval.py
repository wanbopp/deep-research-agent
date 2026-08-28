"""Hybrid RAG 各检索阶段共享的结果契约."""

from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Any
from uuid import UUID


class RetrievalChannel(StrEnum):
    """候选结果来自哪条检索或融合路径."""

    DENSE = "dense"
    SPARSE = "sparse"
    FUSED = "fused"
    RERANKED = "reranked"


@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    """可追溯、可排序但不暴露 embedding 的统一候选结果."""

    chunk_id: UUID
    document_id: UUID
    text: str
    score: float
    rank: int
    channel: RetrievalChannel
    source_locations: tuple[dict[str, Any], ...]

    def with_ranking(self, *, score: float, rank: int, channel: RetrievalChannel) -> "RetrievedChunk":
        """返回新排名视图，不修改其他检索器仍在使用的原候选对象."""
        return replace(self, score=score, rank=rank, channel=channel)


__all__ = ["RetrievalChannel", "RetrievedChunk"]
