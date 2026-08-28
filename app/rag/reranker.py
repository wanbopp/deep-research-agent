"""候选精排协议、确定性本地实现与失败降级."""

from collections.abc import Sequence
from typing import Protocol

from app.rag.bm25_store import tokenize_for_bm25
from app.rag.retrieval import RetrievalChannel, RetrievedChunk


class Reranker(Protocol):
    """只在小候选集上重新排序，不负责召回或权限过滤."""

    async def rerank(
        self,
        *,
        query: str,
        candidates: Sequence[RetrievedChunk],
        top_k: int,
    ) -> tuple[RetrievedChunk, ...]:
        """返回不超过 top_k 个、chunk_id 不变的重排候选."""
        ...


class TokenOverlapReranker:
    """无需模型的本地基线：按查询 token 覆盖率精排候选."""

    async def rerank(
        self,
        *,
        query: str,
        candidates: Sequence[RetrievedChunk],
        top_k: int,
    ) -> tuple[RetrievedChunk, ...]:
        """计算可解释的词项覆盖分数，并用原融合 rank 稳定打破平局."""
        query_tokens = set(tokenize_for_bm25(query))
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")
        scored = []
        for candidate in candidates:
            candidate_tokens = set(tokenize_for_bm25(candidate.text))
            overlap = len(query_tokens & candidate_tokens) / max(1, len(query_tokens))
            scored.append((candidate, overlap))
        scored.sort(key=lambda item: (-item[1], item[0].rank, str(item[0].chunk_id)))
        return tuple(
            candidate.with_ranking(
                score=score,
                rank=rank,
                channel=RetrievalChannel.RERANKED,
            )
            for rank, (candidate, score) in enumerate(scored[:top_k], start=1)
        )


__all__ = ["Reranker", "TokenOverlapReranker"]
