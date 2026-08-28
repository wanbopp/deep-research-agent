"""并行召回、RRF、可降级精排和可观察计数的 Hybrid Retriever."""

import asyncio
from typing import Protocol
from uuid import UUID

from app.core.logging import logger
from app.rag.fusion import reciprocal_rank_fusion
from app.rag.reranker import Reranker
from app.rag.retrieval import RetrievedChunk


class SearchChannel(Protocol):
    """dense 与 sparse 检索器共享的调用形状."""

    async def search(
        self,
        *,
        user_id: UUID,
        query: str,
        top_k: int,
        document_ids: frozenset[UUID] | None = None,
    ) -> tuple[RetrievedChunk, ...]:
        """返回已在存储层执行 owner 过滤的有序候选."""
        ...


class HybridRetriever:
    """并行执行 dense/sparse，按 rank 融合后进行小候选精排."""

    def __init__(self, *, dense: SearchChannel, sparse: SearchChannel, reranker: Reranker) -> None:
        """注入三个独立阶段，便于替换 provider 或执行降级."""
        self._dense = dense
        self._sparse = sparse
        self._reranker = reranker

    async def search(
        self,
        *,
        user_id: UUID,
        query: str,
        candidate_k: int,
        final_k: int,
        document_ids: frozenset[UUID] | None = None,
    ) -> tuple[RetrievedChunk, ...]:
        """返回精排结果；reranker 失败时安全降级到 RRF，不丢弃召回结果."""
        dense, sparse = await asyncio.gather(
            self._dense.search(user_id=user_id, query=query, top_k=candidate_k, document_ids=document_ids),
            self._sparse.search(user_id=user_id, query=query, top_k=candidate_k, document_ids=document_ids),
        )
        fused = reciprocal_rank_fusion((dense, sparse), top_k=candidate_k)
        try:
            result = await self._reranker.rerank(query=query, candidates=fused, top_k=final_k)
        except Exception as error:
            logger.warning("rag_reranker_failed", error_type=type(error).__name__)
            result = fused[:final_k]
        logger.info(
            "rag_retrieval_completed",
            dense_count=len(dense),
            sparse_count=len(sparse),
            fused_count=len(fused),
            reranked_count=len(result),
        )
        return result


__all__ = ["HybridRetriever", "SearchChannel"]
