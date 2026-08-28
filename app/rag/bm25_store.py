"""可替换分词策略的 owner-scoped BM25 稀疏检索."""

import re
from typing import Protocol
from uuid import UUID

from rank_bm25 import BM25Okapi

from app.rag.retrieval import RetrievalChannel, RetrievedChunk

_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_.:/-]+|[\u3400-\u9fff]")


class SparseCandidateSource(Protocol):
    """提供已经在存储层按 owner/document 过滤的候选语料."""

    async def list_for_search(
        self,
        *,
        user_id: UUID,
        document_ids: frozenset[UUID] | None,
    ) -> tuple[RetrievedChunk, ...]:
        """返回当前用户可见的 chunks，不得返回其他 namespace 数据."""
        ...


def tokenize_for_bm25(text: str) -> list[str]:
    """提取英文标识符和中文字符，并补充中文二元词以改善短语匹配."""
    base = [token.lower() for token in _TOKEN_PATTERN.findall(text)]
    cjk = [token for token in base if len(token) == 1 and "\u3400" <= token <= "\u9fff"]
    return base + [left + right for left, right in zip(cjk, cjk[1:], strict=False)]


class BM25Retriever:
    """在 owner-scoped 候选集上构建轻量 BM25 索引并返回稀疏结果."""

    def __init__(self, source: SparseCandidateSource) -> None:
        """保存候选源；生产规模扩大后可替换成外部稀疏索引而不改融合层."""
        self._source = source

    async def search(
        self,
        *,
        user_id: UUID,
        query: str,
        top_k: int,
        document_ids: frozenset[UUID] | None = None,
    ) -> tuple[RetrievedChunk, ...]:
        """使用 BM25 原始分数排序；该分数只在 sparse 通道内部比较."""
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")
        query_tokens = tokenize_for_bm25(query)
        if not query_tokens:
            return ()
        candidates = await self._source.list_for_search(user_id=user_id, document_ids=document_ids)
        if not candidates:
            return ()
        corpus = [tokenize_for_bm25(candidate.text) for candidate in candidates]
        scores = BM25Okapi(corpus).get_scores(query_tokens)
        ranked = sorted(zip(candidates, scores, strict=True), key=lambda item: float(item[1]), reverse=True)
        return tuple(
            candidate.with_ranking(
                score=float(score),
                rank=rank,
                channel=RetrievalChannel.SPARSE,
            )
            for rank, (candidate, score) in enumerate(ranked[:top_k], start=1)
            if float(score) > 0
        )


__all__ = ["BM25Retriever", "SparseCandidateSource", "tokenize_for_bm25"]
