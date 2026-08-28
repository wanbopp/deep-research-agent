"""BM25、RRF、rerank 与引用预算的确定性门禁."""

from uuid import UUID

import pytest

from app.rag.bm25_store import BM25Retriever
from app.rag.context import ContextAssembler
from app.rag.fusion import reciprocal_rank_fusion
from app.rag.hybrid import HybridRetriever
from app.rag.reranker import TokenOverlapReranker
from app.rag.retrieval import RetrievalChannel, RetrievedChunk

USER_ID = UUID("11111111-1111-4111-8111-111111111111")
DOCUMENT_ID = UUID("22222222-2222-4222-8222-222222222222")


def _chunk(number: int, text: str, *, channel: RetrievalChannel, rank: int) -> RetrievedChunk:
    """构造只含公开检索字段的固定候选，不使用 embedding 或数据库内部状态."""
    return RetrievedChunk(
        chunk_id=UUID(f"00000000-0000-4000-8000-{number:012d}"),
        document_id=DOCUMENT_ID,
        text=text,
        score=1.0 / rank,
        rank=rank,
        channel=channel,
        source_locations=({"page_number": 1, "section_path": ["事实"]},),
    )


class _Catalog:
    """测试 BM25 算法的固定 owner-scoped 候选源."""

    def __init__(self, candidates: tuple[RetrievedChunk, ...]) -> None:
        self._candidates = candidates

    async def list_for_search(
        self,
        *,
        user_id: UUID,
        document_ids: frozenset[UUID] | None,
    ) -> tuple[RetrievedChunk, ...]:
        assert user_id == USER_ID
        return self._candidates if document_ids is None or DOCUMENT_ID in document_ids else ()


class _Channel:
    """只验证 Hybrid 编排顺序的固定已过滤通道."""

    def __init__(self, results: tuple[RetrievedChunk, ...]) -> None:
        self._results = results

    async def search(
        self,
        *,
        user_id: UUID,
        query: str,
        top_k: int,
        document_ids: frozenset[UUID] | None = None,
    ) -> tuple[RetrievedChunk, ...]:
        del query, document_ids
        assert user_id == USER_ID
        return self._results[:top_k]


@pytest.mark.anyio
async def test_bm25_rrf_rerank_and_context_keep_identity_and_citations() -> None:
    """精确词、语义候选、去重、重排和 token 预算应形成稳定闭环."""
    exact = _chunk(1, "设备型号 ZX-9000 的额定功率是 1200W", channel=RetrievalChannel.SPARSE, rank=1)
    semantic = _chunk(2, "这台设备在满负载时消耗一点二千瓦", channel=RetrievalChannel.DENSE, rank=1)
    distractor = _chunk(3, "普通办公设备说明", channel=RetrievalChannel.DENSE, rank=2)

    sparse = await BM25Retriever(_Catalog((exact, semantic, distractor))).search(
        user_id=USER_ID,
        query="ZX-9000 功率",
        top_k=3,
    )
    assert sparse[0].chunk_id == exact.chunk_id

    fused = reciprocal_rank_fusion(((semantic, exact), (exact, distractor)), top_k=3)
    assert fused[0].chunk_id == exact.chunk_id
    assert len({item.chunk_id for item in fused}) == len(fused)

    hybrid = HybridRetriever(
        dense=_Channel((semantic, distractor)),
        sparse=_Channel((exact,)),
        reranker=TokenOverlapReranker(),
    )
    reranked = await hybrid.search(
        user_id=USER_ID,
        query="ZX-9000 功率",
        candidate_k=3,
        final_k=2,
    )
    assert reranked[0].chunk_id == exact.chunk_id

    context = ContextAssembler(max_tokens=60).assemble(reranked)
    assert context.citations
    assert len({item.citation_id for item in context.citations}) == len(context.citations)
    assert all(f"[{item.citation_id}]" in context.text for item in context.citations)
    assert context.token_count <= 60
