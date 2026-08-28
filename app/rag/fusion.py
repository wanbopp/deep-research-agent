"""按 rank 融合不可直接比较的检索通道."""

from collections import defaultdict
from collections.abc import Sequence

from app.rag.retrieval import RetrievalChannel, RetrievedChunk


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[RetrievedChunk]],
    *,
    rank_constant: int = 60,
    top_k: int,
) -> tuple[RetrievedChunk, ...]:
    """使用 RRF ``sum(1 / (k + rank))`` 融合并按 chunk_id 去重."""
    if rank_constant < 0 or top_k <= 0:
        raise ValueError("rank_constant and top_k must be valid")
    scores: dict[object, float] = defaultdict(float)
    canonical: dict[object, RetrievedChunk] = {}
    for ranking in rankings:
        seen: set[object] = set()
        for position, candidate in enumerate(ranking, start=1):
            if candidate.chunk_id in seen:
                continue
            seen.add(candidate.chunk_id)
            canonical.setdefault(candidate.chunk_id, candidate)
            scores[candidate.chunk_id] += 1.0 / (rank_constant + position)
    ordered = sorted(scores, key=lambda chunk_id: (-scores[chunk_id], str(chunk_id)))[:top_k]
    return tuple(
        canonical[chunk_id].with_ranking(
            score=scores[chunk_id],
            rank=rank,
            channel=RetrievalChannel.FUSED,
        )
        for rank, chunk_id in enumerate(ordered, start=1)
    )


__all__ = ["reciprocal_rank_fusion"]
