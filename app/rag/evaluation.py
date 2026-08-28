"""检索质量的可重复离线指标与配置比较."""

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass


def recall_at_k(ranking: Sequence[str], relevant: frozenset[str], *, k: int) -> float:
    """计算前 k 个结果覆盖了多少 ground-truth 相关项."""
    if k <= 0 or not relevant:
        raise ValueError("k and relevant set must be non-empty")
    return len(set(ranking[:k]) & relevant) / len(relevant)


def reciprocal_rank(ranking: Sequence[str], relevant: frozenset[str]) -> float:
    """返回第一个相关结果名次的倒数；未召回则为 0."""
    for rank, item in enumerate(ranking, start=1):
        if item in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(ranking: Sequence[str], relevant: frozenset[str], *, k: int) -> float:
    """使用二元相关性计算 nDCG@k，评价多个相关结果的整体排序."""
    if k <= 0 or not relevant:
        raise ValueError("k and relevant set must be non-empty")
    dcg = sum(1.0 / math.log2(rank + 1) for rank, item in enumerate(ranking[:k], start=1) if item in relevant)
    ideal_count = min(k, len(relevant))
    ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_count + 1))
    return dcg / ideal


@dataclass(frozen=True, slots=True)
class RetrievalMetrics:
    """一种检索配置在固定数据集上的平均指标."""

    recall_at_k: float
    mrr: float
    ndcg_at_k: float
    query_count: int


def evaluate_rankings(
    rows: Iterable[tuple[Sequence[str], frozenset[str]]],
    *,
    k: int,
) -> RetrievalMetrics:
    """对多条查询取宏平均，防止长候选查询支配结果."""
    materialized = tuple(rows)
    if not materialized:
        raise ValueError("evaluation rows must not be empty")
    recalls = [recall_at_k(ranking, relevant, k=k) for ranking, relevant in materialized]
    reciprocal_ranks = [reciprocal_rank(ranking, relevant) for ranking, relevant in materialized]
    ndcgs = [ndcg_at_k(ranking, relevant, k=k) for ranking, relevant in materialized]
    count = len(materialized)
    return RetrievalMetrics(
        recall_at_k=sum(recalls) / count,
        mrr=sum(reciprocal_ranks) / count,
        ndcg_at_k=sum(ndcgs) / count,
        query_count=count,
    )


__all__ = ["RetrievalMetrics", "evaluate_rankings", "ndcg_at_k", "recall_at_k", "reciprocal_rank"]
