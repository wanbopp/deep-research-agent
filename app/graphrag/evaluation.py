"""GraphRAG 链接、路径和引用覆盖率的确定性指标."""

from collections.abc import Sequence
from dataclasses import dataclass


def entity_linking_accuracy(predicted: Sequence[str], expected: Sequence[str]) -> float:
    """按集合精确命中计算实体链接准确率."""
    if not expected:
        raise ValueError("expected entity ids must not be empty")
    return len(set(predicted) & set(expected)) / len(set(expected))


def path_relevance_at_k(ranked_path_ids: Sequence[str], relevant_path_ids: frozenset[str], *, k: int) -> float:
    """计算前 k 条路径中相关路径占比."""
    if k <= 0 or not relevant_path_ids:
        raise ValueError("k and relevant path ids must be non-empty")
    top = ranked_path_ids[:k]
    return sum(item in relevant_path_ids for item in top) / min(k, len(top)) if top else 0.0


def citation_coverage(cited_chunk_ids: Sequence[str], required_chunk_ids: frozenset[str]) -> float:
    """回答引用覆盖了多少必须出现的原始 chunk."""
    if not required_chunk_ids:
        raise ValueError("required chunk ids must not be empty")
    return len(set(cited_chunk_ids) & required_chunk_ids) / len(required_chunk_ids)


@dataclass(frozen=True, slots=True)
class GraphRAGMetrics:
    """一种 GraphRAG 配置在固定数据集上的汇总指标."""

    entity_linking_accuracy: float
    path_relevance_at_k: float
    citation_coverage: float
    query_count: int


__all__ = [
    "GraphRAGMetrics",
    "citation_coverage",
    "entity_linking_accuracy",
    "path_relevance_at_k",
]
