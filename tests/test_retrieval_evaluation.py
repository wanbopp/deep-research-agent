"""Recall、MRR 与 nDCG 的已知排序样例."""

import pytest

from app.rag.evaluation import evaluate_rankings, ndcg_at_k, recall_at_k, reciprocal_rank


def test_retrieval_metrics_match_known_rankings() -> None:
    """指标必须能区分未召回、首位命中和相关项靠后的排序."""
    relevant = frozenset({"a", "b"})
    ranking = ("x", "a", "b")

    assert recall_at_k(ranking, relevant, k=2) == 0.5
    assert reciprocal_rank(ranking, relevant) == 0.5
    assert ndcg_at_k(("a", "b"), relevant, k=2) == pytest.approx(1.0)

    metrics = evaluate_rankings(((("a", "x"), frozenset({"a"})), (("x", "b"), frozenset({"b"}))), k=2)
    assert metrics.recall_at_k == 1.0
    assert metrics.mrr == 0.75
    assert metrics.ndcg_at_k < 1.0
