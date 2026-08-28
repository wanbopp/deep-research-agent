"""校验 GraphRAG 固定数据集上的路由、术语和引用覆盖."""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.graphrag.evaluation import citation_coverage


@dataclass(frozen=True, slots=True)
class EvaluationRow:
    """一条固定问题及其最小期望."""

    item_id: str
    category: str
    expected_route: str
    required_terms: tuple[str, ...]


def _load_jsonl(path: Path) -> tuple[dict[str, Any], ...]:
    """读取非空 JSONL；格式错误由调用者直接看到并修复数据集."""
    rows = tuple(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    if not rows:
        raise ValueError("evaluation jsonl must not be empty")
    return rows


def evaluate(*, dataset_path: Path, predictions_path: Path) -> dict[str, object]:
    """按 id 对齐预测，计算路由准确率、术语覆盖和引用覆盖."""
    dataset = {
        row["id"]: EvaluationRow(
            item_id=row["id"],
            category=row["category"],
            expected_route=row["expected_route"],
            required_terms=tuple(row["required_terms"]),
        )
        for row in _load_jsonl(dataset_path)
    }
    predictions = {row["id"]: row for row in _load_jsonl(predictions_path)}
    if set(predictions) != set(dataset):
        raise ValueError("prediction ids must exactly match dataset ids")

    route_hits = term_scores = citation_scores = 0.0
    for item_id, expected in dataset.items():
        prediction = predictions[item_id]
        route_hits += prediction["route"] == expected.expected_route
        answer = str(prediction["answer"]).casefold()
        term_scores += sum(term.casefold() in answer for term in expected.required_terms) / len(
            expected.required_terms
        )
        required_chunks = frozenset(str(value) for value in prediction["required_chunk_ids"])
        citation_scores += citation_coverage(
            tuple(str(value) for value in prediction["cited_chunk_ids"]),
            required_chunks,
        )

    count = len(dataset)
    return {
        "query_count": count,
        "route_accuracy": route_hits / count,
        "required_term_coverage": term_scores / count,
        "citation_coverage": citation_scores / count,
    }


def main() -> None:
    """运行离线报告；真实 predictions 由 smoke 或评测任务生成."""
    parser = argparse.ArgumentParser()
    parser.add_argument("predictions", type=Path)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(__file__).with_name("dataset.jsonl"),
    )
    args = parser.parse_args()
    print(json.dumps(evaluate(dataset_path=args.dataset, predictions_path=args.predictions), ensure_ascii=False))


if __name__ == "__main__":
    main()
