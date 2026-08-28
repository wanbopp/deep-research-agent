"""读取版本化检索数据集并生成四种配置的 JSON 指标报告."""

import argparse
from datetime import UTC, datetime
from hashlib import sha256
import json
from pathlib import Path
import subprocess
from typing import Any

from app.rag.evaluation import evaluate_rankings

CONFIGURATIONS = ("vector", "bm25", "hybrid", "hybrid_rerank")


def _git_revision(project_root: Path) -> str:
    """记录代码版本；无 Git 环境时使用 stable unknown，不让评测直接失败."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def run(dataset_path: Path, output_path: Path, *, k: int) -> dict[str, Any]:
    """计算报告并原子写文件，避免中断后留下半个 JSON."""
    raw = dataset_path.read_bytes()
    rows = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    metrics: dict[str, Any] = {}
    for configuration in CONFIGURATIONS:
        evaluated = evaluate_rankings(
            ((row["rankings"][configuration], frozenset(row["relevant_chunk_ids"])) for row in rows),
            k=k,
        )
        metrics[configuration] = {
            "recall_at_k": round(evaluated.recall_at_k, 6),
            "mrr": round(evaluated.mrr, 6),
            "ndcg_at_k": round(evaluated.ndcg_at_k, 6),
            "query_count": evaluated.query_count,
        }
    project_root = Path(__file__).resolve().parents[2]
    report = {
        "generated_at": datetime.now(UTC).isoformat(),
        "code_revision": _git_revision(project_root),
        "dataset_sha256": sha256(raw).hexdigest(),
        "k": k,
        "configurations": metrics,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output_path)
    return report


def main() -> None:
    """解析 CLI 参数并打印不含语料正文的报告摘要."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path(__file__).with_name("dataset.jsonl"))
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "reports" / "latest.json")
    parser.add_argument("--k", type=int, default=3)
    args = parser.parse_args()
    report = run(args.dataset, args.output, k=args.k)
    print(json.dumps({"ok": True, **report}, ensure_ascii=False))


if __name__ == "__main__":
    main()
