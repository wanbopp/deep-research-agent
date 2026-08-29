"""DeepResearch 图的持久状态和节点局部更新规则."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Annotated, NotRequired, TypedDict, TypeVar

from app.schemas.research import (
    Evidence,
    ResearchConfig,
    ResearchPlan,
    ResearchReport,
    ResearchStatus,
    RetrievalFailure,
    ValidationResult,
)

ValueT = TypeVar("ValueT")


def append_values(current: Sequence[ValueT], update: Sequence[ValueT]) -> tuple[ValueT, ...]:
    """把并行节点的增量按到达顺序追加为不可变元组."""
    return (*current, *update)


def merge_evidence(current: Sequence[Evidence], update: Sequence[Evidence]) -> tuple[Evidence, ...]:
    """按 evidence_id 去重合并并行结果，保留首次出现的可信内容.

    并行查找的完成顺序并不固定，因此业务逻辑不能依赖列表到达顺序。相同来源
    可能同时被文字搜索和图搜索找到；用稳定 ID 去重可避免报告把同一证据算两次。
    """
    merged = {item.evidence_id: item for item in current}
    for item in update:
        merged.setdefault(item.evidence_id, item)
    return tuple(sorted(merged.values(), key=lambda item: (item.step_id, -item.score, item.evidence_id)))


class ResearchState(TypedDict):
    """可被 LangGraph checkpoint 保存并在进程重启后恢复的研究状态.

    这里只放数据，不放数据库连接、LLMService、检索器或 Session。那些对象无法
    稳定序列化，也不属于一次研究任务的业务事实，应由 runtime/node 闭包注入。
    """

    topic: str
    config: ResearchConfig
    status: ResearchStatus
    plan: NotRequired[ResearchPlan]
    current_iteration: int
    evidence: Annotated[tuple[Evidence, ...], merge_evidence]
    retrieval_failures: Annotated[tuple[RetrievalFailure, ...], append_values]
    validation: NotRequired[ValidationResult]
    report: NotRequired[ResearchReport]
    stop_reason: NotRequired[str]


class ResearchStateUpdate(TypedDict, total=False):
    """单个节点返回的状态增量；合并方式由 ResearchState 字段声明."""

    status: ResearchStatus
    plan: ResearchPlan
    current_iteration: int
    evidence: tuple[Evidence, ...]
    retrieval_failures: tuple[RetrievalFailure, ...]
    validation: ValidationResult
    report: ResearchReport
    stop_reason: str


__all__ = ["ResearchState", "ResearchStateUpdate", "append_values", "merge_evidence"]
