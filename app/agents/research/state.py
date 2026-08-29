"""DeepResearch 图的持久状态和节点局部更新规则."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Annotated, NotRequired, TypedDict, TypeVar

ValueT = TypeVar("ValueT")
type JsonObject = dict[str, object]


def append_values(current: Sequence[ValueT], update: Sequence[ValueT]) -> tuple[ValueT, ...]:
    """把并行节点的增量按到达顺序追加为不可变元组."""
    return (*current, *update)


def merge_evidence(current: Sequence[JsonObject], update: Sequence[JsonObject]) -> tuple[JsonObject, ...]:
    """按 evidence_id 去重合并并行结果，保留首次出现的可信内容.

    并行查找的完成顺序并不固定，因此业务逻辑不能依赖列表到达顺序。相同来源
    可能同时被文字搜索和图搜索找到；用稳定 ID 去重可避免报告把同一证据算两次。
    """
    merged = {str(item["evidence_id"]): item for item in current}
    for item in update:
        merged.setdefault(str(item["evidence_id"]), item)

    def sort_key(item: JsonObject) -> tuple[str, float, str]:
        """读取已校验证据的排序字段，并守住持久化数据的运行时边界."""
        score = item["score"]
        if not isinstance(score, int | float):
            raise TypeError("serialized evidence score must be numeric")
        return str(item["step_id"]), -float(score), str(item["evidence_id"])

    return tuple(sorted(merged.values(), key=sort_key))


class ResearchState(TypedDict):
    """可被 LangGraph checkpoint 保存并在进程重启后恢复的研究状态.

    这里只放 JSON 可表达的数据，不放数据库连接、LLMService、检索器或 Session。
    甚至 Pydantic 模型也不直接放进来：checkpoint 可能在新进程、新版本中恢复，
    保存普通字典比保存 Python 类实例更稳定。节点入口再用 ``model_validate`` 把
    字典恢复成严格模型，因此“持久化格式简单”和“业务使用有类型”可以同时满足。
    """

    topic: str
    config: JsonObject
    status: str
    plan: NotRequired[JsonObject]
    current_iteration: int
    evidence: Annotated[tuple[JsonObject, ...], merge_evidence]
    retrieval_failures: Annotated[tuple[JsonObject, ...], append_values]
    validation: NotRequired[JsonObject]
    report: NotRequired[JsonObject]
    stop_reason: NotRequired[str]


class ResearchStateUpdate(TypedDict, total=False):
    """单个节点返回的状态增量；合并方式由 ResearchState 字段声明."""

    status: str
    plan: JsonObject
    current_iteration: int
    evidence: tuple[JsonObject, ...]
    retrieval_failures: tuple[JsonObject, ...]
    validation: JsonObject
    report: JsonObject
    stop_reason: str


__all__ = ["JsonObject", "ResearchState", "ResearchStateUpdate", "append_values", "merge_evidence"]
