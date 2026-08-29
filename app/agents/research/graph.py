"""构建有硬停止条件的 DeepResearch LangGraph."""

from collections.abc import Awaitable
from typing import Literal, Protocol

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.runtime import Runtime

from app.agents.research.context import ResearchRuntimeContext
from app.agents.research.state import ResearchState, ResearchStateUpdate
from app.schemas.research import ResearchStatus

type ResearchGraph = CompiledStateGraph[
    ResearchState,
    ResearchRuntimeContext,
    ResearchState,
    ResearchState,
]


class ResearchNode(Protocol):
    """研究节点共享的调用形状，允许同步或异步实现."""

    def __call__(
        self,
        state: ResearchState,
        *,
        runtime: Runtime[ResearchRuntimeContext],
    ) -> ResearchStateUpdate | Awaitable[ResearchStateUpdate]:
        """读取完整状态和可信上下文，返回本节点负责的局部更新."""
        ...


def route_after_planner(state: ResearchState) -> Literal["retrieve", "end"]:
    """模糊问题停在澄清状态，计划可执行时才进入查找."""
    return "retrieve" if state["status"] == ResearchStatus.RESEARCHING.value else "end"


def route_after_validation(state: ResearchState) -> Literal["retrieve", "write", "end"]:
    """根据验证结果补查、写报告或以明确不充分状态结束."""
    if state["status"] == ResearchStatus.RESEARCHING.value:
        return "retrieve"
    if state["status"] in {
        ResearchStatus.WRITING.value,
        ResearchStatus.INSUFFICIENT_EVIDENCE.value,
        ResearchStatus.BUDGET_EXHAUSTED.value,
    }:
        return "write"
    return "end"


def build_research_graph(
    *,
    planner: ResearchNode,
    retriever: ResearchNode,
    validator: ResearchNode,
    writer: ResearchNode,
    checkpointer: BaseCheckpointSaver[str] | None = None,
) -> ResearchGraph:
    """编译 Planner、检索、验证循环和 Writer 的完整研究图.

    循环并不意味着无限运行。Validator 每次补查都会增加 current_iteration，达到
    ResearchConfig.max_iterations 后把状态改成 insufficient_evidence，条件边随后
    进入 END。进程重启时 checkpoint 保存的计数仍然有效。
    """
    builder = StateGraph(state_schema=ResearchState, context_schema=ResearchRuntimeContext)
    builder.add_node("planner", planner)
    builder.add_node("retrieve", retriever)
    builder.add_node("validate", validator)
    builder.add_node("write", writer)

    builder.add_edge(START, "planner")
    builder.add_conditional_edges("planner", route_after_planner, {"retrieve": "retrieve", "end": END})
    builder.add_edge("retrieve", "validate")
    builder.add_conditional_edges(
        "validate",
        route_after_validation,
        {"retrieve": "retrieve", "write": "write", "end": END},
    )
    builder.add_edge("write", END)
    return builder.compile(checkpointer=checkpointer)


__all__ = ["ResearchGraph", "ResearchNode", "build_research_graph"]
