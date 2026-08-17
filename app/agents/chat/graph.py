"""Build the chat StateGraph."""

from collections.abc import Awaitable
from typing import Literal, Protocol

from langchain_core.messages import AIMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.agents.chat.state import ChatState


class StateNode(Protocol):
    """定义聊天图可接受的同步或异步节点调用形状.

    Protocol 使用结构化类型检查。任何接收 ChatState，并返回
    ChatState 状态增量或其 Awaitable 的可调用对象，都可以作为图节点。
    """

    def __call__(
        self,
        state: ChatState,
    ) -> ChatState | Awaitable[ChatState]:
        """根据当前状态返回同步或异步的状态增量."""
        ...


def route_after_chat(state: ChatState) -> Literal["tools", "end"]:
    """根据最后一条 AIMessage 是否包含工具调用决定下一节点."""
    messages = state["messages"]
    if not messages:
        raise ValueError("chat route requires at least one message")

    # 条件边在 chat 节点写入状态之后执行，因此最后一条消息
    # 必须是 chat 节点刚刚返回的 AIMessage。
    last_message = messages[-1]
    if not isinstance(last_message, AIMessage):
        raise TypeError("chat route requires the last message to be an AIMessage")

    return "tools" if last_message.tool_calls else "end"


def build_chat_graph(
    chat_node: StateNode,
    *,
    tool_node: StateNode | None = None,
    checkpointer: BaseCheckpointSaver[str] | None = None,
) -> CompiledStateGraph:
    """构建聊天图，并按需启用工具循环和短期状态存储."""
    builder = StateGraph(ChatState)

    builder.add_node("chat", chat_node)
    builder.add_edge(START, "chat")

    if tool_node is None:
        # 未注入工具节点时保留 Lab 08 的最小聊天图：
        # START -> chat -> END。
        builder.add_edge("chat", END)
    else:
        builder.add_node("tools", tool_node)

        # chat 返回工具调用时进入 tools；否则结束本轮图执行。
        builder.add_conditional_edges(
            "chat",
            route_after_chat,
            {
                "tools": "tools",
                "end": END,
            },
        )

        # 工具结果写入 ToolMessage 后必须回到 chat，
        # 让模型读取工具结果并决定继续调用工具还是生成最终答案。
        builder.add_edge("tools", "chat")

    # checkpointer=None 时保持无持久状态行为；调用方注入
    # InMemorySaver 等实现后，LangGraph 会按 thread_id 保存状态。
    return builder.compile(
        checkpointer=checkpointer,
    )
