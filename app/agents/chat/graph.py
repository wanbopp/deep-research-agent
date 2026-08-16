"""Build the minimal chat StateGraph."""

from typing import Protocol
from collections.abc import Awaitable

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.agents.chat.state import ChatState


class ChatNode(Protocol):
    """定义最小聊天图可以接收的 Node 调用约束.

    ChatState是图的数据协议，包含数据和数据约束
    ChatNode Protocol 是节点插槽，Protocol 是表示结构化类型约束，
        这里使用 Protocol，因为未来可以注入两类节点
        只要一个对象可以接收 ChatState，返回同步或异步的 ChatState 状态增量
        它就可以作为 ChatNode,
    """

    def __call__(
        self,
        state: ChatState,
    ) -> ChatState | Awaitable[ChatState]:
        """根据当前状态返回同步或异步的状态增量."""
        ...


def build_chat_graph(
    chat_node: ChatNode,
    *,
    checkpointer: BaseCheckpointSaver[str] | None = None,
) -> CompiledStateGraph:
    """注册单个 chat node 并编译START 到END最小图.

    并按需注入短期状态存储.
    """
    builder = StateGraph(ChatState)

    builder.add_node("chat", chat_node)
    builder.add_edge(START, "chat")
    builder.add_edge("chat", END)

    # checkpointer=none 时保持原来的无状态行为。
    # 调用方注入 InMemorySaver 时，LangGraph 会按 thread_id
    # 保存并恢复每次执行完成后的 ChatState
    return builder.compile(
        checkpointer=checkpointer,
    )
