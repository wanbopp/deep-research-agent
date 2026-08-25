"""Build the chat StateGraph."""

from collections.abc import Awaitable
from typing import Literal, Protocol

from langchain_core.messages import AIMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.runtime import Runtime

from app.agents.chat.context import ChatRuntimeContext
from app.agents.chat.state import ChatState

# CompiledStateGraph 的四个泛型参数依次描述 state、runtime context、图输入和
# 图输出。集中定义别名可以防止 runtime/service 某一层退回裸类型，并错误地把
# context 推断成 None。
type ChatGraph = CompiledStateGraph[
    ChatState,
    ChatRuntimeContext,
    ChatState,
    ChatState,
]


class StateNode(Protocol):
    """定义能够读取可信运行时上下文的聊天图节点接口.

    Protocol 使用结构化类型检查，不要求节点继承某个基类。``runtime`` 必须与
    LangGraph 的 ``_NodeWithRuntime`` 调用约定一致，作为 keyword-only 参数注入；
    这样框架可以调用 ``node(state, runtime=runtime)``，参数名称也是接口的一部分。
    """

    def __call__(
        self,
        state: ChatState,
        *,
        runtime: Runtime[ChatRuntimeContext],
    ) -> ChatState | Awaitable[ChatState]:
        """根据当前状态和本次可信上下文返回同步或异步状态增量."""
        ...


def route_after_chat(state: ChatState) -> Literal["tools", "end"]:
    """根据最后一条 AIMessage 是否包含工具调用决定下一个节点."""
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
) -> ChatGraph:
    """构建聊天图，并按需启用工具循环和短期状态存储.

    Args:
        chat_node: 负责调用模型并返回消息增量的节点。
        tool_node: 可选工具执行节点；不传时构建不含工具循环的最小图。
        checkpointer: 可选状态保存器；生产 runtime 当前默认提供 InMemorySaver。

    Returns:
        已编译的 ChatGraph。它的可变状态是 ChatState，单次执行上下文是
        ChatRuntimeContext，输入和输出也都遵循 ChatState。

    Notes:
        ``context_schema=ChatRuntimeContext`` 这里只声明类型。图在应用启动或首次
        获取 service 时编译；具体 ``ChatRuntimeContext`` 实例要到每次 ainvoke /
        astream 时再传入，不能在共享图上绑定某个用户。
    """
    builder = StateGraph(
        state_schema=ChatState,
        context_schema=ChatRuntimeContext,
    )

    builder.add_node("chat", chat_node)  # chat_node 是一个调用LLM执行一次请求
    builder.add_edge(START, "chat")  # 图的开始节点从这里开始

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
