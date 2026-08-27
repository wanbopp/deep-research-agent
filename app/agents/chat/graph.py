"""构建支持可选长期记忆、工具循环和 checkpoint 的聊天 StateGraph."""

from collections.abc import Awaitable
from typing import Literal, Protocol

from langchain_core.messages import AIMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.runtime import Runtime

from app.agents.chat.context import ChatRuntimeContext
from app.agents.chat.state import ChatState, ChatStateUpdate

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
    ) -> ChatStateUpdate | Awaitable[ChatStateUpdate]:
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


def route_after_tools(state: ChatState) -> Literal["chat", "memory"]:
    """根据当前 invocation 是否仍有完整记忆上下文决定工具后的去向.

    ``UntrackedValue`` 在同一次 invocation 的多个节点之间持续可用，所以普通
    ``chat -> tools -> chat`` 循环可以直接复用检索结果。HITL 或进程恢复会从
    checkpoint 创建新的 invocation；由于临时记忆从不写入 checkpoint，此时
    tools 完成后必须先回到 memory node 补查一次。

    Notes:
        这里判断字段是否存在，而不是判断 ``memory_context`` 是否为真。空元组
        同样表示 memory node 已正常执行，只是没有找到相关记忆；若按真值判断，
        每次正常空结果都会造成不必要的重复检索。
    """
    has_complete_memory_result = "memory_context" in state and "memory_status" in state

    return "chat" if has_complete_memory_result else "memory"


def build_chat_graph(
    chat_node: StateNode,
    *,
    memory_node: StateNode | None = None,
    tool_node: StateNode | None = None,
    checkpointer: BaseCheckpointSaver[str] | None = None,
) -> ChatGraph:
    """构建聊天图，并按需启用工具循环和短期状态存储.

    Args:
        chat_node: 负责调用模型并返回消息增量的节点。
        memory_node: 可选长期记忆检索节点。提供后，Graph 从 memory 开始；不提供
            时保留原有 ``START -> chat`` 行为，供旧 smoke 和无记忆场景使用。
        tool_node: 可选工具执行节点；不传时构建不含工具循环的最小图。
        checkpointer: 可选状态保存器。它保存消息和节点进度，但不会保存声明为
            ``UntrackedValue`` 的当前检索结果。

    Returns:
        已编译的 ChatGraph。它的可变状态是 ChatState，单次执行上下文是
        ChatRuntimeContext，输入和输出也都遵循 ChatState。

    Notes:
        ``context_schema=ChatRuntimeContext`` 这里只声明类型。图在应用启动或首次
        获取 service 时编译；具体 ``ChatRuntimeContext`` 实例要到每次 ainvoke /
        astream 时再传入，不能在共享图上绑定某个用户。

        构建器支持三种主要形态：仅 chat、memory + chat，以及
        memory + chat + tools。可选参数使历史 smoke 不必为了测试其他能力而构造
        MemoryService，同时 production runtime 可以显式启用完整记忆链。
    """
    builder = StateGraph(
        state_schema=ChatState,
        context_schema=ChatRuntimeContext,
    )

    # chat 是所有图形态都必须存在的核心节点；memory 和 tools 则按依赖注入决定。
    builder.add_node("chat", chat_node)

    if memory_node is None:
        # 未启用长期记忆时保持原入口，避免改变历史最小图和工具 smoke 的行为。
        builder.add_edge(START, "chat")
    else:
        # memory 只在每次 invocation 的入口执行；普通工具循环不会重新经过 START。
        builder.add_node("memory", memory_node)
        builder.add_edge(START, "memory")
        builder.add_edge("memory", "chat")

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

        if memory_node is None:
            # 无记忆能力时，工具结果直接回 chat 交给模型继续推理。
            builder.add_edge("tools", "chat")
        else:
            # 有记忆能力时，同 invocation 直接复用检索结果；HITL 恢复后的新
            # invocation 因临时 channel 缺失而回 memory 补查。
            builder.add_conditional_edges(
                "tools",
                route_after_tools,
                {
                    "chat": "chat",
                    "memory": "memory",
                },
            )

    # checkpointer=None 时保持无持久状态行为；调用方注入
    # InMemorySaver 等实现后，LangGraph 会按 thread_id 保存状态。
    return builder.compile(
        checkpointer=checkpointer,
    )
