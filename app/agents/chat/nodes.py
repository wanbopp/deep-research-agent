"""Node implementations for the minimal chat Agent."""

from collections.abc import Sequence

from langchain_core.messages import AIMessage, ToolMessage

from app.agents.chat.graph import StateNode
from app.agents.chat.state import ChatState
from app.agents.chat.tools.registry import ToolRegistry
from app.services.llm.service import LLMService


def create_chat_node(
    llm_service: LLMService,
    aliases: Sequence[str],
    *,
    tool_registry: ToolRegistry | None = None,
) -> StateNode:
    """创建只通过 LLMService 调用模型的异步 state node."""
    if isinstance(aliases, str) or not aliases:
        raise ValueError("aliases must contain at least one model alias")

    # 保存不可变副本，避免调用方后续修改原始列表。
    model_aliases = tuple(aliases)
    model_tools = tool_registry.tools() if tool_registry is not None else ()

    async def chat_node(state: ChatState) -> ChatState:
        """把当前消息交给 LLMService，并返回一条消息增量."""
        response = await llm_service.call(
            state["messages"],
            aliases=model_aliases,
            tools=model_tools,
        )

        # ainvoke() 应返回一条完整的 AIMessage。
        # 这里既验证真实运行时类型，也让 pyright 把 BaseMessage
        # 收窄为 ChatState.messages 所接受的 AIMessage。
        if not isinstance(response, AIMessage):
            raise TypeError("LLMService.call() must return an AIMessage")

        # 这里只返回本轮新生成的消息。
        # LangGraph 会通过 add_messages 将它合并进已有消息历史。
        return {"messages": [response]}

    return chat_node


def create_tool_node(
    registry: ToolRegistry,
) -> StateNode:
    """创建执行单个工具调用的异步节点."""

    async def tool_node(state: ChatState) -> ChatState:
        """执行最后一条 AIMessage 中的单个工具调用."""
        messages = state["messages"]
        if not messages:
            raise ValueError("tool node requires at least one message")

        # tool node 只能处理模型提出的工具调用，不能把 HumanMessage
        # 或 ToolMessage 误当成待执行指令。
        last_message = messages[-1]
        if not isinstance(last_message, AIMessage):
            raise TypeError("tool node requires the last message to be an AIMessage")

        # 本 checkpoint 只处理单个调用；多个调用会在后续并发阶段实现。
        if len(last_message.tool_calls) != 1:
            raise ValueError("tool node requires exactly one tool call")

        tool_call = last_message.tool_calls[0]
        tool_call_id = tool_call["id"]

        # LangChain 的 ToolCall.id 类型允许 None，但 ToolMessage 必须使用
        # 非空字符串才能与模型提出的调用准确配对。
        if not isinstance(tool_call_id, str) or not tool_call_id:
            raise ValueError("tool call id must be a non-empty string")

        try:
            tool = registry.resolve(tool_call["name"])
        except LookupError:
            # 未知工具通常来自模型决策偏差。把错误作为 ToolMessage
            # 返回后，模型可以在下一轮选择其他工具或修正答案。
            return {
                "messages": [
                    ToolMessage(
                        content=f"Tool {tool_call['name']!r} is not available.",
                        name=tool_call["name"],
                        tool_call_id=tool_call_id,
                        status="error",
                    )
                ]
            }

        # 这里只把模型已经通过 schema 生成的参数交给白名单工具。
        # 工具结果转换为字符串后，作为下一轮模型可以读取的消息内容。
        result = await tool.ainvoke(tool_call["args"])

        return {
            "messages": [
                ToolMessage(
                    content=str(result),
                    name=tool.name,
                    tool_call_id=tool_call_id,
                )
            ]
        }

    return tool_node
