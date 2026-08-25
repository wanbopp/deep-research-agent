"""Node implementations for the chat Agent."""

import asyncio
from collections.abc import Sequence

from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.messages.tool import ToolCall
from langgraph.errors import GraphBubbleUp
from langgraph.runtime import Runtime

from app.agents.chat.context import ChatRuntimeContext
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

    async def chat_node(
        state: ChatState,
        *,
        runtime: Runtime[ChatRuntimeContext],
    ) -> ChatState:
        """把当前消息交给 LLMService，并返回一条消息增量.

        ``runtime.context`` 只来自服务端调用 ``ainvoke/astream`` 时传入的
        ChatRuntimeContext。这里主动读取它以确认可信身份已经到达节点，但不会把
        user_id 写入 Prompt、消息或模型参数；模型因此无法查看或覆盖授权身份。
        """
        runtime_context = runtime.context
        if not isinstance(runtime_context, ChatRuntimeContext):
            raise RuntimeError("chat node requires a trusted runtime context")

        # 当前普通聊天节点尚不执行用户资源查询。保留这个局部变量便于断点确认
        # 身份注入成功；未来身份相关能力只能使用它，不能相信 ToolCall 参数。
        _trusted_user_id = runtime_context.user_id

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
    registry: ToolRegistry,  # 位置参数：可以按位置或按名称传
    *,  # 分隔符：之后的参数必须按名称传
    tool_timeout_seconds: float = 10.0,  # keyword-only 参数
) -> StateNode:
    """创建并发执行一个或多个工具调用的异步节点.

    LLM 返回的 tool_calls 可能包含多条调用。这个节点会为每条
    ToolCall 创建一个协程，由 asyncio.gather() 并发等待，再返回
    顺序稳定的 ToolMessage 列表，交给 add_messages 追加到状态。

    gather() 按输入协程的顺序返回结果，而不是按完成时间排序。
    因此即使 call-2 最先完成，输出仍保持 call-1、call-2、call-3，
    便于模型按 tool_call_id 找到每条调用对应的工具结果。
    """
    if tool_timeout_seconds <= 0:
        raise ValueError("tool_timeout_seconds must be greater than 0")

    async def _execute_tool_call(tool_call: ToolCall) -> ToolMessage:
        """执行一条调用，并转换为成功或错误 ToolMessage."""
        tool_call_id = tool_call["id"]
        if not isinstance(tool_call_id, str) or not tool_call_id:
            raise ValueError("tool call id must be a non-empty string")

        tool_name = tool_call["name"]

        try:
            tool = registry.resolve(tool_name)
        except LookupError:
            return ToolMessage(
                content=f"Tool {tool_name!r} is not available.",
                name=tool_name,
                tool_call_id=tool_call_id,
                status="error",
            )

        try:
            async with asyncio.timeout(tool_timeout_seconds):
                result = await tool.ainvoke(tool_call["args"])
        except TimeoutError:
            return ToolMessage(
                content=f"Tool {tool_name!r} timed out.",
                name=tool_name,
                tool_call_id=tool_call_id,
                status="error",
            )
        except GraphBubbleUp:
            # interrupt 和其他 LangGraph 控制流不能转换成错误消息。
            # 必须交还 LangGraph 调度器，由它保存暂停状态。
            raise
        except Exception as error:
            # 每条协程在内部把普通执行异常转换为错误消息，使一个工具
            # 失败时 gather() 仍能收集其他工具的成功结果。
            # 只暴露异常类型，不把可能包含敏感数据的 str(error) 交给模型。
            return ToolMessage(
                content=(f"Tool {tool_name!r} failed with {type(error).__name__}."),
                name=tool_name,
                tool_call_id=tool_call_id,
                status="error",
            )

        return ToolMessage(
            content=str(result),
            name=tool.name,
            tool_call_id=tool_call_id,
        )

    async def tool_node(
        state: ChatState,
        *,
        runtime: Runtime[ChatRuntimeContext],
    ) -> ChatState:
        """在可信用户上下文中并发执行全部工具调用.

        当前工具都不访问用户数据，因此 user_id 不会被拼进模型生成的 ``args``。
        读取 runtime context 的意义是建立今后的授权边界：身份相关工具应由服务端
        注入此值，而不是要求模型在 ToolCall 中提供 user_id。
        """
        runtime_context = runtime.context
        if not isinstance(runtime_context, ChatRuntimeContext):
            raise RuntimeError("tool node requires a trusted runtime context")

        _trusted_user_id = runtime_context.user_id

        messages = state["messages"]
        if not messages:
            raise ValueError("tool node requires at least one message")

        # tool node 只能处理模型提出的工具调用，不能把 HumanMessage
        # 或 ToolMessage 误当成待执行指令。
        last_message = messages[-1]
        if not isinstance(last_message, AIMessage):
            raise TypeError("tool node requires the last message to be an AIMessage")

        tool_calls = last_message.tool_calls
        if not tool_calls:
            raise ValueError("tool node requires at least one tool call")

        # 所有协程在 gather() 调用时一起开始推进；返回列表仍与
        # tool_calls 的输入顺序一致，不受各工具完成先后的影响。
        outputs = await asyncio.gather(*(_execute_tool_call(tool_call) for tool_call in tool_calls))

        return {"messages": list(outputs)}

    return tool_node
