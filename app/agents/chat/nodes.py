"""Node implementations for the chat Agent."""

import asyncio
from collections.abc import Sequence
import json

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.messages.tool import ToolCall
from langgraph.errors import GraphBubbleUp
from langgraph.runtime import Runtime

from app.agents.chat.context import ChatRuntimeContext
from app.agents.chat.graph import StateNode
from app.agents.chat.state import ChatState, ChatStateUpdate
from app.agents.chat.tools.registry import ToolRegistry
from app.agents.prompts.loader import load_prompt_artifact
from app.schemas.memory import (
    MAX_QUERY_LENGTH,
    MAX_QUERY_LIMIT,
    MemoryQuery,
    MemorySearchStatus,
)
from app.services.llm.service import LLMService
from app.services.memory_service import MemoryService
from app.tools import ToolExecutionContext, ToolExecutor
from app.tools.policy import ToolApprovalRequired, ToolAuthorizationError


def create_memory_node(
    memory_service: MemoryService,
    *,
    query_limit: int = 5,
) -> StateNode:
    """创建每次 Graph invocation 最多执行一次的长期记忆检索节点.

    Args:
        memory_service: 可跨请求共享的长期记忆应用服务。它负责缓存、用户隔离复核
            和基础设施降级，不把数据库或 Redis 客户端暴露给 Agent。
        query_limit: 本轮最多注入模型上下文的记忆数量。

    Returns:
        读取完整 ChatState、返回局部 ChatStateUpdate 的异步节点。

    Raises:
        ValueError: query_limit 超出 MemoryQuery 允许范围，或者运行状态中没有可供
            检索的用户消息。
        RuntimeError: LangGraph 没有注入可信的 ChatRuntimeContext。
        TypeError: 最近一条 HumanMessage 不是当前聊天接口支持的纯文本消息。
    """
    # 这个检查发生在 runtime 组装阶段。若配置非法，应用应在 startup 时失败，
    # 不能等到第一位用户触发 Graph 后才暴露问题。bool 是 int 的子类，因此也要
    # 显式拒绝 True/False，避免把开关值误当成查询数量。
    if isinstance(query_limit, bool) or not 1 <= query_limit <= MAX_QUERY_LIMIT:
        raise ValueError(f"query_limit must be between 1 and {MAX_QUERY_LIMIT}")

    async def memory_node(
        state: ChatState,
        *,
        runtime: Runtime[ChatRuntimeContext],
    ) -> ChatStateUpdate:
        """使用可信用户身份检索与本轮问题相关的长期记忆.

        该节点只读取完整状态，并返回 ``memory_context`` 和 ``memory_status`` 两个
        局部更新。它不会修改消息历史，也不会把 user_id 交给模型决定。
        """
        runtime_context = runtime.context
        if not isinstance(runtime_context, ChatRuntimeContext):
            raise RuntimeError("memory node requires a trusted runtime context")

        messages = state.get("messages")
        if not messages:
            raise ValueError("memory node requires at least one message")

        # checkpoint 可能包含多轮对话。倒序查找能定位触发当前 invocation 的最近
        # 一条用户消息，而不是错误地使用 thread 中最早的用户问题。
        human_message = next(
            (message for message in reversed(messages) if isinstance(message, HumanMessage)),
            None,
        )
        if human_message is None:
            raise ValueError("memory node requires a HumanMessage")

        # LangChain 也允许多模态内容块，但当前 ChatRequest 只接受纯文本。显式收窄
        # 类型可以防止对 list 内容调用 strip()，也为未来多模态支持保留清晰边界。
        content = human_message.content
        if not isinstance(content, str):
            raise TypeError("memory node requires text HumanMessage content")

        # ChatRequest 最多 8000 字符，而向量查询最多 1000 字符。记忆增强属于辅助
        # 能力，不能因为长输入违反 MemoryQuery 边界而阻断正常聊天，因此先去除
        # 首尾空白并做有界截断。若未来需要更好的长文本召回，应单独引入查询压缩。
        query_text = content.strip()[:MAX_QUERY_LENGTH]
        if not query_text:
            raise ValueError("memory node requires non-empty HumanMessage content")

        memory_query = MemoryQuery(
            text=query_text,
            limit=query_limit,
        )

        # user_id 只能来自服务端创建的 runtime context。MemoryService 会把预期的
        # Redis/PostgreSQL/provider 故障转换为 degraded 结果，所以节点无需捕获宽泛
        # Exception；真正的编程错误仍应向上暴露，避免被伪装成普通空记忆。
        memory_search_result = await memory_service.search(
            user_id=runtime_context.user_id,
            query=memory_query,
        )

        # 不返回 messages：memory node 没有生成消息。LangGraph 会用 UntrackedValue
        # 覆盖这两个字段，并保持 add_messages 管理的历史完全不变。
        return {
            "memory_context": memory_search_result.items,
            "memory_status": memory_search_result.status,
        }

    return memory_node


def _build_model_messages(state: ChatState) -> tuple[AnyMessage, ...]:
    """为本次模型调用构造临时消息视图，不修改 Graph 持久状态.

    记忆正文来自历史用户数据，可能包含类似指令的文本，因此只能作为
    低信任背景资料。这里不能把它提升成 SystemMessage，也不能追加回
    state["messages"]。


    记忆消息只存在于 model_messages 这个临时变量中，传给 LLM 后就被丢弃了。
    ·chat_node 只返回 {"messages": [response]}，LangGraph 只会把节点返回值合并进 state 并写入 checkpoint——记忆消息从未出现在返回值里，所以永远不会被持久化
    """
    prompt = load_prompt_artifact("chat_assistant")
    # Graph checkpoint 只应保存用户、助手和工具消息。即便旧数据意外含有
    # SystemMessage，也不能让它与当前服务端固定系统规则并列。
    messages = tuple(message for message in state["messages"] if not isinstance(message, SystemMessage))
    system_message = SystemMessage(content=prompt.content)
    memory_items = state.get("memory_context", ())
    memory_status = state.get("memory_status")

    # Store 降级或正常空结果都不阻断聊天，直接使用原始会话历史。
    # DEGRADED 的可观察性由 MemoryService 和日志负责，无需告诉模型。
    if memory_status is not MemorySearchStatus.AVAILABLE or not memory_items:
        return (system_message, *messages)

    # 倒序定位最近一条 HumanMessage 的下标。
    # 记忆消息应该插在本轮用户问题之前，这样模型先看到背景资料再看到用户提问，
    # 而不是把记忆放在整个历史开头（可能远离当前上下文）。
    latest_human_index = next(
        (len(messages) - 1 - i for i, message in enumerate(reversed(messages)) if isinstance(message, HumanMessage)),
        None,
    )

    if latest_human_index is None:
        raise ValueError("chat node requires a HumanMessage")

    # 只提取 kind 和 content，不把 user_id、数据库主键或审计时间发送给模型。
    memory_payload = json.dumps(
        {"retrieved_memory": [{"kind": item.kind.value, "content": item.content} for item in memory_items]},
        ensure_ascii=False,
        separators=(",", ": "),
    )

    # System Prompt 已声明 retrieved_memory 是低信任数据；这里仅发送 JSON，
    # 不在 HumanMessage 中混入新的行为指令。
    memory_message = HumanMessage(content=memory_payload)

    # 这是新 tuple，没有原地修改 messages。
    return (
        system_message,
        *messages[:latest_human_index],
        memory_message,
        *messages[latest_human_index:],
    )


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
    ) -> ChatStateUpdate:
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

        # model_messages 是本次模型调用的临时视图。
        # 它可以包含检索记忆，但不会被 chat node 返回，因此不会进入 checkpoint。
        model_messages = _build_model_messages(state)
        prompt = load_prompt_artifact("chat_assistant")

        response = await llm_service.call(
            model_messages,
            aliases=model_aliases,
            tools=model_tools,
            prompt=prompt,
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
    executor: ToolExecutor | None = None,
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

    async def _execute_tool_call(
        tool_call: ToolCall,
        execution_context: ToolExecutionContext,
    ) -> ToolMessage:
        """执行一条调用，并转换为成功或错误 ToolMessage."""
        tool_call_id = tool_call["id"]
        if not isinstance(tool_call_id, str) or not tool_call_id:
            raise ValueError("tool call id must be a non-empty string")

        tool_name = tool_call["name"]

        if executor is not None:
            try:
                result = await executor.execute(
                    tool_name,
                    dict(tool_call["args"]),
                    context=execution_context,
                )
            except GraphBubbleUp:
                raise
            except (LookupError, ToolApprovalRequired, ToolAuthorizationError):
                return ToolMessage(
                    content=f"Tool {tool_name!r} is not available for this request.",
                    name=tool_name,
                    tool_call_id=tool_call_id,
                    status="error",
                )
            return ToolMessage(
                content=result.content,
                name=tool_name,
                tool_call_id=tool_call_id,
                status="success" if result.status == "success" else "error",
            )

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
    ) -> ChatStateUpdate:
        """在可信用户上下文中并发执行全部工具调用.

        当前工具都不访问用户数据，因此 user_id 不会被拼进模型生成的 ``args``。
        读取 runtime context 的意义是建立今后的授权边界：身份相关工具应由服务端
        注入此值，而不是要求模型在 ToolCall 中提供 user_id。
        """
        runtime_context = runtime.context
        if not isinstance(runtime_context, ChatRuntimeContext):
            raise RuntimeError("tool node requires a trusted runtime context")

        execution_context = ToolExecutionContext(user_id=runtime_context.user_id)

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
        outputs = await asyncio.gather(*(_execute_tool_call(tool_call, execution_context) for tool_call in tool_calls))

        return {"messages": list(outputs)}

    return tool_node
