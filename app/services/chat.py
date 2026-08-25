"""Application service for running chat Agent turns."""

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any
from uuid import UUID

from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    HumanMessage,
    ToolMessage,
)
from langchain_core.runnables import RunnableConfig
from langgraph.types import Command

from app.agents.chat.context import ChatRuntimeContext
from app.agents.chat.graph import ChatGraph
from app.agents.chat.state import ChatState
from app.core.logging import logger
from app.schemas.chat import (
    ChatMessage,
    ChatRequest,
    ChatResponse,
    ChatResumeRequest,
    ChatStreamEvent,
    DoneStreamEvent,
    ErrorStreamEvent,
    InterruptStreamEvent,
    TokenStreamEvent,
    ToolStreamEvent,
)


@dataclass(frozen=True, slots=True)
class ChatInterrupt:
    """Agent 暂停后交给上层处理的稳定应用结果."""

    thread_id: str
    question: str


ChatTurnResult = ChatResponse | ChatInterrupt


class ChatResumeNotAvailableError(LookupError):
    """请求的用户会话不存在，或当前没有可恢复的人工中断."""


class ChatService:
    """在 API 模型和 LangGraph runtime 之间执行一次聊天用例."""

    def __init__(
        self,
        graph: ChatGraph,
        *,
        graph_timeout_seconds: float = 90.0,
    ) -> None:
        """保存共享 graph，并验证单轮图执行的时间预算.

        Args:
            graph: 已声明 ChatRuntimeContext schema 的共享 ChatGraph。service 可以
                跨用户复用它，但不能把任何当前用户保存到 graph 或实例字段。
            graph_timeout_seconds: 单次 ainvoke/astream 的总执行时间上限，必须大于 0。

        Raises:
            ValueError: graph_timeout_seconds 小于或等于 0。
        """
        if graph_timeout_seconds <= 0:
            raise ValueError("graph_timeout_seconds must be greater than 0")

        self._graph = graph
        self._graph_timeout_seconds = graph_timeout_seconds

    @staticmethod
    def _build_checkpoint_thread_id(user_id: UUID, public_thread_id: str) -> str:
        """构造只在服务端使用的用户隔离 checkpoint key.

        Args:
            user_id: ``get_current_user`` 已完成 JWT 验签和数据库确认的用户 UUID。
            public_thread_id: 客户端提交并在响应中继续使用的公开会话标识。

        Returns:
            同时包含固定长度用户 UUID 和公开 thread ID 的内部 key。不同用户即使
            提交相同 public_thread_id，也会访问不同的 checkpoint 身份空间。

        Notes:
            该值不是安全凭据，无需加密；真正的边界来自 user_id 只能由认证依赖
            提供。它不会返回客户端，也不会进入 Prompt 或 Agent state。
        """
        return f"user:{user_id.hex}:thread:{public_thread_id}"

    @classmethod
    def _build_config(
        cls,
        *,
        user_id: UUID,
        public_thread_id: str,
    ) -> RunnableConfig:
        """为一次用户会话构造隔离的 LangGraph 运行配置.

        Args:
            user_id: 当前认证用户的可信 UUID。
            public_thread_id: 当前 HTTP 请求携带的公开 thread ID。

        Returns:
            供 invoke、stream、resume 和 snapshot 查询共同使用的配置。
        """
        config: RunnableConfig = {
            "configurable": {
                "thread_id": cls._build_checkpoint_thread_id(
                    user_id,
                    public_thread_id,
                ),
            },
            "recursion_limit": 8,
        }
        return config

    @staticmethod
    def _build_runtime_context(user_id: UUID) -> ChatRuntimeContext:
        """把可信用户 UUID 包装成单次图执行的不可变上下文."""
        return ChatRuntimeContext(user_id=user_id)

    @staticmethod
    def _events_from_message_part(data: object) -> tuple[ChatStreamEvent, ...]:
        """把 messages mode 数据转换为客户端可见 token.

        ``messages`` 观察的是模型生成过程。v2 stream 的 data 不是单独一条
        消息，而是 ``(message, metadata)`` 二元组。message 可能是模型文本
        分片，也可能是 ToolMessage 等其他消息，所以需要先做类型收窄。

        断点建议：依次观察 data、message、content 和最终返回值，理解一个
        provider chunk 为什么可能转换为一个事件，也可能不产生任何事件。
        """
        # 此时 data 的静态类型只是 object。运行时检查既保护边界，也帮助
        # Pyright 在后续代码中确认它可以安全地按二元组解包。
        if not isinstance(data, tuple) or len(data) != 2:
            raise RuntimeError("messages stream part must contain message and metadata")

        # metadata 包含节点名等框架上下文；当前事件协议只需要 message，
        # 但仍显式解包并命名，便于以后按节点过滤时扩展。
        message, _metadata = data

        # messages mode 还可能推送 ToolMessage。工具完成事件已经统一从
        # updates mode 的完整节点结果产生，因此这里忽略它，避免重复通知。
        if not isinstance(message, AIMessageChunk):
            return ()

        # AIMessageChunk.content 在框架中不保证一定是字符串；多模态模型可能
        # 返回结构化内容块，所以这里暂时只支持当前 SSE 协议定义的文本 token。
        content = message.content

        # 空字符串常见于 ToolCall 分片或 finish chunk，不应该产生 token。
        # 非空字符串必须原样保留，不能 strip，否则空格和换行会被破坏。
        if not isinstance(content, str) or not content:
            return ()

        # helper 统一返回 tuple：零个事件使用 ()，一个事件使用单元素 tuple，
        # 多个事件则返回多个元素。这样分发层始终可以使用同一种 for 循环。
        return (TokenStreamEvent(text=content),)

    @staticmethod
    def _events_from_update_part(data: object) -> tuple[ChatStreamEvent, ...]:
        """把节点完成后的状态增量转换为工具生命周期事件.

        本方法只处理 ``updates``，不处理模型逐 token 返回的 ``messages``。
        chat 节点中的完整 ToolCall 表示工具即将执行；tools 节点中的
        ToolMessage 表示该次执行已经成功或失败。

        断点建议：观察 data、node_name、node_update、message、tool_call 和
        events，重点理解 part type、节点名称与 LangChain 消息类型是三层概念。
        """
        events: list[ChatStreamEvent] = []

        # version="v2" 的外层 part 是 {"type", "ns", "data"}；stream_turn
        # 会先取出 part["data"]，所以这里收到的是“节点名 -> 状态增量”。
        if not isinstance(data, dict):
            raise RuntimeError("updates stream data must be a dict")

        for node_name, node_update in data.items():
            # node_name 是 LangGraph 节点名，不是外层 part["type"]：
            # part_type 区分 messages/updates，node_name 区分 chat/tools。
            # 当前只翻译 chat/tools 消息；__interrupt__ 等特殊更新在流结束后
            # 通过 graph snapshot 判断，不能误当成普通节点结果。
            if node_name not in {"chat", "tools"}:
                continue

            # 节点返回的是局部状态，例如 {"messages": [AIMessage(...)]}，
            # LangGraph 稍后才会用 add_messages 将它合并进完整 ChatState。
            if not isinstance(node_update, dict):
                raise RuntimeError("node update must be a dict")

            messages = node_update.get("messages")
            if not isinstance(messages, list):
                raise RuntimeError("node messages update must be a list")

            # 一个节点更新可能一次写回多条消息。例如并行执行多个工具时，
            # tools 节点可能返回多个 ToolMessage，因此这里不能只取最后一条。
            for message in messages:
                if node_name == "chat" and isinstance(message, AIMessage):
                    # update 中的 AIMessage 已经完整，可以可靠读取 name 和 id。
                    # messages mode 的 tool_call_chunks 可能只是残缺 JSON 分片，
                    # 因此不能用它们提前猜测一次工具调用是否已经开始。
                    for tool_call in message.tool_calls:
                        tool_name = tool_call.get("name")
                        tool_call_id = tool_call.get("id")

                        # ToolCall.id 的类型允许为 None，但流式协议要求每次调用
                        # 都可被唯一追踪，因此缺少名称或 ID 应立即视为内部错误。
                        if not isinstance(tool_name, str) or not tool_name.strip():
                            raise RuntimeError("tool call must contain a non-empty name")
                        if not isinstance(tool_call_id, str) or not tool_call_id.strip():
                            raise RuntimeError("tool call must contain a non-empty id")

                        # args 可能包含用户输入或敏感参数，本事件只暴露生命周期，
                        # 不把参数发送给客户端。
                        events.append(
                            ToolStreamEvent(
                                name=tool_name,
                                tool_call_id=tool_call_id,
                                status="started",
                            )
                        )

                elif node_name == "tools" and isinstance(message, ToolMessage):
                    # tool_call 已经执行完成
                    tool_name = message.name
                    tool_call_id = message.tool_call_id

                    if not isinstance(tool_name, str) or not tool_name.strip():
                        raise RuntimeError("tool result must contain a non-empty name")
                    if not isinstance(tool_call_id, str) or not tool_call_id.strip():
                        raise RuntimeError("tool result must contain a non-empty id")

                    # ToolMessage.content 可能是较大的结果或内部错误详情；应用层
                    # 只公开安全状态。相同 tool_call_id 将本事件与 started 配对。
                    events.append(
                        ToolStreamEvent(
                            name=tool_name,
                            tool_call_id=tool_call_id,
                            status=message.status,
                        )
                    )

        return tuple(events)

    async def _parse_graph_result(
        self,
        result: dict[str, Any],
        *,
        config: RunnableConfig,
        thread_id: str,
    ) -> ChatTurnResult:
        """把 LangGraph 原始结果转换成稳定的应用层结果."""
        result_dict = dict(result)
        if "__interrupt__" in result_dict:
            snapshot = await self._graph.aget_state(config)
            interrupts = snapshot.tasks[0].interrupts if snapshot.tasks else ()

            if len(interrupts) != 1:
                raise RuntimeError("interrupted graph must contain exactly one interrupt")

            question = interrupts[0].value

            if not isinstance(question, str) or not question.strip():
                raise RuntimeError("chat interrupt must contain a non-empty question")

            return ChatInterrupt(
                thread_id=thread_id,
                question=question.strip(),
            )

        messages = result["messages"]
        if not messages:
            raise RuntimeError("completed chat graph returned no messages")

        final_message = messages[-1]

        if not isinstance(final_message, AIMessage):
            raise RuntimeError("completed chat graph must end with an AIMessage")

        # tool_calls 非空说明 Agent 仍想执行工具，不能把它当成最终回答。
        if final_message.tool_calls:
            raise RuntimeError("completed chat graph contains unresolved tool calls")

        content = final_message.content
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError("completed chat graph must contain text content")

        return ChatResponse(
            thread_id=thread_id,
            message=ChatMessage(
                role="assistant",
                content=content,
            ),
        )

    async def run_turn(
        self,
        request: ChatRequest,
        *,
        user_id: UUID,
    ) -> ChatTurnResult:
        """使用可信用户身份执行一轮 Agent，并返回完成或暂停结果.

        Args:
            request: 已由 Pydantic 校验的公开 thread ID 和本轮消息。
            user_id: 由 HTTP 认证依赖提供的可信用户 UUID，不能来自 request。
        """
        # 显式使用 ChatState，而不是让 Python 把字面量推断成
        # dict[str, list[HumanMessage]]。list 是不变类型；虽然 HumanMessage 属于
        # AnyMessage，list[HumanMessage] 仍不能自动替代 list[AnyMessage]。
        graph_input: ChatState = {
            "messages": [
                HumanMessage(content=request.message),
            ]
        }

        config = self._build_config(
            user_id=user_id,
            public_thread_id=request.thread_id,
        )
        runtime_context = self._build_runtime_context(user_id)

        # 在整图时间预算内调用 ainvoke
        result = await asyncio.wait_for(
            self._graph.ainvoke(
                graph_input,
                config=config,
                context=runtime_context,
            ),
            timeout=self._graph_timeout_seconds,
        )

        return await self._parse_graph_result(
            result,
            config=config,
            thread_id=request.thread_id,
        )

    async def resume_turn(
        self,
        request: ChatResumeRequest,
        *,
        user_id: UUID,
    ) -> ChatTurnResult:
        """只在当前用户自己的 checkpoint 空间恢复人工中断.

        Args:
            request: 公开 thread ID 和人工回答。
            user_id: 当前认证用户的可信 UUID。

        Raises:
            ChatResumeNotAvailableError: 当前用户的内部 checkpoint 不存在，或其中
                没有等待恢复的 interrupt。相同错误也用于隐藏其他用户会话是否存在。
        """
        config = self._build_config(
            user_id=user_id,
            public_thread_id=request.thread_id,
        )
        runtime_context = self._build_runtime_context(user_id)

        # 先在“当前用户 + 公开 thread ID”的内部空间检查 interrupt。用户 B 即使
        # 猜到用户 A 的公开 thread ID，读取的也是 B 自己的空 checkpoint。
        snapshot = await self._graph.aget_state(config)
        interrupts = tuple(interrupt for task in snapshot.tasks for interrupt in task.interrupts)
        if not interrupts:
            raise ChatResumeNotAvailableError
        if len(interrupts) != 1:
            raise RuntimeError("resumable chat graph must contain exactly one interrupt")

        result = await asyncio.wait_for(
            self._graph.ainvoke(
                Command(resume=request.response),
                config=config,
                context=runtime_context,
            ),
            timeout=self._graph_timeout_seconds,
        )

        return await self._parse_graph_result(
            result,
            config=config,
            thread_id=request.thread_id,
        )

    async def stream_turn(
        self,
        request: ChatRequest,
        *,
        user_id: UUID,
    ) -> AsyncIterator[ChatStreamEvent]:
        """流式执行一轮 Agent，并逐个产生稳定的应用层事件.

        这是异步生成器：每次执行到 ``yield`` 就把一个事件交给上层并暂停；
        上层请求下一个事件时，才从暂停位置继续执行，而不是等待整图结束。

        astream() 负责“边执行图，边报告过程”；本方法负责把框架报告翻译成
        应用协议，并处理取消、错误、安全隔离和最终状态判断。
        """
        config = self._build_config(
            user_id=user_id,
            public_thread_id=request.thread_id,
        )
        runtime_context = self._build_runtime_context(user_id)
        # 与非流式入口使用相同的 ChatState 输入契约，保证 ainvoke/astream 不会
        # 因局部变量推断差异产生两套类型边界。
        graph_input: ChatState = {
            "messages": [HumanMessage(content=request.message)],
        }

        try:
            async with asyncio.timeout(self._graph_timeout_seconds):
                # 同时观察模型文本分片和节点完成后的状态增量。
                async for part in self._graph.astream(
                    graph_input,
                    config=config,
                    context=runtime_context,
                    stream_mode=("messages", "updates"),
                    version="v2",
                ):
                    # 每轮断点停在这里时，part 表示“刚到达的一片流数据”。
                    # v2 part 应具有 type、ns、data；它不是 ChatState 全量快照。
                    if not isinstance(part, dict):
                        raise RuntimeError("graph stream part must be a dict")

                    # type 是 stream mode 标签，而不是消息类型或节点名称。
                    # data 的形状由 type 决定：messages 是二元组，updates 是字典。
                    part_type = part.get("type")
                    part_data = part.get("data")

                    # 分发层不理解 ToolCall 细节，只把不同 payload 交给对应
                    # helper 翻译。这样框架数据解析不会和主控制流混在一起。
                    if part_type == "messages":
                        events = self._events_from_message_part(part_data)
                    elif part_type == "updates":
                        events = self._events_from_update_part(part_data)
                    else:
                        raise RuntimeError("graph stream part contains an unknown type")

                    # helper 返回 tuple：可能为空，也可能包含多个并发工具事件。
                    # 必须逐个 yield；yield events 会错误地产生“事件元组”。
                    for event in events:
                        # 断点执行过这一行后，控制权会暂时交回 API/SSE 调用方；
                        # 下一次请求事件时，才回到这里继续处理后续 part。
                        yield event

                # astream 停止产生 part 不等于图一定到达 END：ask_human 会让图
                # 暂停并结束本轮 stream。因此必须读取同一 thread 的最新快照，
                # 不能根据“最后一个 part 是什么”猜测最终状态。
                snapshot = await self._graph.aget_state(config)

                # 一个 snapshot 可能包含多个 task。每个 task 对应一次节点执行，
                # interrupt 保存在具体 task 中，所以需要展开所有 task，而不能只
                # 检查 tasks[0]。断点时可同时观察 snapshot.next 和 snapshot.tasks。
                interrupts = tuple(interrupt for task in snapshot.tasks for interrupt in task.interrupts)

                if interrupts:
                    # 当前最小聊天 Agent 一次只支持一个人工问题。多个 interrupt
                    # 无法安全映射成现有的单 question 协议，应作为内部状态错误。
                    if len(interrupts) != 1:
                        raise RuntimeError("interrupted chat stream must contain exactly one interrupt")

                    question = interrupts[0].value
                    if not isinstance(question, str) or not question.strip():
                        raise RuntimeError("chat stream interrupt must contain a non-empty question")

                    # interrupt 不是失败，而是“本轮正常结束并等待人工输入”。先把
                    # 问题交给客户端，再用 done(interrupted) 明确关闭本次事件流。
                    yield InterruptStreamEvent(question=question.strip())
                    yield DoneStreamEvent(status="interrupted")

                    # 必须立即 return；否则代码会继续向下发送 completed，导致
                    # 同一轮 stream 同时拥有两个相互冲突的终止状态。
                    return

                # 没有 interrupt 时，正常完成的图应该已经到达 END，此时 next
                # 为空。若 next 仍有节点，说明图在未知的非终态停止，不能谎报成功。
                if snapshot.next:
                    raise RuntimeError("chat stream ended before graph reached a terminal state")

                # completed 表示图真正到达 END，也是正常流的最后一个事件。
                yield DoneStreamEvent(status="completed")
                return

        except asyncio.CancelledError:
            # 客户端断开时必须让取消继续向上传播。
            # 避免继续执行图、造成资源浪费、模型消耗
            raise
        except TimeoutError:
            yield ErrorStreamEvent(
                code="CHAT_STREAM_TIMEOUT",
                message="Chat stream timed out",
            )
            return
        except Exception as error:
            logger.exception(
                "chat_stream_failed",
                error_type=type(error).__name__,
                thread_id=request.thread_id,
            )
            yield ErrorStreamEvent(
                code="CHAT_STREAM_ERROR",
                message="Chat stream failed",
            )
            return
