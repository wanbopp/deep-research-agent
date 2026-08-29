"""Application service for running chat Agent turns."""

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, Literal
from uuid import UUID
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    HumanMessage,
    ToolMessage,
)
from langchain_core.runnables import RunnableConfig
from langgraph.types import Command

from app.services.memory_extraction import ChatMemoryWriter
from app.services.chat_title import ChatTitleWriter
from app.agents.chat.context import ChatRuntimeContext
from app.agents.chat.graph import ChatGraph
from app.agents.chat.state import ChatState
from app.core.logging import logger
from app.schemas.chat import (
    MAX_MESSAGE_LENGTH,
    ChatMessage,
    ChatMessageHistoryResponse,
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
from app.services.chat_session_ownership import (
    ChatSessionNotFoundError,
    ChatSessionOwnershipVerifier,
)
from app.services.chat_guard import (
    ChatExecutionGuard,
    ChatExecutionGuardUnavailableError,
    ChatThreadBusyError,
)


@dataclass(frozen=True, slots=True)
class ChatInterrupt:
    """Agent 暂停后交给上层处理的稳定应用结果."""

    thread_id: UUID
    question: str


ChatTurnResult = ChatResponse | ChatInterrupt


class ChatResumeNotAvailableError(LookupError):
    """请求的用户会话不存在，或当前没有可恢复的人工中断."""


class ChatService:
    """在 API 模型、业务授权和 LangGraph runtime 之间执行聊天用例.

    该对象由 lifespan 创建并跨请求共享，因此只能保存无请求状态的 Graph、guard、
    ownership verifier 和配置值。当前 user_id、会话 UUID、RunnableConfig 与
    AsyncSession 都必须在单次方法调用中创建和释放，不能进入实例字段。
    """

    def __init__(
        self,
        graph: ChatGraph,
        *,
        execution_guard: ChatExecutionGuard,
        ownership_verifier: ChatSessionOwnershipVerifier,
        graph_timeout_seconds: float = 90.0,
        memory_writer: ChatMemoryWriter | None = None,
        title_writer: ChatTitleWriter | None = None,
    ) -> None:
        """保存共享 graph，并验证单轮图执行的时间预算.

        Args:
            graph: 已声明 ChatRuntimeContext schema 的共享 ChatGraph。service 可以
                跨用户复用它，但不能把任何当前用户保存到 graph 或实例字段。
            execution_guard: 按内部 thread ID 协调执行权的应用层接口。production
                注入 Redis 实现，保证不同 worker 也能看到相同 busy 状态。
            ownership_verifier: 在 Graph 执行前验证业务 ChatSession 所有权的应用层
                协议。production 注入只保存 sessionmaker 的 PostgreSQL 实现；它不
                让共享 ChatService 持有请求级 AsyncSession。
            graph_timeout_seconds: 单次 ainvoke/astream 的总执行时间上限，必须大于 0。
            memory_writer:可选的后台记忆写入边界。None 用于兼容尚未启用长期记忆的最小 Graph 和旧 smoke；production
                lifespan 后续必须显式注入，并由真实 smoke 证明该能力没有被静默关闭。
            title_writer: 可选的后台会话命名边界。production 注入数据库 claim
                实现；旧的单图 smoke 可保持 None，避免测试被迫构造 ORM 资源。
        Raises:
            ValueError: graph_timeout_seconds 小于或等于 0。

        Notes:
            构造发生在应用 startup；依赖对象均可跨请求共享，但不能保存当前用户。
        """
        if graph_timeout_seconds <= 0:
            raise ValueError("graph_timeout_seconds must be greater than 0")

        self._graph = graph
        self._execution_guard = execution_guard
        self._graph_timeout_seconds = graph_timeout_seconds
        self._ownership_verifier = ownership_verifier
        self._memory_writer = memory_writer
        self._title_writer = title_writer

    @staticmethod
    def _build_checkpoint_thread_id(user_id: UUID, public_thread_id: UUID) -> str:
        """构造只在服务端使用的用户隔离 checkpoint key.

        Args:
            user_id: ``get_current_user`` 已完成 JWT 验签和数据库确认的用户 UUID。
            public_thread_id: Pydantic 已解析的公开业务会话 UUID。

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
        public_thread_id: UUID,
    ) -> RunnableConfig:
        """为一次用户会话构造隔离的 LangGraph 运行配置.

        Args:
            user_id: 当前认证用户的可信 UUID。
            public_thread_id: 当前 HTTP 请求携带的公开 thread ID。

        Returns:
            供 invoke、stream、resume 和 snapshot 查询共同使用的配置。
        """
        internal_thread_id = cls._build_checkpoint_thread_id(
            user_id,
            public_thread_id,
        )
        return cls._build_config_from_internal_thread_id(internal_thread_id)

    @staticmethod
    def _build_runtime_context(user_id: UUID) -> ChatRuntimeContext:
        """把可信用户 UUID 包装成单次图执行的不可变上下文."""
        return ChatRuntimeContext(user_id=user_id)

    @staticmethod
    def _build_config_from_internal_thread_id(
        internal_thread_id: str,
    ) -> RunnableConfig:
        """使用已经验证的内部 thread ID 构造 LangGraph 运行配置."""
        return {
            "configurable": {
                "thread_id": internal_thread_id,
            },
            "recursion_limit": 8,
        }

    @asynccontextmanager
    async def _hold_thread_execution(
        self,
        *,
        user_id: UUID,
        public_thread_id: UUID,
    ) -> AsyncIterator[RunnableConfig]:
        """取得内部 thread 执行权并产生对应的 LangGraph 配置.

        Args:
            user_id: 已由认证依赖确认的当前用户 UUID。
            public_thread_id: Pydantic 已验证的业务会话 UUID。

        Yields:
            与 guard 使用同一个内部 thread ID 构造的 RunnableConfig。

        Raises:
            ChatThreadBusyError: 相同内部 thread 已有 Graph 正在执行。
            ChatExecutionGuardUnavailableError: guard 后端无法安全协调执行权。
            ChatSessionNotFoundError: 当前用户不拥有该业务会话。
            SQLAlchemyError: 所有权查询的数据库不可用。基础设施故障必须继续向上
                传播，不能伪装成“会话不存在”。

        Notes:
            这个 helper 是三个公开入口的共同临界区边界。不要分别在 run、resume、
            stream 中手写 Redis 获取和释放，否则某条异常路径很容易遗漏 finally。
        """
        # public_thread_id 已是 UUID 对象，其字符串表示唯一规范。guard key 和
        # checkpoint key 因而不会因客户端大小写或 UUID 表示方式产生分叉。
        internal_thread_id = self._build_checkpoint_thread_id(
            user_id,
            public_thread_id,
        )

        async with self._execution_guard.hold(internal_thread_id):
            # 授权查询必须位于同 key guard 内。10F-D 的删除流程也会取得这把锁，
            # 因而不会在“检查通过”和 Graph 真正执行之间删除业务行/checkpoint。
            await self._ownership_verifier.require_owned(
                session_id=public_thread_id,
                user_id=user_id,
            )

            # guard key 和 configurable.thread_id 必须来自同一个局部变量。
            yield self._build_config_from_internal_thread_id(internal_thread_id)

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

    @staticmethod
    def _latest_human_message_text(
        state_values: dict[str, Any],
    ) -> str:
        """从 Graph 状态中读取最近一条纯文本用户消息.

        Args:
            state_values: LangGraph result 或 snapshot.values 中的完整状态字典。

        Returns:
            去除首尾空白后的最近一条 HumanMessage 文本。

        Raises:
            RuntimeError: 状态没有消息、没有 HumanMessage，或者当前消息不是纯文本。

        Notes:
            HITL 的 resume 值通常进入 ToolMessage，并不代表触发本轮 Agent 的原始问题。
            后台记忆提取需要使用 checkpoint 中最近的 HumanMessage，避免把简单的
            “approved”或“继续”错误保存为长期记忆。
        """
        messages = state_values.get("messages")
        if not isinstance(messages, list) or not messages:
            raise RuntimeError("chat state must contain messages")

        human_message = next(
            (message for message in reversed(messages) if isinstance(message, HumanMessage)),
            None,
        )
        if human_message is None:
            raise RuntimeError("chat state must contain a HumanMessage")

        content = human_message.content
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError("HumanMessage must contain text content")

        return content.strip()

    async def _parse_graph_result(
        self,
        result: dict[str, Any],
        *,
        config: RunnableConfig,
        thread_id: UUID,
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

    def _submit_completed_turn_enhancements(
        self,
        result: ChatTurnResult,
        *,
        user_id: UUID,
        source_thread_id: UUID,
        user_message: str,
    ) -> None:
        """只为已经验证完成的最终回复提交独立后台增强任务.

        Args:
            result: Graph 解析后的稳定应用结果，可能是完成响应或人工中断。
            user_id: 认证链提供的可信用户 UUID。
            source_thread_id: 已通过所有权检查的业务会话 UUID。
            user_message: 触发本轮 Graph 的用户输入。

        Notes:
            本方法不等待提取和数据库写入。interrupt 没有最终助手回复，因此不能提交；
            两个 writer 分别提交独立 Task。一个提交器接线失败时仍继续尝试另一个，
            并且都不能把已经完成的聊天响应改成失败。
        """
        if not isinstance(result, ChatResponse):
            return

        if self._memory_writer is not None:
            try:
                self._memory_writer.submit_turn(
                    user_id=user_id,
                    source_thread_id=source_thread_id,
                    user_message=user_message,
                    assistant_message=result.message.content,
                )
            except Exception as error:
                # submit_turn 理论上只负责创建后台任务，但 event loop 或 shutdown
                # 竞态仍可能同步拒绝。长期记忆是增强能力，不能逆转 ChatResponse。
                logger.exception(
                    "memory_write_submission_failed",
                    error_type=type(error).__name__,
                )

        if self._title_writer is not None:
            try:
                self._title_writer.submit_turn(
                    user_id=user_id,
                    source_thread_id=source_thread_id,
                    user_message=user_message,
                    assistant_message=result.message.content,
                )
            except Exception as error:
                # 自动标题同样采用最终一致性。提交失败只影响列表稍后仍显示默认
                # 标题，不影响本轮 Agent 已经完成的业务结果。
                logger.exception(
                    "chat_title_submission_failed",
                    error_type=type(error).__name__,
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

        runtime_context = self._build_runtime_context(user_id)

        # guard 必须在 ainvoke 之前取得，并持续覆盖结果解析。interrupt 解析还会用
        # 同一个 config 读取 snapshot；若提前释放，另一个请求可能在解析期间推进状态。
        async with self._hold_thread_execution(
            user_id=user_id,
            public_thread_id=request.thread_id,
        ) as config:
            # 总预算必须覆盖 Graph 和 interrupt 结果解析。后者可能读取 PostgreSQL
            # snapshot；若只给 ainvoke 计时，临界区可能无限持有到 Redis lease 过期。
            async with asyncio.timeout(self._graph_timeout_seconds):
                result = await self._graph.ainvoke(
                    graph_input,
                    config=config,
                    context=runtime_context,
                )

                turn_result = await self._parse_graph_result(
                    result,
                    config=config,
                    thread_id=request.thread_id,
                )

        # 执行到这里说明 timeout 和 guard 都已经正常退出。
        # 后台模型调用不能占用同一 thread 的 Graph 执行权。所以必须放到外面
        # 关键点:
        #   Graph 异常时不会执行提交。
        #   timeout 时不会执行提交。
        #   ChatInterrupt 会进入 helper，但立即返回。
        #   ChatResponse 只提交一次。
        #   后台提取失败不会修改已经生成的 turn_result。
        #   不要在 _parse_graph_result() 中提交，因为它只应负责解析，不应隐藏副作用
        self._submit_completed_turn_enhancements(
            turn_result,
            user_id=user_id,
            source_thread_id=request.thread_id,
            user_message=request.message,
        )

        return turn_result

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
        runtime_context = self._build_runtime_context(user_id)

        # snapshot 检查也属于 resume 的临界区。否则普通请求可能在“检查到 interrupt”
        # 与真正 Command(resume=...) 之间抢先推进同一个 checkpoint。
        async with self._hold_thread_execution(
            user_id=user_id,
            public_thread_id=request.thread_id,
        ) as config:
            # 恢复前读取 snapshot、执行 Command 和解析最终结果共享一个总预算。
            # 这样所有持锁 I/O 都被 90 秒上限覆盖，lease 仍保留 30 秒释放余量。
            async with asyncio.timeout(self._graph_timeout_seconds):
                # 用户 B 即使猜到用户 A 的公开 thread ID，读取的也是 B 自己的内部空间。
                snapshot = await self._graph.aget_state(config)
                interrupts = tuple(interrupt for task in snapshot.tasks for interrupt in task.interrupts)
                if not interrupts:
                    raise ChatResumeNotAvailableError
                if len(interrupts) != 1:
                    raise RuntimeError("resumable chat graph must contain exactly one interrupt")

                result = await self._graph.ainvoke(
                    Command(resume=request.response),
                    config=config,
                    context=runtime_context,
                )

                turn_result = await self._parse_graph_result(
                    result,
                    config=config,
                    thread_id=request.thread_id,
                )

                # resume 值会返回给被中断的工具；长期记忆的用户来源仍应是
                # checkpoint 中触发当前 Agent 流程的最近一条 HumanMessage。
                source_user_message = self._latest_human_message_text(
                    dict(snapshot.values),
                )
        # 此时 checkpoint 的读取、恢复、结果解析和 guard 释放都已经完成。
        # 如果恢复后再次 interrupt，helper 会识别 ChatInterrupt 并跳过提交。
        self._submit_completed_turn_enhancements(
            turn_result,
            user_id=user_id,
            source_thread_id=request.thread_id,
            user_message=source_user_message,
        )

        return turn_result

    @staticmethod
    def _history_message_from(message: object) -> ChatMessage | None:
        """把一条 checkpoint 消息映射为客户端可见的公开消息.

        Args:
            message: LangGraph snapshot 中的 LangChain 消息，静态类型未知。

        Returns:
            映射成功时返回不可变 ChatMessage；不属于公开对话内容的消息返回
            None，由调用方跳过。

        Notes:
            公开 API 只暴露用户与助手的文本消息：ToolMessage、仍带未决
            tool_calls 的 AIMessage 属于 Agent 执行内部状态；多模态或空内容
            无法安全映射为纯文本协议。超长内容截断到公开上限，避免历史中的
            长回复使响应构造直接失败。
        """
        if isinstance(message, HumanMessage):
            role: Literal["user", "assistant"] = "user"
        elif isinstance(message, AIMessage) and not message.tool_calls:
            role = "assistant"
        else:
            return None

        content = message.content
        if not isinstance(content, str):
            return None

        content = content.strip()
        if not content:
            return None

        if len(content) > MAX_MESSAGE_LENGTH:
            content = content[:MAX_MESSAGE_LENGTH]

        return ChatMessage(role=role, content=content)

    async def get_message_history(
        self,
        *,
        session_id: UUID,
        user_id: UUID,
    ) -> ChatMessageHistoryResponse:
        """在当前用户自己的 checkpoint 空间内读取可见历史消息.

        Args:
            session_id: 客户端使用的公开业务会话 UUID。
            user_id: 认证链提供的可信用户 UUID，不能来自请求正文。

        Returns:
            按 checkpoint 顺序排列的用户/助手消息。会话存在但还没有任何
            checkpoint（从未执行过 Agent）时返回空数组。

        Raises:
            ChatSessionNotFoundError: 会话不存在或属于其他用户。与读取、删除
                入口相同的错误语义，跨用户不泄漏资源是否存在。
            SQLAlchemyError: 所有权查询的数据库不可用，继续向上传播。

        Notes:
            这是只读快照查询，不进入 ``_hold_thread_execution`` 执行锁：读取
            历史不应与正在执行的 turn 互斥，否则前端切换会话拉历史会收到
            409。并发删除的最坏结果是读到空快照，没有安全影响；所有权校验
            仍然先行，保证内部 checkpoint key 只能由会话属主构造。
        """
        await self._ownership_verifier.require_owned(
            session_id=session_id,
            user_id=user_id,
        )

        config = self._build_config(
            user_id=user_id,
            public_thread_id=session_id,
        )
        snapshot = await self._graph.aget_state(config)

        state_values = dict(snapshot.values)
        messages = state_values.get("messages")
        if not isinstance(messages, list):
            return ChatMessageHistoryResponse(messages=())

        history: list[ChatMessage] = []
        for message in messages:
            public_message = self._history_message_from(message)
            if public_message is not None:
                history.append(public_message)

        return ChatMessageHistoryResponse(messages=tuple(history))

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
        runtime_context = self._build_runtime_context(user_id)
        # 与非流式入口使用相同的 ChatState 输入契约，保证 ainvoke/astream 不会
        # 因局部变量推断差异产生两套类型边界。
        graph_input: ChatState = {
            "messages": [HumanMessage(content=request.message)],
        }

        try:
            # 只有 Graph 被最终 snapshot 证明已经到达 END 时才赋值。
            # interrupted、timeout、异常和取消路径始终保持 None。
            completed_result: ChatResponse | None = None

            # 异步生成器的函数体要到第一次迭代才执行，因此 SSE 响应头通常已经发出。
            # busy 不能再改成 HTTP 409，必须在下面转换为稳定 ErrorStreamEvent。
            async with self._hold_thread_execution(
                user_id=user_id,
                public_thread_id=request.thread_id,
            ) as config:
                terminal_event: DoneStreamEvent

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
                        terminal_event = DoneStreamEvent(status="interrupted")
                    else:
                        # 没有 interrupt 时，正常完成的图应该已经到达 END，此时 next
                        # 为空。若 next 仍有节点，说明图在未知的非终态停止。
                        if snapshot.next:
                            raise RuntimeError("chat stream ended before graph reached a terminal state")

                        # 不拼接之前 yield 的 token。token 可能来自多个模型阶段，也可能因网络
                        # 断开而不完整。checkpoint 中经过 reducer 合并的最终 AIMessage 才是
                        # 后台记忆提取可以信任的完整助手回复。
                        parsed_result = await self._parse_graph_result(
                            dict(snapshot.values),
                            config=config,
                            thread_id=request.thread_id,
                        )

                        # 当前分支已经排除了 interrupt；若解析器仍返回 ChatInterrupt，说明
                        # snapshot 状态与控制流判断矛盾，不能静默提交记忆。
                        if not isinstance(parsed_result, ChatResponse):
                            raise RuntimeError("completed chat stream must produce a ChatResponse")

                        completed_result = parsed_result

                        terminal_event = DoneStreamEvent(status="completed")

            # 此处已经离开 Graph timeout 和 thread execution guard。
            # completed_result 为 None 时，说明当前流是 interrupted，不能提交。
            if completed_result is not None:
                self._submit_completed_turn_enhancements(
                    completed_result,
                    user_id=user_id,
                    source_thread_id=request.thread_id,
                    user_message=request.message,
                )

            # submit 只创建后台任务，不等待模型提取或数据库写入。
            # 客户端仍然可以立即收到最终 done 事件。
            # 先正常离开 guard 上下文并确认 Redis owner token 已释放，再发送 done。
            # 因此客户端看到 done 时，本轮执行权已经可供下一请求获取。
            yield terminal_event
            return

        except asyncio.CancelledError:
            # 客户端断开时必须让取消继续向上传播。
            # 避免继续执行图、造成资源浪费、模型消耗
            raise
        except ChatSessionNotFoundError:
            # StreamingResponse 的 200 响应头通常已在异步生成器开始迭代前发送，
            # 此时不能再改成 HTTP 404。使用固定流内错误，随后结束本次事件流；
            # 不输出 user_id、内部 key，也不区分不存在和属于其他用户。
            yield ErrorStreamEvent(
                code="CHAT_SESSION_NOT_FOUND",
                message="Chat session was not found",
            )
            return
        except ChatThreadBusyError:
            # SSE 已经开始，只能发送流事件。固定文案不暴露锁名、内部 key 或 owner。
            yield ErrorStreamEvent(
                code="CHAT_THREAD_BUSY",
                message="Chat thread is already being processed",
            )
            return
        except ChatExecutionGuardUnavailableError:
            yield ErrorStreamEvent(
                code="CHAT_EXECUTION_GUARD_UNAVAILABLE",
                message="Chat execution is temporarily unavailable",
            )
            return
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
                thread_id=str(request.thread_id),
            )
            yield ErrorStreamEvent(
                code="CHAT_STREAM_ERROR",
                message="Chat stream failed",
            )
            return
