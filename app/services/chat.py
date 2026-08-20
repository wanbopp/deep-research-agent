"""Application service for running chat Agent turns."""

import asyncio
from dataclasses import dataclass
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command

from app.schemas.chat import ChatMessage, ChatRequest, ChatResponse, ChatResumeRequest


@dataclass(frozen=True, slots=True)
class ChatInterrupt:
    """Agent 暂停后交给上层处理的稳定应用结果."""

    thread_id: str
    question: str


ChatTurnResult = ChatResponse | ChatInterrupt


class ChatService:
    """在 API 模型和 LangGraph runtime 之间执行一次聊天用例."""

    def __init__(
        self,
        graph: CompiledStateGraph,
        *,
        graph_timeout_seconds: float = 90.0,
    ) -> None:
        """保存共享 graph，并验证单轮图执行的时间预算."""
        if graph_timeout_seconds <= 0:
            raise ValueError("graph_timeout_seconds must be greater than 0")

        self._graph = graph
        self._graph_timeout_seconds = graph_timeout_seconds

    @staticmethod
    def _build_config(thread_id: str) -> RunnableConfig:
        """为指定会话构造 LangGraph 运行配置."""
        # 根据 thread_id 创建 RunnableConfig。
        config: RunnableConfig = {
            "configurable": {
                "thread_id": thread_id,
            },
            "recursion_limit": 8,
        }
        return config

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
    ) -> ChatTurnResult:
        """执行一轮 Agent，并返回完成或暂停结果."""
        graph_input = {
            "messages": [
                HumanMessage(content=request.message),
            ]
        }

        config = self._build_config(request.thread_id)

        # 在整图时间预算内调用 ainvoke
        result = await asyncio.wait_for(
            self._graph.ainvoke(
                graph_input,
                config=config,
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
    ) -> ChatTurnResult:
        """使用人工回答恢复已暂停的 Agent."""
        config = self._build_config(request.thread_id)

        result = await asyncio.wait_for(
            self._graph.ainvoke(
                Command(resume=request.response),
                config=config,
            ),
            timeout=self._graph_timeout_seconds,
        )

        return await self._parse_graph_result(
            result,
            config=config,
            thread_id=request.thread_id,
        )
