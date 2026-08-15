"""Node implementations for the minimal chat Agent."""

from collections.abc import Sequence

from app.agents.chat.graph import ChatNode
from app.agents.chat.state import ChatState
from app.services.llm.service import LLMService
from langchain_core.messages import AIMessage


def create_chat_node(
    llm_service: LLMService,
    aliases: Sequence[str],
) -> ChatNode:
    """创建只通过 LLMService 调用模型的异步 chat node."""
    if isinstance(aliases, str) or not aliases:
        raise ValueError("aliases must contain at least one model alias")

    # 保存不可变副本，避免调用方后续修改原始 alias 列表。
    model_aliases = tuple(aliases)

    async def chat_node(state: ChatState) -> ChatState:
        """把当前消息交给 LLMService，并返回一条消息增量."""
        response = await llm_service.call(
            state["messages"],
            aliases=model_aliases,
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
