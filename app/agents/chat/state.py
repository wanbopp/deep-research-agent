"""State contracts for minimal chat Agent."""

from typing import TypedDict, Annotated

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class ChatState(TypedDict):
    """保存最小聊天图在节点之间的共享的状态."""

    # 新 ID 的消息会追加到历史；相同 ID 的消息会替换已有消息。
    # 因此 node 只需返回本次消息增量，不需要复制完整历史。
    messages: Annotated[list[AnyMessage], add_messages]
