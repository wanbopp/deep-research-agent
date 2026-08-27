"""聊天 Agent 的完整状态与节点局部更新契约."""

from typing import Annotated, NotRequired, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.channels import UntrackedValue
from langgraph.graph.message import add_messages

from app.schemas.memory import MemoryItem, MemorySearchStatus


class ChatState(TypedDict):
    """描述 LangGraph 在节点之间维护的完整聊天状态."""

    # 新 ID 的消息会追加到历史；相同 ID 的消息会替换已有消息。
    # 因此 node 只需返回本次消息增量，不需要复制完整历史。
    messages: Annotated[list[AnyMessage], add_messages]

    # 当前 invocation 检索到的长期记忆。
    # NotRequired 表示恢复旧 checkpoint 或进入 memory node 前可以不存在。
    # UntrackedValue 表示该值只在当前 Graph invocation 中传递，不写入 checkpoint。
    memory_context: Annotated[
        NotRequired[tuple[MemoryItem, ...]],
        UntrackedValue,
    ]

    # 明确区分正常空结果与 MemoryStore 降级。
    memory_status: Annotated[
        NotRequired[MemorySearchStatus],
        UntrackedValue,
    ]


class ChatStateUpdate(TypedDict, total=False):
    """描述一个 Agent node 对共享状态产生的局部更新.

    ``total=False`` 表示所有字段都可以省略。节点只返回自己真正修改的字段，
    LangGraph 再按照 ``ChatState`` 中声明的 channel 规则完成合并。

    ``Annotated``、reducer 和 ``UntrackedValue`` 是合并规则，定义在完整状态上；
    节点返回的是待合并数据。因此更新 schema 不应重复声明 channel 规则。
    """

    # chat/tools node 写入新消息，最终由 add_messages 合并。
    messages: list[AnyMessage]

    # memory node 写入当前 invocation 的检索结果。
    memory_context: tuple[MemoryItem, ...]

    # memory node 同时写入 available/degraded 状态。
    memory_status: MemorySearchStatus
