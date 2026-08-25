"""可信服务端上下文定义，供 Chat Agent 图执行期间使用."""

from dataclasses import dataclass
from uuid import UUID

__all__ = ["ChatRuntimeContext"]


@dataclass(frozen=True, slots=True)
class ChatRuntimeContext:
    """一次 Chat Agent 图执行所使用的可信服务端上下文.

    该对象由服务端在调用 ``graph.ainvoke(..., context=...)`` 或
    ``graph.astream(..., context=...)`` 时创建。它不是 ChatState 的一部分，
    因而不会由模型消息、节点状态增量或 ToolCall 参数写入和覆盖。

    Attributes:
        user_id: 当前认证用户的 UUID。该值只能来自 ``get_current_user`` 返回的
            ``AuthenticatedUser.user_id``，不能来自请求正文、Prompt、模型输出或
            工具参数。dataclass 自动生成的 ``__init__`` 也只接收这一个参数。

    Notes:
        ``frozen=True`` 阻止节点在运行期间重新赋值 user_id；``slots=True`` 则固定
        对象字段集合，避免临时附加 JWT、email 等不应进入 Agent runtime 的数据。
        这两项只减少误用，真正的信任仍来自 API dependency 已完成的验签和查库。
    """

    user_id: UUID
