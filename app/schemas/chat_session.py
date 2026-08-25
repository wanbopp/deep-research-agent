"""业务聊天会话 API 的请求与响应模型.

本模块只描述客户端可见的产品资源，不保存 LangGraph 的 messages、节点位置、
interrupt 或 checkpoint 数据。业务 ``ChatSession`` 负责回答“这个会话属于谁”，
checkpointer 则负责回答“这个 Agent 执行到了哪里”，两者不能混成同一个模型。
"""

from typing import Self
from uuid import UUID

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, field_validator, model_validator

DEFAULT_CHAT_SESSION_TITLE = "New chat"
MAX_CHAT_SESSION_TITLE_LENGTH = 200


def _normalize_title(value: object) -> object:
    """去除字符串标题首尾空白，同时保留非法类型交给 Pydantic 拒绝.

    Args:
        value: 尚未完成类型校验的外部输入。

    Returns:
        清理后的字符串，或未经转换的非字符串值。

    这里不能使用 ``str(value)``。强制转换会把整数等错误输入变成看似合法的
    标题，从而掩盖客户端协议错误。
    """
    return value.strip() if isinstance(value, str) else value


class _StrictChatSessionModel(BaseModel):
    """为业务会话 API 模型提供统一的严格配置."""

    # extra="forbid" 让拼错字段立即失败；frozen=True 防止已经校验的数据
    # 在 route、service 和响应组装之间被意外修改。
    model_config = ConfigDict(extra="forbid", frozen=True)


class ChatSessionCreateRequest(_StrictChatSessionModel):
    """创建业务聊天会话时接收的公开请求.

    客户端只可以决定展示标题，不能提交 user_id、内部 checkpoint key 或数据库
    主键。可信 user_id 来自 Bearer token，会话 UUID 则由服务端创建。
    """

    title: str = Field(
        default=DEFAULT_CHAT_SESSION_TITLE,
        min_length=1,
        max_length=MAX_CHAT_SESSION_TITLE_LENGTH,
        description="客户端可见的会话标题",
    )

    @field_validator("title", mode="before")
    @classmethod
    def normalize_title(cls, value: object) -> object:
        """先清理标题，再执行非空和长度约束."""
        return _normalize_title(value)


class ChatSessionResponse(_StrictChatSessionModel):
    """向客户端公开的一条业务聊天会话.

    ``thread_id`` 采用业务 ``ChatSession.id`` 的 UUID，而不是直接暴露内部
    ``user_id:thread_id`` checkpoint key。UUID 只是资源标识；route 仍必须使用
    ``thread_id + 当前认证 user_id`` 查询，不能把“难以猜测”误当成授权。
    """

    thread_id: UUID = Field(description="业务会话 UUID，也是客户端使用的公开 thread ID")
    title: str = Field(
        min_length=1,
        max_length=MAX_CHAT_SESSION_TITLE_LENGTH,
        description="客户端可见的会话标题",
    )
    # AwareDatetime 会拒绝无时区 datetime，避免客户端把服务器本地时间误解为 UTC。
    created_at: AwareDatetime
    updated_at: AwareDatetime

    @field_validator("title", mode="before")
    @classmethod
    def normalize_title(cls, value: object) -> object:
        """保证从 service 组装的响应也遵守标题规范化规则."""
        return _normalize_title(value)

    @model_validator(mode="after")
    def validate_timestamp_order(self) -> Self:
        """拒绝更新时间早于创建时间的矛盾资源状态."""
        if self.updated_at < self.created_at:
            raise ValueError("updated_at must not be earlier than created_at")
        return self


class ChatSessionListResponse(_StrictChatSessionModel):
    """列出当前认证用户可见的业务聊天会话."""

    # tuple 与 frozen 模型配合，避免调用方通过 sessions.append() 绕过模型冻结。
    # Pydantic 在 JSON 中仍会把 tuple 序列化为标准数组，客户端无需了解 Python 类型。
    sessions: tuple[ChatSessionResponse, ...] = ()


__all__ = [
    "ChatSessionCreateRequest",
    "ChatSessionListResponse",
    "ChatSessionResponse",
    "DEFAULT_CHAT_SESSION_TITLE",
    "MAX_CHAT_SESSION_TITLE_LENGTH",
]
