"""Public request and response models for the chat API.

这些模型属于 HTTP/API 层，不直接暴露 LangChain 的 HumanMessage、AIMessage
或 LangGraph 的 ChatState。这样框架内部对象发生变化时，公开 API 仍能保持稳定。
"""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

THREAD_ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._:-]*$"
MAX_THREAD_ID_LENGTH = 128
MAX_MESSAGE_LENGTH = 8000


def _strip_text(value: object) -> object:
    """在字段约束执行前去除字符串首尾空白.

    非字符串值保持原样，让 Pydantic 的字段类型统一负责拒绝它。这里不使用
    ``str(value)`` 强制转换，否则整数等非法输入可能被悄悄接受为聊天文本。
    """
    return value.strip() if isinstance(value, str) else value


class ChatMessage(BaseModel):
    """客户端可见的一条聊天消息."""

    # extra="forbid" 让拼错的字段立即失败；frozen=True 防止已经验证的
    # API 对象在 route/service 传递过程中被意外修改。
    model_config = ConfigDict(extra="forbid", frozen=True)

    role: Literal["user", "assistant"]
    content: str = Field(
        min_length=1,
        max_length=MAX_MESSAGE_LENGTH,
        description="客户端可见的聊天文本",
    )

    @field_validator("content", mode="before")
    @classmethod
    def normalize_content(cls, value: object) -> object:
        """先去除空白，再执行字符串类型和长度约束."""
        return _strip_text(value)


class ChatRequest(BaseModel):
    """提交给非流式聊天接口的一条新用户输入."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    # 当前阶段由客户端临时提供 thread_id。Phase 4 引入身份与持久化后，
    # 服务端会把它映射到经过授权的会话资源。
    thread_id: str = Field(
        min_length=1,
        max_length=MAX_THREAD_ID_LENGTH,
        pattern=THREAD_ID_PATTERN,
        description="LangGraph checkpoint 使用的临时会话标识",
    )
    # 每次请求只提交当前一条用户输入；既有历史由 checkpointer 按
    # thread_id 保存，避免客户端重复提交完整历史。
    message: str = Field(
        min_length=1,
        max_length=MAX_MESSAGE_LENGTH,
        description="本轮新增的用户消息",
    )

    @field_validator("thread_id", "message", mode="before")
    @classmethod
    def normalize_text_fields(cls, value: object) -> object:
        """在 pattern 和长度检查前统一去除首尾空白."""
        return _strip_text(value)


class ChatResponse(BaseModel):
    """非流式聊天接口返回的最终助手消息."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    # 响应回传 thread_id，客户端才能在下一轮继续同一个 checkpoint。
    thread_id: str = Field(
        min_length=1,
        max_length=MAX_THREAD_ID_LENGTH,
        pattern=THREAD_ID_PATTERN,
    )

    # Literal 不只表示字段是字符串，还把可接受的值收窄为
    # "completed"。它与 ChatInterruptResponse.status 共同构成判别字段。
    status: Literal["completed"] = "completed"

    # 第一版 API 只公开最终用户/助手消息，不暴露 ToolMessage、provider
    # metadata、token usage 或 LangGraph checkpoint 内部结构。
    message: ChatMessage

    @field_validator("thread_id", mode="before")
    @classmethod
    def normalize_thread_id(cls, value: object) -> object:
        """让独立构造的响应也遵守与请求相同的 thread_id 规则."""
        return _strip_text(value)


class ChatInterruptResponse(BaseModel):
    """Agent 暂停并等待人工回答时的公开响应."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    status: Literal["interrupted"] = "interrupted"
    thread_id: str = Field(
        min_length=1,
        max_length=MAX_THREAD_ID_LENGTH,
        pattern=THREAD_ID_PATTERN,
    )
    question: str = Field(
        min_length=1,
        max_length=MAX_MESSAGE_LENGTH,
    )

    @field_validator("thread_id", "question", mode="before")
    @classmethod
    def normalize_text_fields(cls, value: object) -> object:
        """清理会话标识和人工问题的首尾空白."""
        return _strip_text(value)


class ChatResumeRequest(BaseModel):
    """为已暂停的 Agent 提供人工回答."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    thread_id: str = Field(
        min_length=1,
        max_length=MAX_THREAD_ID_LENGTH,
        pattern=THREAD_ID_PATTERN,
    )
    response: str = Field(
        min_length=1,
        max_length=MAX_MESSAGE_LENGTH,
    )

    @field_validator("thread_id", "response", mode="before")
    @classmethod
    def normalize_text_fields(cls, value: object) -> object:
        """清理会话标识和恢复回答的首尾空白."""
        return _strip_text(value)


# status 是判别字段。Pydantic 根据它的 Literal 值选择唯一响应模型，
# FastAPI 也会据此生成带 discriminator 的 OpenAPI oneOf。
ChatAPIResponse = Annotated[
    ChatResponse | ChatInterruptResponse,
    Field(discriminator="status"),
]


# 流事件在应用层表达“Agent 正在发生什么”；下一层才会把
# 这些对象编码成 SSE event/data 文本。
class _ChatStreamEventBase(BaseModel):
    """所有聊天流事件共享的严格模型配置."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class TokenStreamEvent(_ChatStreamEventBase):
    """模型生成的一段客户端可见文本."""

    event: Literal["token"] = "token"
    text: str = Field(min_length=1)


class ToolStreamEvent(_ChatStreamEventBase):
    """工具调用开始或结束事件."""

    event: Literal["tool"] = "tool"
    name: str = Field(min_length=1, max_length=128)
    tool_call_id: str = Field(min_length=1, max_length=256)
    status: Literal["started", "success", "error"]


class InterruptStreamEvent(_ChatStreamEventBase):
    """Agent 等待人工输入."""

    event: Literal["interrupt"] = "interrupt"
    question: str = Field(min_length=1, max_length=MAX_MESSAGE_LENGTH)


class ErrorStreamEvent(_ChatStreamEventBase):
    """流开始后发生的安全错误事件."""

    event: Literal["error"] = "error"
    code: str = Field(min_length=1, max_length=128)
    message: str = Field(min_length=1, max_length=512)


class DoneStreamEvent(_ChatStreamEventBase):
    """一次流正常结束."""

    event: Literal["done"] = "done"
    status: Literal["completed", "interrupted"]


# event 是判别字段；Pydantic 根据它直接选择唯一事件模型。
ChatStreamEvent = Annotated[
    TokenStreamEvent | ToolStreamEvent | InterruptStreamEvent | ErrorStreamEvent | DoneStreamEvent,
    Field(discriminator="event"),
]
