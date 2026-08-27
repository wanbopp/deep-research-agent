"""自动会话标题的结构化模型输出边界."""

from typing import Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.models import DEFAULT_CHAT_SESSION_TITLE

MAX_AUTO_CHAT_SESSION_TITLE_LENGTH = 80


class ChatSessionTitleResult(BaseModel):
    """标题模型必须返回的最小结构.

    这里只接受展示标题，不允许模型返回 user_id、thread_id、数据库状态或执行
    指令。服务端会在结构化解析后再次执行长度和默认值校验。
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    title: str = Field(
        min_length=1,
        max_length=MAX_AUTO_CHAT_SESSION_TITLE_LENGTH,
        description="概括本次会话主题的简短展示标题",
    )

    @field_validator("title", mode="before")
    @classmethod
    def normalize_title(cls, value: object) -> object:
        """把模型常见的换行和重复空白压缩成单行标题."""
        if not isinstance(value, str):
            return value
        return " ".join(value.split())

    @model_validator(mode="after")
    def reject_default_placeholder(self) -> Self:
        """拒绝没有提供任何主题信息的系统占位标题."""
        if self.title.casefold() == DEFAULT_CHAT_SESSION_TITLE.casefold():
            raise ValueError("generated title must replace the default placeholder")
        return self


__all__ = [
    "ChatSessionTitleResult",
    "MAX_AUTO_CHAT_SESSION_TITLE_LENGTH",
]
