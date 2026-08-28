"""与具体文件格式无关的文档解析输入、输出和失败契约."""

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class ParseErrorCode(StrEnum):
    """可以安全写入 IndexJob 的稳定解析失败分类.

    这里保存的是应用能够处理的类别，不是第三方解析库的原始异常文本。
    原始异常可能包含文件路径或正文片段，因此不能进入数据库或公开响应。
    """

    UNSUPPORTED = "PARSER_UNSUPPORTED"
    CORRUPT = "PARSER_CORRUPT"
    EMPTY = "PARSER_EMPTY"
    OCR_REQUIRED = "PARSER_OCR_REQUIRED"


class ParsedBlockKind(StrEnum):
    """解析器识别出的结构类型，供后续 chunker 决定边界优先级."""

    PARAGRAPH = "paragraph"
    HEADING = "heading"
    LIST = "list"
    CODE = "code"
    TABLE = "table"
    OTHER = "other"


@dataclass(frozen=True, slots=True)
class ParseRequest:
    """一次 parser 调用所需的最小可信输入.

    Attributes:
        filename: 仅用于格式诊断和来源展示的安全 basename，不能当作文件路径打开。
        content_type: KnowledgeService 已校验并保存的 MIME 类型。
        content_sha256: 原始文件的内容摘要。后续确定性 chunk ID 会把它作为输入，
            因此重复解析同一内容时仍能获得稳定身份。
        content: FileStorage 读取到的原始字节；parser 不负责访问数据库或文件系统。

    解析器接收完整值对象而不是 ``Document`` ORM 实例，这样解析层不会意外依赖
    Session、owner 或 storage key，也更容易在 worker 与离线脚本中复用。
    """

    filename: str
    content_type: str
    content_sha256: str
    content: bytes

    def __post_init__(self) -> None:
        """拒绝不完整或不可信的 parser 输入，并规范化 MIME/摘要大小写."""
        filename = self.filename.strip()
        content_type = self.content_type.strip().lower()
        content_sha256 = self.content_sha256.strip().lower()

        if not filename:
            raise ValueError("filename must not be empty")
        if not content_type:
            raise ValueError("content_type must not be empty")
        if not self.content:
            raise ValueError("content must not be empty")
        if len(content_sha256) != 64 or any(character not in "0123456789abcdef" for character in content_sha256):
            raise ValueError("content_sha256 must be a 64-character hexadecimal digest")

        object.__setattr__(self, "filename", filename)
        object.__setattr__(self, "content_type", content_type)
        object.__setattr__(self, "content_sha256", content_sha256)


class _FrozenParserModel(BaseModel):
    """解析值对象的共同严格配置，避免运行中被修改或吞掉拼错字段."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class SourceLocation(_FrozenParserModel):
    """一个解析块在原文结构中的可回查位置.

    ``page_number`` 面向 PDF 等分页格式，使用从 1 开始的展示页码；
    ``section_path`` 面向 Markdown/DOCX 的标题层级，例如
    ``("安装", "Windows")``。两者都允许缺失，因为某些文档只有线性正文。

    ``source_start``/``source_end`` 是解析器规范化文本流中的半开字符区间
    ``[start, end)``，不是原始文件的 byte offset。字符偏移量不会因为中文 UTF-8
    字符占多个字节而产生歧义。具体 parser 若不能可靠提供 offset，应同时留空，
    不能伪造一个看似精确的位置。
    """

    page_number: int | None = Field(default=None, ge=1)
    section_path: tuple[str, ...] = ()
    source_start: int | None = Field(default=None, ge=0)
    source_end: int | None = Field(default=None, ge=0)

    @field_validator("section_path")
    @classmethod
    def validate_section_path(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        """清理标题边界空白，同时拒绝会破坏引用展示的空标题."""
        normalized = tuple(part.strip() for part in value)
        if any(not part for part in normalized):
            raise ValueError("section_path must not contain empty parts")
        return normalized

    @model_validator(mode="after")
    def validate_source_range(self) -> "SourceLocation":
        """保证 offset 要么都不存在，要么构成非空半开区间."""
        has_start = self.source_start is not None
        has_end = self.source_end is not None
        if has_start != has_end:
            raise ValueError("source_start and source_end must be provided together")
        if has_start and self.source_end <= self.source_start:  # type: ignore[operator]
            raise ValueError("source_end must be greater than source_start")
        return self


class ParsedBlock(_FrozenParserModel):
    """parser 保留下来的最小结构化正文单元.

    ``ordinal`` 只表达 parser 输出顺序；它不是最终 chunk 序号。一个 chunk 后续
    可以合并多个 block，也可以拆分一个超长 block。保留 kind 和 location，正是为了
    让 chunker 在改变边界之后仍能构造可回查引用。
    """

    ordinal: int = Field(ge=0)
    kind: ParsedBlockKind
    text: str
    location: SourceLocation

    @field_validator("text")
    @classmethod
    def validate_text(cls, value: str) -> str:
        """拒绝空白块，但保留正文原有边界，避免 offset 与文本悄悄错位."""
        if not value.strip():
            raise ValueError("parsed block text must not be blank")
        return value


class ParsedDocument(_FrozenParserModel):
    """一次确定性解析产生的完整、不可变结果."""

    parser_name: str = Field(min_length=1)
    parser_version: str = Field(min_length=1)
    content_type: str = Field(min_length=1)
    blocks: tuple[ParsedBlock, ...] = Field(min_length=1)

    @field_validator("parser_name", "parser_version", "content_type")
    @classmethod
    def normalize_identity(cls, value: str) -> str:
        """规范化会参与缓存、日志或确定性身份计算的 parser 标识."""
        normalized = value.strip()
        if not normalized:
            raise ValueError("parser identity fields must not be blank")
        return normalized

    @model_validator(mode="after")
    def validate_block_order(self) -> "ParsedDocument":
        """要求序号从 0 连续递增，使相同解析结果拥有唯一、稳定的顺序."""
        ordinals = tuple(block.ordinal for block in self.blocks)
        expected = tuple(range(len(self.blocks)))
        if ordinals != expected:
            raise ValueError("parsed block ordinals must be contiguous and start at zero")
        return self


class DocumentParseError(RuntimeError):
    """parser 对可预期文档问题给出的稳定、安全失败."""

    def __init__(self, code: ParseErrorCode) -> None:
        """保存稳定错误码，不把第三方异常或文档正文放进异常消息.

        Args:
            code: 可持久化、可重试策略可识别的解析失败分类。
        """
        self.code = code
        super().__init__("Document parsing failed")


class DocumentParser(Protocol):
    """具体 PDF、Markdown、DOCX adapter 必须实现的最小解析能力."""

    @property
    def name(self) -> str:
        """返回稳定 parser 名称，用于追踪结果由哪个实现产生."""
        ...

    @property
    def version(self) -> str:
        """返回行为版本；清洗规则改变时必须升级，避免旧结果被误当成等价结果."""
        ...

    @property
    def supported_content_types(self) -> frozenset[str]:
        """返回该 adapter 可以可靠解析的 MIME 集合."""
        ...

    async def parse(self, request: ParseRequest) -> ParsedDocument:
        """把原始字节转换为有顺序、有来源位置的结构化正文.

        Args:
            request: 已从 FileStorage 读取且带可信内容摘要的解析请求。

        Returns:
            不可变、顺序稳定的解析结果。

        Raises:
            DocumentParseError: 文档不支持、损坏、无正文或需要 OCR。
        """
        ...


__all__ = [
    "DocumentParseError",
    "DocumentParser",
    "ParseErrorCode",
    "ParseRequest",
    "ParsedBlock",
    "ParsedBlockKind",
    "ParsedDocument",
    "SourceLocation",
]
