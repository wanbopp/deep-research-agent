"""按 MIME 类型选择文档解析器的严格注册表."""

from collections.abc import Iterable

from app.rag.parsers.contracts import (
    DocumentParseError,
    DocumentParser,
    ParseErrorCode,
    ParseRequest,
    ParsedDocument,
)


class ParserRegistry:
    """把格式分派集中在一个地方，避免 worker 了解每种文件格式."""

    def __init__(self, parsers: Iterable[DocumentParser]) -> None:
        """建立 MIME 到 parser 的不可变映射，并拒绝含糊的重复声明."""
        by_content_type: dict[str, DocumentParser] = {}
        for parser in parsers:
            if not parser.name.strip() or not parser.version.strip():
                raise ValueError("Parser name and version must not be empty")
            for content_type in parser.supported_content_types:
                normalized = content_type.strip().lower()
                if not normalized:
                    raise ValueError("Parser content type must not be empty")
                if normalized in by_content_type:
                    raise ValueError(f"Duplicate parser for content type: {normalized}")
                by_content_type[normalized] = parser
        if not by_content_type:
            raise ValueError("Parser registry must not be empty")
        self._by_content_type = by_content_type

    def resolve(self, content_type: str) -> DocumentParser:
        """返回精确匹配 MIME 的 parser；未知格式使用稳定业务错误."""
        parser = self._by_content_type.get(content_type.strip().lower())
        if parser is None:
            raise DocumentParseError(ParseErrorCode.UNSUPPORTED)
        return parser

    async def parse(self, request: ParseRequest) -> ParsedDocument:
        """分派解析并验证 adapter 没有返回错误 MIME 或身份信息."""
        parser = self.resolve(request.content_type)
        result = await parser.parse(request)
        if result.content_type != request.content_type:
            raise RuntimeError("Parser returned a mismatched content type")
        if result.parser_name != parser.name or result.parser_version != parser.version:
            raise RuntimeError("Parser returned mismatched identity metadata")
        return result


__all__ = ["ParserRegistry"]
