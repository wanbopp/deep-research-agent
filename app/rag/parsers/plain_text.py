"""UTF-8 纯文本的最小确定性 parser."""

import asyncio

from app.rag.parsers.contracts import (
    DocumentParseError,
    ParseErrorCode,
    ParseRequest,
    ParsedBlock,
    ParsedBlockKind,
    ParsedDocument,
    SourceLocation,
)
from app.rag.parsers.normalization import normalize_text


class PlainTextParser:
    """按空行提取纯文本段落，不把井号或竖线解释成 Markdown 语法."""

    name = "plain-text"
    version = "1"
    supported_content_types = frozenset({"text/plain"})

    async def parse(self, request: ParseRequest) -> ParsedDocument:
        """在线程中解码 UTF-8 文本并保持稳定段落顺序."""
        return await asyncio.to_thread(self._parse_sync, request)

    def _parse_sync(self, request: ParseRequest) -> ParsedDocument:
        try:
            text = normalize_text(request.content.decode("utf-8-sig"))
        except UnicodeDecodeError:
            raise DocumentParseError(ParseErrorCode.CORRUPT) from None
        if not text:
            raise DocumentParseError(ParseErrorCode.EMPTY)
        blocks: list[ParsedBlock] = []
        cursor = 0
        for paragraph in text.split("\n\n"):
            start = text.find(paragraph, cursor)
            blocks.append(
                ParsedBlock(
                    ordinal=len(blocks),
                    kind=ParsedBlockKind.PARAGRAPH,
                    text=paragraph,
                    location=SourceLocation(source_start=start, source_end=start + len(paragraph)),
                )
            )
            cursor = start + len(paragraph)
        return ParsedDocument(
            parser_name=self.name,
            parser_version=self.version,
            content_type=request.content_type,
            blocks=tuple(blocks),
        )


__all__ = ["PlainTextParser"]
