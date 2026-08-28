"""基于 pypdf 的文本型 PDF parser adapter."""

import asyncio
from collections import Counter
from io import BytesIO

from pypdf import PdfReader
from pypdf.errors import PdfReadError

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


class PdfParser:
    """提取文本型 PDF，并将无文本多页文档识别为需要 OCR."""

    name = "pdf"
    version = "1"
    supported_content_types = frozenset({"application/pdf"})

    async def parse(self, request: ParseRequest) -> ParsedDocument:
        """在线程中执行 PDF 对象解析，避免 CPU/同步读取阻塞事件循环."""
        return await asyncio.to_thread(self._parse_sync, request)

    def _parse_sync(self, request: ParseRequest) -> ParsedDocument:
        try:
            reader = PdfReader(BytesIO(request.content), strict=False)
            if reader.is_encrypted:
                raise DocumentParseError(ParseErrorCode.CORRUPT)
            pages = [normalize_text(page.extract_text() or "") for page in reader.pages]
        except DocumentParseError:
            raise
        except (PdfReadError, ValueError, TypeError, KeyError):
            raise DocumentParseError(ParseErrorCode.CORRUPT) from None

        if not pages:
            raise DocumentParseError(ParseErrorCode.EMPTY)
        if not any(pages):
            # 有页面对象却没有可提取字符，最常见原因是扫描图像。它与真正空文件
            # 不同：后续可以进入 OCR 管线，而不是反复重试同一个文本 parser。
            raise DocumentParseError(ParseErrorCode.OCR_REQUIRED)

        cleaned_pages = self._remove_repeated_margins(pages)
        blocks: list[ParsedBlock] = []
        offset = 0
        for page_number, page_text in enumerate(cleaned_pages, start=1):
            for paragraph in filter(None, (part.strip() for part in page_text.split("\n\n"))):
                blocks.append(
                    ParsedBlock(
                        ordinal=len(blocks),
                        kind=ParsedBlockKind.PARAGRAPH,
                        text=paragraph,
                        location=SourceLocation(
                            page_number=page_number,
                            source_start=offset,
                            source_end=offset + len(paragraph),
                        ),
                    )
                )
                offset += len(paragraph) + 1
        if not blocks:
            raise DocumentParseError(ParseErrorCode.EMPTY)
        return ParsedDocument(
            parser_name=self.name,
            parser_version=self.version,
            content_type=request.content_type,
            blocks=tuple(blocks),
        )

    @staticmethod
    def _remove_repeated_margins(pages: list[str]) -> list[str]:
        """删除多数页面完全相同的首尾行，降低页眉页脚污染."""
        nonempty_lines = [[line for line in page.splitlines() if line.strip()] for page in pages]
        threshold = max(2, (len(pages) + 1) // 2)
        first_counts = Counter(lines[0] for lines in nonempty_lines if lines)
        last_counts = Counter(lines[-1] for lines in nonempty_lines if lines)
        repeated_first = {line for line, count in first_counts.items() if count >= threshold}
        repeated_last = {line for line, count in last_counts.items() if count >= threshold}
        cleaned: list[str] = []
        for lines in nonempty_lines:
            if lines and lines[0] in repeated_first:
                lines = lines[1:]
            if lines and lines[-1] in repeated_last:
                lines = lines[:-1]
            cleaned.append(normalize_text("\n".join(lines)))
        return cleaned


__all__ = ["PdfParser"]
