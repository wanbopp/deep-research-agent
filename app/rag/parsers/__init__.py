"""文档解析契约与具体格式 adapter."""

from app.rag.parsers.contracts import (
    DocumentParseError,
    DocumentParser,
    ParseErrorCode,
    ParseRequest,
    ParsedBlock,
    ParsedBlockKind,
    ParsedDocument,
    SourceLocation,
)
from app.rag.parsers.docx import DocxParser
from app.rag.parsers.markdown import MarkdownParser
from app.rag.parsers.pdf import PdfParser
from app.rag.parsers.plain_text import PlainTextParser
from app.rag.parsers.registry import ParserRegistry

__all__ = [
    "DocumentParseError",
    "DocumentParser",
    "DocxParser",
    "MarkdownParser",
    "ParseErrorCode",
    "ParseRequest",
    "ParsedBlock",
    "ParsedBlockKind",
    "ParsedDocument",
    "ParserRegistry",
    "PdfParser",
    "PlainTextParser",
    "SourceLocation",
]
