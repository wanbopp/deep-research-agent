"""保留标题、代码、表格和列表结构的 Markdown parser."""

import asyncio
import re

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

_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*$")


class MarkdownParser:
    """使用确定性行扫描解析 Markdown，不执行 HTML 或外部资源."""

    name = "markdown"
    version = "1"
    supported_content_types = frozenset({"text/markdown", "text/x-markdown"})

    async def parse(self, request: ParseRequest) -> ParsedDocument:
        """在线程中解码和扫描 Markdown，避免阻塞异步 worker 事件循环."""
        return await asyncio.to_thread(self._parse_sync, request)

    def _parse_sync(self, request: ParseRequest) -> ParsedDocument:
        try:
            source = normalize_text(request.content.decode("utf-8-sig"))
        except UnicodeDecodeError:
            raise DocumentParseError(ParseErrorCode.CORRUPT) from None
        if not source:
            raise DocumentParseError(ParseErrorCode.EMPTY)

        blocks: list[ParsedBlock] = []
        section_levels: dict[int, str] = {}
        cursor = 0
        lines = source.splitlines(keepends=True)
        index = 0
        while index < len(lines):
            raw_line = lines[index]
            line = raw_line.rstrip("\n")
            line_start = cursor
            cursor += len(raw_line)
            if not line.strip():
                index += 1
                continue

            heading = _HEADING.match(line)
            if heading:
                level = len(heading.group(1))
                title = heading.group(2).strip()
                section_levels = {key: value for key, value in section_levels.items() if key < level}
                section_levels[level] = title
                blocks.append(
                    self._block(
                        blocks=blocks,
                        kind=ParsedBlockKind.HEADING,
                        text=title,
                        section_path=tuple(section_levels[key] for key in sorted(section_levels)),
                        start=line_start,
                        end=line_start + len(line),
                    )
                )
                index += 1
                continue

            block_lines = [line]
            kind = self._kind(line)
            in_fence = line.lstrip().startswith("```")
            while index + 1 < len(lines):
                next_line = lines[index + 1].rstrip("\n")
                if not in_fence and (not next_line.strip() or _HEADING.match(next_line)):
                    break
                index += 1
                consumed = lines[index]
                cursor += len(consumed)
                block_lines.append(next_line)
                if next_line.lstrip().startswith("```"):
                    in_fence = not in_fence
            text = "\n".join(block_lines).strip()
            blocks.append(
                self._block(
                    blocks=blocks,
                    kind=kind,
                    text=text,
                    section_path=tuple(section_levels[key] for key in sorted(section_levels)),
                    start=line_start,
                    end=line_start + len(text),
                )
            )
            index += 1

        if not blocks:
            raise DocumentParseError(ParseErrorCode.EMPTY)
        return ParsedDocument(
            parser_name=self.name,
            parser_version=self.version,
            content_type=request.content_type,
            blocks=tuple(blocks),
        )

    @staticmethod
    def _kind(line: str) -> ParsedBlockKind:
        stripped = line.lstrip()
        if stripped.startswith("```"):
            return ParsedBlockKind.CODE
        if stripped.startswith(("- ", "* ", "+ ")) or re.match(r"\d+[.)]\s", stripped):
            return ParsedBlockKind.LIST
        if "|" in line:
            return ParsedBlockKind.TABLE
        return ParsedBlockKind.PARAGRAPH

    @staticmethod
    def _block(
        *,
        blocks: list[ParsedBlock],
        kind: ParsedBlockKind,
        text: str,
        section_path: tuple[str, ...],
        start: int,
        end: int,
    ) -> ParsedBlock:
        return ParsedBlock(
            ordinal=len(blocks),
            kind=kind,
            text=text,
            location=SourceLocation(
                section_path=section_path,
                source_start=start,
                source_end=end,
            ),
        )


__all__ = ["MarkdownParser"]
