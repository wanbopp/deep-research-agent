"""文档解析契约的聚焦门禁."""

from pydantic import ValidationError
import pytest

from app.rag.parsers import (
    ParseRequest,
    ParsedBlock,
    ParsedBlockKind,
    ParsedDocument,
    SourceLocation,
)


def test_parser_contract_preserves_provenance_and_rejects_ambiguous_output() -> None:
    """解析结果必须可定位、顺序确定，并拒绝容易制造错误引用的形状."""
    request = ParseRequest(
        filename="  guide.md  ",
        content_type="TEXT/MARKDOWN",
        content_sha256="A" * 64,
        content="第一章\n正文".encode(),
    )
    location = SourceLocation(
        section_path=(" 第一章 ",),
        source_start=0,
        source_end=5,
    )
    block = ParsedBlock(
        ordinal=0,
        kind=ParsedBlockKind.PARAGRAPH,
        text="第一章\n正文",
        location=location,
    )
    document = ParsedDocument(
        parser_name="markdown",
        parser_version="1",
        content_type=request.content_type,
        blocks=(block,),
    )

    assert request.filename == "guide.md"
    assert request.content_type == "text/markdown"
    assert request.content_sha256 == "a" * 64
    assert document.blocks[0].text == "第一章\n正文"
    assert document.blocks[0].location.section_path == ("第一章",)

    # 单边 offset 无法表达范围；不连续 ordinal 会让确定性 chunk 身份发生歧义。
    with pytest.raises(ValidationError, match="provided together"):
        SourceLocation(source_start=0)
    with pytest.raises(ValidationError, match="contiguous"):
        ParsedDocument(
            parser_name="markdown",
            parser_version="1",
            content_type="text/markdown",
            blocks=(block.model_copy(update={"ordinal": 1}),),
        )
