"""保留来源位置的确定性 token-aware 文档分块器."""

from hashlib import sha256
from uuid import UUID

import tiktoken
from pydantic import BaseModel, ConfigDict, Field

from app.rag.parsers import ParsedDocument, SourceLocation


class ChunkSource(BaseModel):
    """一个 chunk 所覆盖的 parser block 与原文位置."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    block_ordinal: int = Field(ge=0)
    location: SourceLocation


class TextChunk(BaseModel):
    """准备持久化和 embedding 的确定性文本单元."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: UUID
    ordinal: int = Field(ge=0)
    text: str = Field(min_length=1)
    token_count: int = Field(gt=0)
    content_sha256: str = Field(min_length=64, max_length=64)
    sources: tuple[ChunkSource, ...] = Field(min_length=1)
    parser_name: str
    parser_version: str
    chunker_version: str


class TokenAwareChunker:
    """按真实 tokenizer 预算切分，并生成可重复计算的 chunk ID.

    短 block 会在不超过预算时合并；超长 block 使用 token 滑动窗口切开。
    overlap 仅在切分超长正文时复制尾部 token，避免把相邻标题路径混在一起。
    """

    version = "1"

    def __init__(
        self,
        *,
        chunk_size: int,
        chunk_overlap: int,
        encoding_name: str = "cl100k_base",
    ) -> None:
        """保存稳定分块配置.

        Args:
            chunk_size: 单个 chunk 的最大 token 数。
            chunk_overlap: 超长 block 相邻窗口重复的 token 数，必须小于 chunk_size。
            encoding_name: 固定 tokenizer 名称；改变它会改变边界，应视为配置版本变化。
        """
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than zero")
        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be between zero and chunk_size")
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._encoding_name = encoding_name
        self._encoding = tiktoken.get_encoding(encoding_name)

    def split(
        self,
        *,
        document_id: UUID,
        content_sha256: str,
        document: ParsedDocument,
    ) -> tuple[TextChunk, ...]:
        """把 parser blocks 转为顺序稳定且携带 provenance 的 chunks."""
        groups: list[list[tuple[str, int, ChunkSource]]] = []
        current: list[tuple[str, int, ChunkSource]] = []
        current_tokens = 0

        for block in document.blocks:
            source = ChunkSource(block_ordinal=block.ordinal, location=block.location)
            tokens = self._encoding.encode(block.text)
            if len(tokens) > self._chunk_size:
                if current:
                    groups.append(current)
                    current = []
                    current_tokens = 0
                step = self._chunk_size - self._chunk_overlap
                for start in range(0, len(tokens), step):
                    window = tokens[start : start + self._chunk_size]
                    if not window:
                        break
                    text = self._encoding.decode(window).strip()
                    if text:
                        groups.append([(text, len(window), source)])
                    if start + self._chunk_size >= len(tokens):
                        break
                continue

            separator_tokens = 2 if current else 0
            if current and current_tokens + separator_tokens + len(tokens) > self._chunk_size:
                groups.append(current)
                current = []
                current_tokens = 0
            current.append((block.text, len(tokens), source))
            current_tokens += len(tokens) + (2 if len(current) > 1 else 0)

        if current:
            groups.append(current)

        chunks: list[TextChunk] = []
        config_identity = f"{self.version}:{self._encoding_name}:{self._chunk_size}:{self._chunk_overlap}"
        for ordinal, group in enumerate(groups):
            text = "\n\n".join(item[0] for item in group).strip()
            token_count = len(self._encoding.encode(text))
            source_items = tuple(dict.fromkeys(item[2] for item in group))
            digest = sha256(
                f"{document_id}:{content_sha256}:{document.parser_name}:{document.parser_version}:"
                f"{config_identity}:{ordinal}:{sha256(text.encode()).hexdigest()}".encode()
            ).digest()
            chunks.append(
                TextChunk(
                    id=UUID(bytes=digest[:16]),
                    ordinal=ordinal,
                    text=text,
                    token_count=token_count,
                    content_sha256=sha256(text.encode()).hexdigest(),
                    sources=source_items,
                    parser_name=document.parser_name,
                    parser_version=document.parser_version,
                    chunker_version=config_identity,
                )
            )
        if not chunks:
            raise ValueError("parsed document produced no chunks")
        return tuple(chunks)


__all__ = ["ChunkSource", "TextChunk", "TokenAwareChunker"]
