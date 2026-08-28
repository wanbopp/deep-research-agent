"""可追溯的文档 chunk 与 pgvector 向量模型."""

from typing import Any
from uuid import UUID

from pgvector.sqlalchemy import Vector
from sqlalchemy import CheckConstraint, Column, Index, JSON, String, Text, UniqueConstraint
from sqlmodel import Field

from app.models.base import UUIDTimestampModel

DOCUMENT_EMBEDDING_DIMENSIONS = 1536


class DocumentChunk(UUIDTimestampModel, table=True):
    """一段可检索文本、来源位置及其版本化向量."""

    __tablename__: Any = "document_chunks"
    __table_args__ = (
        CheckConstraint("ordinal >= 0", name="ck_document_chunks_ordinal"),
        CheckConstraint("token_count > 0", name="ck_document_chunks_token_count"),
        CheckConstraint("char_length(content_sha256) = 64", name="ck_document_chunks_sha256_length"),
        UniqueConstraint("document_id", "ordinal", name="uq_document_chunks_document_ordinal"),
        # 安全查询先按 owner 缩小集合，再做 pgvector 排序；document_id 支持知识库范围过滤。
        Index("ix_document_chunks_user_document", "user_id", "document_id"),
    )

    user_id: UUID = Field(foreign_key="users.id", nullable=False, index=True)
    document_id: UUID = Field(
        foreign_key="documents.id",
        nullable=False,
        index=True,
        ondelete="CASCADE",
    )
    ordinal: int = Field(nullable=False)
    content: str = Field(sa_column=Column(Text, nullable=False))
    content_sha256: str = Field(sa_column=Column(String(64), nullable=False))
    token_count: int = Field(nullable=False)
    # JSON 只保存严格 SourceLocation 投影，不接受 parser 任意 metadata；这样引用结构
    # 在版本升级时仍可验证，且不会把未知对象塞进公开响应。
    source_locations: list[dict[str, Any]] = Field(sa_column=Column(JSON, nullable=False))
    parser_name: str = Field(sa_column=Column(String(64), nullable=False))
    parser_version: str = Field(sa_column=Column(String(32), nullable=False))
    chunker_version: str = Field(sa_column=Column(String(128), nullable=False))
    embedding_model: str = Field(sa_column=Column(String(255), nullable=False))
    embedding_version: str = Field(sa_column=Column(String(64), nullable=False))
    embedding: list[float] = Field(sa_column=Column(Vector(DOCUMENT_EMBEDDING_DIMENSIONS), nullable=False))


__all__ = ["DOCUMENT_EMBEDDING_DIMENSIONS", "DocumentChunk"]
