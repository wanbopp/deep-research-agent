"""用户文档业务表模型."""

from datetime import datetime
from enum import StrEnum
from typing import Any
from uuid import UUID

from sqlalchemy import BigInteger, CheckConstraint, Column, String, UniqueConstraint
from sqlmodel import Field

from app.models.base import UTCDateTime, UUIDTimestampModel

MAX_DOCUMENT_FILENAME_LENGTH = 255
MAX_DOCUMENT_CONTENT_TYPE_LENGTH = 255
MAX_DOCUMENT_STORAGE_KEY_LENGTH = 512
MAX_DOCUMENT_FAILURE_CODE_LENGTH = 64
SHA256_HEX_LENGTH = 64


class DocumentStatus(StrEnum):
    """文档从上传到可检索或删除的业务生命周期."""

    PENDING = "pending"
    INDEXING = "indexing"
    READY = "ready"
    FAILED = "failed"
    DELETING = "deleting"


class Document(UUIDTimestampModel, table=True):
    """用户拥有的一份原始文档及其索引状态.

    ``Document`` 保存业务身份和可恢复状态，不保存文件二进制。原始字节由
    ``FileStorage`` 管理，解析后的 chunk 和 embedding 分别在 Lab 18/19 建立。
    """

    __tablename__: Any = "documents"

    __table_args__ = (
        CheckConstraint(
            "status IN ('pending', 'indexing', 'ready', 'failed', 'deleting')",
            name="ck_documents_status",
        ),
        CheckConstraint("size_bytes > 0", name="ck_documents_size_positive"),
        CheckConstraint(
            "char_length(content_sha256) = 64",
            name="ck_documents_sha256_length",
        ),
        CheckConstraint(
            "(status = 'failed' AND failure_code IS NOT NULL) OR (status <> 'failed' AND failure_code IS NULL)",
            name="ck_documents_failure_code_state",
        ),
        # 内容去重必须包含 owner。相同文件可以合法地分别属于两个用户；全局
        # hash 唯一约束既会产生跨用户冲突，也可能泄漏“别人已经上传过”。
        UniqueConstraint(
            "user_id",
            "content_sha256",
            name="uq_documents_user_content_sha256",
        ),
    )

    user_id: UUID = Field(
        foreign_key="users.id",
        nullable=False,
        index=True,
    )
    original_filename: str = Field(
        sa_column=Column(String(MAX_DOCUMENT_FILENAME_LENGTH), nullable=False),
    )
    content_type: str = Field(
        sa_column=Column(String(MAX_DOCUMENT_CONTENT_TYPE_LENGTH), nullable=False),
    )
    size_bytes: int = Field(sa_type=BigInteger, nullable=False)
    content_sha256: str = Field(
        sa_column=Column(String(SHA256_HEX_LENGTH), nullable=False),
    )
    # storage_key 只能由服务端生成。它是 FileStorage 内部定位符，不是用户提供的
    # 文件名，也不能直接作为公开下载路径返回给客户端。
    storage_key: str = Field(
        sa_column=Column(String(MAX_DOCUMENT_STORAGE_KEY_LENGTH), nullable=False, unique=True),
    )
    status: DocumentStatus = Field(
        default=DocumentStatus.PENDING,
        sa_column=Column(String(32), nullable=False, index=True),
    )
    failure_code: str | None = Field(
        default=None,
        sa_column=Column(String(MAX_DOCUMENT_FAILURE_CODE_LENGTH), nullable=True),
    )
    indexed_at: datetime | None = Field(
        default=None,
        sa_type=UTCDateTime,
        nullable=True,
    )


__all__ = [
    "Document",
    "DocumentStatus",
    "MAX_DOCUMENT_CONTENT_TYPE_LENGTH",
    "MAX_DOCUMENT_FAILURE_CODE_LENGTH",
    "MAX_DOCUMENT_FILENAME_LENGTH",
    "MAX_DOCUMENT_STORAGE_KEY_LENGTH",
    "SHA256_HEX_LENGTH",
]
