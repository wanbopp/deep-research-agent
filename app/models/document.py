"""用户文档业务表模型."""

from enum import StrEnum
from typing import Any
from uuid import UUID

from sqlalchemy import CheckConstraint, Column, String
from sqlmodel import Field

from app.models.base import UUIDTimestampModel


class DocumentStatus(StrEnum):
    """文档进入后续解析流程前需要保留的最小状态."""

    PENDING = "pending"
    READY = "ready"
    FAILED = "failed"


class Document(UUIDTimestampModel, table=True):
    """用户拥有的一份文档元数据.

    当前只描述文档身份、所有者和处理状态。文件二进制、对象存储地址、
    解析后的分块以及向量索引都属于后续文档管道，不存放在这个最小模型中。
    """

    __tablename__: Any = "documents"

    # Python 枚举保护应用代码，CheckConstraint 保护所有数据库写入入口。
    __table_args__ = (
        CheckConstraint(
            "status IN ('pending', 'ready', 'failed')",
            name="ck_documents_status",
        ),
    )

    # 文档必须属于一个用户。Repository 后续必须同时使用 document_id 和
    # user_id 查询，不能只凭文档 ID 绕过资源所有权检查。
    user_id: UUID = Field(
        foreign_key="users.id",
        nullable=False,
        index=True,
    )

    original_filename: str = Field(
        sa_column=Column(String(255), nullable=False),
    )

    # MIME 类型可能在上传时缺失，因此 Python 和数据库两层都允许为空。
    content_type: str | None = Field(
        default=None,
        sa_column=Column(String(255), nullable=True),
    )

    status: DocumentStatus = Field(
        default=DocumentStatus.PENDING,
        sa_column=Column(String(32), nullable=False, index=True),
    )
