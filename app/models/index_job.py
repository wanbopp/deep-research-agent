"""文档索引任务及其跨 worker 租约状态."""

from datetime import datetime
from enum import StrEnum
from typing import Any
from uuid import UUID

from sqlalchemy import CheckConstraint, Column, String, UniqueConstraint
from sqlmodel import Field

from app.models.base import UTCDateTime, UUIDTimestampModel

MAX_INDEX_JOB_ERROR_CODE_LENGTH = 64
MAX_INDEX_JOB_WORKER_ID_LENGTH = 128


class IndexJobStatus(StrEnum):
    """索引任务可持久恢复的执行状态."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class IndexJob(UUIDTimestampModel, table=True):
    """一份文档当前的可重试索引工作项.

    一个 Document 在 Lab 17 只保留一个当前 IndexJob。失败重试会把同一 job
    重新置为 pending，并通过 ``attempt_count`` 保留已执行次数。真正的历史审计
    可以在后续引入单独 attempt 表，而不是无限复制当前工作项。
    """

    __tablename__: Any = "index_jobs"
    __table_args__ = (
        CheckConstraint(
            "status IN ('pending', 'running', 'completed', 'failed')",
            name="ck_index_jobs_status",
        ),
        CheckConstraint(
            "attempt_count >= 0 AND max_attempts > 0 AND attempt_count <= max_attempts",
            name="ck_index_jobs_attempts",
        ),
        CheckConstraint(
            "(status = 'running' AND claim_token IS NOT NULL "
            "AND claimed_by IS NOT NULL AND lease_expires_at IS NOT NULL "
            "AND started_at IS NOT NULL AND finished_at IS NULL) OR "
            "(status <> 'running' AND claim_token IS NULL "
            "AND claimed_by IS NULL AND lease_expires_at IS NULL)",
            name="ck_index_jobs_claim_state",
        ),
        CheckConstraint(
            "(status IN ('completed', 'failed') AND finished_at IS NOT NULL) OR "
            "(status IN ('pending', 'running') AND finished_at IS NULL)",
            name="ck_index_jobs_finished_state",
        ),
        CheckConstraint(
            "(status = 'failed' AND error_code IS NOT NULL) OR (status <> 'failed' AND error_code IS NULL)",
            name="ck_index_jobs_error_state",
        ),
        UniqueConstraint("document_id", name="uq_index_jobs_document_id"),
    )

    document_id: UUID = Field(
        foreign_key="documents.id",
        ondelete="CASCADE",
        nullable=False,
        index=True,
    )
    status: IndexJobStatus = Field(
        default=IndexJobStatus.PENDING,
        sa_column=Column(String(32), nullable=False, index=True),
    )
    attempt_count: int = Field(default=0, nullable=False)
    max_attempts: int = Field(default=3, nullable=False)
    claim_token: UUID | None = Field(default=None, nullable=True)
    claimed_by: str | None = Field(
        default=None,
        sa_column=Column(String(MAX_INDEX_JOB_WORKER_ID_LENGTH), nullable=True),
    )
    lease_expires_at: datetime | None = Field(
        default=None,
        sa_type=UTCDateTime,
        nullable=True,
        index=True,
    )
    started_at: datetime | None = Field(
        default=None,
        sa_type=UTCDateTime,
        nullable=True,
    )
    finished_at: datetime | None = Field(
        default=None,
        sa_type=UTCDateTime,
        nullable=True,
    )
    error_code: str | None = Field(
        default=None,
        sa_column=Column(String(MAX_INDEX_JOB_ERROR_CODE_LENGTH), nullable=True),
    )


__all__ = [
    "IndexJob",
    "IndexJobStatus",
    "MAX_INDEX_JOB_ERROR_CODE_LENGTH",
    "MAX_INDEX_JOB_WORKER_ID_LENGTH",
]
