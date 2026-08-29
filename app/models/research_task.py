"""可恢复深度研究任务与持久进度事件表模型."""

from datetime import datetime
from enum import StrEnum
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import JSON, CheckConstraint, Column, Index, Integer, String, Text, UniqueConstraint
from sqlmodel import Field, SQLModel

from app.models.base import UTCDateTime, UUIDTimestampModel, utc_now


class ResearchTaskStatus(StrEnum):
    """研究任务跨 HTTP 请求和 worker 重启可观察的生命周期."""

    PENDING = "pending"
    RUNNING = "running"
    RETRYING = "retrying"
    CANCELLING = "cancelling"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ResearchTask(UUIDTimestampModel, table=True):
    """用户发起的一次持久研究任务.

    该表保存产品层状态和最终报告，不复制 LangGraph 的节点内部状态。图进度由
    checkpointer 保存；两者通过 checkpoint_thread_id 关联，各自承担清晰职责。
    """

    __tablename__: Any = "research_tasks"
    __table_args__ = (
        CheckConstraint(
            "status IN ('pending', 'running', 'retrying', 'cancelling', 'completed', 'failed', 'cancelled')",
            name="ck_research_tasks_status",
        ),
        CheckConstraint("attempt_count >= 0", name="ck_research_tasks_attempt_count"),
        CheckConstraint("max_attempts > 0", name="ck_research_tasks_max_attempts"),
        UniqueConstraint("user_id", "idempotency_key", name="uq_research_tasks_user_idempotency"),
    )

    user_id: UUID = Field(foreign_key="users.id", nullable=False, index=True)
    chat_session_id: UUID | None = Field(
        default=None,
        foreign_key="chat_sessions.id",
        ondelete="SET NULL",
        nullable=True,
        index=True,
    )
    topic: str = Field(sa_column=Column(String(1000), nullable=False))
    status: ResearchTaskStatus = Field(
        default=ResearchTaskStatus.PENDING,
        sa_column=Column(String(32), nullable=False, index=True),
    )
    config_json: dict[str, object] = Field(default_factory=dict, sa_column=Column(JSON, nullable=False))
    report_json: dict[str, object] | None = Field(default=None, sa_column=Column(JSON, nullable=True))
    markdown_report: str | None = Field(default=None, sa_column=Column(Text, nullable=True))
    idempotency_key: str = Field(
        default_factory=lambda: f"legacy-{uuid4().hex}",
        sa_column=Column(String(128), nullable=False),
    )
    checkpoint_thread_id: str = Field(
        default_factory=lambda: f"research:{uuid4().hex}",
        sa_column=Column(String(200), nullable=False, unique=True),
    )
    attempt_count: int = Field(default=0, sa_column=Column(Integer, nullable=False))
    max_attempts: int = Field(default=2, sa_column=Column(Integer, nullable=False))
    worker_id: str | None = Field(default=None, sa_column=Column(String(128), nullable=True, index=True))
    error_code: str | None = Field(default=None, sa_column=Column(String(128), nullable=True))
    heartbeat_at: datetime | None = Field(default=None, sa_type=UTCDateTime, nullable=True)
    started_at: datetime | None = Field(default=None, sa_type=UTCDateTime, nullable=True)
    completed_at: datetime | None = Field(default=None, sa_type=UTCDateTime, nullable=True)
    cancellation_requested_at: datetime | None = Field(default=None, sa_type=UTCDateTime, nullable=True)


class ResearchEvent(SQLModel, table=True):
    """SSE 可重放的单调进度事件."""

    __tablename__: Any = "research_events"
    __table_args__ = (Index("ix_research_events_task_id_id", "research_task_id", "id"),)

    id: int | None = Field(default=None, primary_key=True)
    research_task_id: UUID = Field(foreign_key="research_tasks.id", nullable=False, index=True, ondelete="CASCADE")
    user_id: UUID = Field(foreign_key="users.id", nullable=False, index=True)
    event_type: str = Field(sa_column=Column(String(64), nullable=False))
    payload_json: dict[str, object] = Field(default_factory=dict, sa_column=Column(JSON, nullable=False))
    created_at: datetime = Field(default_factory=utc_now, sa_type=UTCDateTime, nullable=False)


__all__ = ["ResearchEvent", "ResearchTask", "ResearchTaskStatus"]
