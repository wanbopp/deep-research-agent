"""深度研究任务业务表模型."""

from enum import StrEnum
from typing import Any
from uuid import UUID

from sqlalchemy import CheckConstraint, Column, String
from sqlmodel import Field

from app.models.base import UUIDTimestampModel


class ResearchTaskStatus(StrEnum):
    """研究任务在业务层可观察到的最小生命周期."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ResearchTask(UUIDTimestampModel, table=True):
    """用户发起的一次深度研究任务.

    ResearchTask 是产品层任务记录，不等于 LangGraph state。图内部的当前节点、
    messages、interrupt 和 checkpoint 仍由 LangGraph checkpointer 管理。
    """

    __tablename__: Any = "research_tasks"

    __table_args__ = (
        CheckConstraint(
            "status IN ('pending', 'running', 'completed', 'failed', 'cancelled')",
            name="ck_research_tasks_status",
        ),
    )

    # 任务始终属于用户，这是后续认证与数据隔离的主要边界。
    user_id: UUID = Field(
        foreign_key="users.id",
        nullable=False,
        index=True,
    )

    # 任务既可以从聊天会话发起，也可以由独立 API、定时任务或后台工作流创建，
    # 所以 chat_session_id 是可空外键。
    chat_session_id: UUID | None = Field(
        default=None,
        foreign_key="chat_sessions.id",
        # 会话删除不应抹掉已经形成的研究任务审计记录。数据库在删除父会话时
        # 把这个可空引用置空，任务仍归原 user_id 所有。
        ondelete="SET NULL",
        nullable=True,
        index=True,
    )

    topic: str = Field(
        sa_column=Column(String(500), nullable=False),
    )

    status: ResearchTaskStatus = Field(
        default=ResearchTaskStatus.PENDING,
        sa_column=Column(String(32), nullable=False, index=True),
    )
