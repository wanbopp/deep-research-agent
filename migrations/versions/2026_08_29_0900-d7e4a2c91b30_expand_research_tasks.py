"""Expand durable research tasks and add replayable events.

Revision ID: d7e4a2c91b30
Revises: b4e91c6a2f73
Create Date: 2026-08-29 09:00:00
"""

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa

revision: str = "d7e4a2c91b30"
down_revision: str | Sequence[str] | None = "b4e91c6a2f73"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """扩展任务状态，并创建可按 ID 重放的研究事件表."""
    op.drop_constraint("ck_research_tasks_status", "research_tasks", type_="check")
    op.create_check_constraint(
        "ck_research_tasks_status",
        "research_tasks",
        "status IN ('pending', 'running', 'retrying', 'cancelling', 'completed', 'failed', 'cancelled')",
    )
    op.alter_column("research_tasks", "topic", type_=sa.String(length=1000), existing_type=sa.String(length=500))
    op.add_column("research_tasks", sa.Column("config_json", sa.JSON(), nullable=False, server_default="{}"))
    op.add_column("research_tasks", sa.Column("report_json", sa.JSON(), nullable=True))
    op.add_column("research_tasks", sa.Column("markdown_report", sa.Text(), nullable=True))
    op.add_column("research_tasks", sa.Column("idempotency_key", sa.String(length=128), nullable=True))
    op.add_column("research_tasks", sa.Column("checkpoint_thread_id", sa.String(length=200), nullable=True))
    op.add_column("research_tasks", sa.Column("attempt_count", sa.Integer(), nullable=False, server_default="0"))
    op.add_column("research_tasks", sa.Column("max_attempts", sa.Integer(), nullable=False, server_default="2"))
    op.add_column("research_tasks", sa.Column("worker_id", sa.String(length=128), nullable=True))
    op.add_column("research_tasks", sa.Column("error_code", sa.String(length=128), nullable=True))
    op.add_column("research_tasks", sa.Column("heartbeat_at", sa.DateTime(timezone=True), nullable=True))
    op.add_column("research_tasks", sa.Column("started_at", sa.DateTime(timezone=True), nullable=True))
    op.add_column("research_tasks", sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True))
    op.add_column("research_tasks", sa.Column("cancellation_requested_at", sa.DateTime(timezone=True), nullable=True))
    op.execute("UPDATE research_tasks SET idempotency_key = 'legacy-' || id::text WHERE idempotency_key IS NULL")
    op.execute("UPDATE research_tasks SET checkpoint_thread_id = 'research:' || id::text WHERE checkpoint_thread_id IS NULL")
    op.alter_column("research_tasks", "idempotency_key", nullable=False)
    op.alter_column("research_tasks", "checkpoint_thread_id", nullable=False)
    op.alter_column("research_tasks", "config_json", server_default=None)
    op.alter_column("research_tasks", "attempt_count", server_default=None)
    op.alter_column("research_tasks", "max_attempts", server_default=None)
    op.create_unique_constraint("uq_research_tasks_user_idempotency", "research_tasks", ["user_id", "idempotency_key"])
    op.create_unique_constraint("uq_research_tasks_checkpoint_thread_id", "research_tasks", ["checkpoint_thread_id"])
    op.create_check_constraint("ck_research_tasks_attempt_count", "research_tasks", "attempt_count >= 0")
    op.create_check_constraint("ck_research_tasks_max_attempts", "research_tasks", "max_attempts > 0")
    op.create_index("ix_research_tasks_worker_id", "research_tasks", ["worker_id"])
    op.create_table(
        "research_events",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("research_task_id", sa.Uuid(), nullable=False),
        sa.Column("user_id", sa.Uuid(), nullable=False),
        sa.Column("event_type", sa.String(length=64), nullable=False),
        sa.Column("payload_json", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["research_task_id"], ["research_tasks.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"]),
        sa.PrimaryKeyConstraint("id", name="pk_research_events"),
    )
    op.create_index("ix_research_events_research_task_id", "research_events", ["research_task_id"])
    op.create_index("ix_research_events_user_id", "research_events", ["user_id"])
    op.create_index("ix_research_events_task_id_id", "research_events", ["research_task_id", "id"])


def downgrade() -> None:
    """移除 Phase 7 字段并恢复最小任务状态集合."""
    op.drop_index("ix_research_events_task_id_id", table_name="research_events")
    op.drop_index("ix_research_events_user_id", table_name="research_events")
    op.drop_index("ix_research_events_research_task_id", table_name="research_events")
    op.drop_table("research_events")
    op.drop_index("ix_research_tasks_worker_id", table_name="research_tasks")
    op.drop_constraint("ck_research_tasks_max_attempts", "research_tasks", type_="check")
    op.drop_constraint("ck_research_tasks_attempt_count", "research_tasks", type_="check")
    op.drop_constraint("uq_research_tasks_checkpoint_thread_id", "research_tasks", type_="unique")
    op.drop_constraint("uq_research_tasks_user_idempotency", "research_tasks", type_="unique")
    for column in (
        "cancellation_requested_at", "completed_at", "started_at", "heartbeat_at", "error_code", "worker_id",
        "max_attempts", "attempt_count", "checkpoint_thread_id", "idempotency_key", "markdown_report",
        "report_json", "config_json",
    ):
        op.drop_column("research_tasks", column)
    op.alter_column("research_tasks", "topic", type_=sa.String(length=500), existing_type=sa.String(length=1000))
    op.drop_constraint("ck_research_tasks_status", "research_tasks", type_="check")
    op.create_check_constraint(
        "ck_research_tasks_status",
        "research_tasks",
        "status IN ('pending', 'running', 'completed', 'failed', 'cancelled')",
    )
