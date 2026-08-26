"""Add recoverable ChatSession cleanup state.

Revision ID: 2c4a7e9b1d63
Revises: 9b8c4d2e1f70
Create Date: 2026-08-26 09:15:00

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op


revision: str = "2c4a7e9b1d63"
down_revision: str | Sequence[str] | None = "9b8c4d2e1f70"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add the cleanup tombstone and make research-task references detachable."""
    # server_default 只负责把升级前的历史行回填为 active。回填完成后立即移除，
    # 让运行时代码继续显式拥有默认值，也避免 Alembic autogenerate 产生 drift。
    op.add_column(
        "chat_sessions",
        sa.Column(
            "status",
            sa.String(length=32),
            nullable=False,
            server_default=sa.text("'active'"),
        ),
    )
    op.alter_column(
        "chat_sessions",
        "status",
        server_default=None,
    )
    op.create_check_constraint(
        "ck_chat_sessions_status",
        "chat_sessions",
        "status IN ('active', 'deleting')",
    )
    op.create_index(
        "ix_chat_sessions_status",
        "chat_sessions",
        ["status"],
        unique=False,
    )

    # ResearchTask.chat_session_id 本来就是可空字段。SET NULL 保留任务审计记录，
    # 同时允许 cleanup coordinator 最终物理删除已清空 checkpoint 的会话行。
    op.drop_constraint(
        "fk_research_tasks_chat_session_id_chat_sessions",
        "research_tasks",
        type_="foreignkey",
    )
    op.create_foreign_key(
        "fk_research_tasks_chat_session_id_chat_sessions",
        "research_tasks",
        "chat_sessions",
        ["chat_session_id"],
        ["id"],
        ondelete="SET NULL",
    )


def downgrade() -> None:
    """Restore the original reference behavior and remove cleanup state."""
    op.drop_constraint(
        "fk_research_tasks_chat_session_id_chat_sessions",
        "research_tasks",
        type_="foreignkey",
    )
    op.create_foreign_key(
        "fk_research_tasks_chat_session_id_chat_sessions",
        "research_tasks",
        "chat_sessions",
        ["chat_session_id"],
        ["id"],
    )

    op.drop_index("ix_chat_sessions_status", table_name="chat_sessions")
    op.drop_constraint(
        "ck_chat_sessions_status",
        "chat_sessions",
        type_="check",
    )
    op.drop_column("chat_sessions", "status")
