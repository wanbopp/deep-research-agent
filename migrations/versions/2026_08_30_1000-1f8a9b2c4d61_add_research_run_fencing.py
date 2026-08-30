"""Add research run fencing token and lifecycle version.

Revision ID: 1f8a9b2c4d61
Revises: d7e4a2c91b30
Create Date: 2026-08-30 10:00:00
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "1f8a9b2c4d61"
down_revision: str | Sequence[str] | None = "d7e4a2c91b30"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """增加每次领取唯一 run_id，并约束运行态必须持有完整 claim."""
    op.add_column("research_tasks", sa.Column("active_run_id", sa.Uuid(), nullable=True))
    op.add_column(
        "research_tasks",
        sa.Column("lifecycle_version", sa.Integer(), nullable=False, server_default="0"),
    )
    # 升级时旧 running 任务可能来自已经退出的 Worker，无法安全补造 run_id。
    # 统一转回 retrying，由新 Worker 领取并产生真实 token；cancelling 则兑现用户
    # 已持久化的取消意图。状态与旧 claim 清理在约束创建前完成。
    op.execute(
        """
        UPDATE research_tasks
        SET status = CASE WHEN status = 'cancelling' THEN 'cancelled' ELSE 'retrying' END,
            worker_id = NULL,
            heartbeat_at = NULL,
            completed_at = CASE WHEN status = 'cancelling' THEN NOW() ELSE completed_at END,
            error_code = 'MIGRATED_STALE_CLAIM',
            lifecycle_version = lifecycle_version + 1,
            updated_at = NOW()
        WHERE status IN ('running', 'cancelling')
        """
    )
    op.alter_column("research_tasks", "lifecycle_version", server_default=None)
    op.create_index("ix_research_tasks_active_run_id", "research_tasks", ["active_run_id"], unique=True)
    op.create_check_constraint(
        "ck_research_tasks_lifecycle_version",
        "research_tasks",
        "lifecycle_version >= 0",
    )
    op.create_check_constraint(
        "ck_research_tasks_active_claim",
        "research_tasks",
        "((status IN ('running', 'cancelling') AND active_run_id IS NOT NULL "
        "AND worker_id IS NOT NULL AND heartbeat_at IS NOT NULL) OR "
        "(status NOT IN ('running', 'cancelling') AND active_run_id IS NULL "
        "AND worker_id IS NULL AND heartbeat_at IS NULL))",
    )


def downgrade() -> None:
    """移除 fencing 字段；不会尝试恢复升级前失效的 Worker claim."""
    op.drop_constraint("ck_research_tasks_active_claim", "research_tasks", type_="check")
    op.drop_constraint("ck_research_tasks_lifecycle_version", "research_tasks", type_="check")
    op.drop_index("ix_research_tasks_active_run_id", table_name="research_tasks")
    op.drop_column("research_tasks", "lifecycle_version")
    op.drop_column("research_tasks", "active_run_id")
