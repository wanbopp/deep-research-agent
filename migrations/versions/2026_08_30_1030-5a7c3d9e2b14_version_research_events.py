"""Version and normalize durable research events.

Revision ID: 5a7c3d9e2b14
Revises: 1f8a9b2c4d61
Create Date: 2026-08-30 10:30:00
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "5a7c3d9e2b14"
down_revision: str | Sequence[str] | None = "1f8a9b2c4d61"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_V1_EVENTS = (
    "task_created",
    "task_started",
    "node_completed",
    "cancellation_requested",
    "task_retrying",
    "run_lease_expired",
    "task_completed",
    "task_failed",
    "task_cancelled",
)


def upgrade() -> None:
    """增加 envelope 版本和 run_id，并把开放节点事件归一为一个事件名."""
    op.add_column(
        "research_events",
        sa.Column("schema_version", sa.Integer(), nullable=False, server_default="0"),
    )
    op.add_column("research_events", sa.Column("run_id", sa.Uuid(), nullable=True))
    op.execute(
        """
        UPDATE research_events
        SET event_type = 'node_completed'
        WHERE event_type LIKE 'node_%_completed'
        """
    )
    quoted = ", ".join(f"'{name}'" for name in _V1_EVENTS)
    op.execute(f"UPDATE research_events SET schema_version = 1 WHERE event_type IN ({quoted})")
    # R3 曾把 run_id 安全字符串放在 payload 中。只有符合 UUID 格式时才迁移，
    # 未知 legacy payload 保持原样，避免 migration 因脏历史整体失败。
    op.execute(
        """
        UPDATE research_events
        SET run_id = (payload_json ->> 'run_id')::uuid,
            payload_json = (payload_json::jsonb - 'run_id')::json
        WHERE payload_json ->> 'run_id' ~*
              '^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$'
        """
    )
    op.alter_column("research_events", "schema_version", server_default="1")
    op.create_index("ix_research_events_run_id", "research_events", ["run_id"])
    op.create_check_constraint(
        "ck_research_events_schema_version",
        "research_events",
        "schema_version IN (0, 1)",
    )


def downgrade() -> None:
    """移除 envelope 字段；归一后的节点事件名保持不逆向猜测."""
    op.drop_constraint("ck_research_events_schema_version", "research_events", type_="check")
    op.drop_index("ix_research_events_run_id", table_name="research_events")
    op.drop_column("research_events", "run_id")
    op.drop_column("research_events", "schema_version")
