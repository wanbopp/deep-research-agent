"""Add recoverable automatic-title claim fields.

Revision ID: 7a3d1f8c5b20
Revises: 5f7c2a9d4e81
Create Date: 2026-08-27 09:00:00

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op


revision: str = "7a3d1f8c5b20"
down_revision: str | Sequence[str] | None = "5f7c2a9d4e81"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """增加自动命名租约和完成时间，不改变已有会话标题."""
    # 三列全部可空，因此历史会话升级后自然处于“尚未申请、尚未自动生成”状态。
    # 不使用 server_default，避免数据库与运行时代码形成两套默认值。
    op.add_column(
        "chat_sessions",
        sa.Column("title_claim_token", sa.Uuid(), nullable=True),
    )
    op.add_column(
        "chat_sessions",
        sa.Column("title_claimed_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column(
        "chat_sessions",
        sa.Column("title_generated_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_check_constraint(
        "ck_chat_sessions_title_claim_pair",
        "chat_sessions",
        "(title_claim_token IS NULL AND title_claimed_at IS NULL) OR "
        "(title_claim_token IS NOT NULL AND title_claimed_at IS NOT NULL)",
    )


def downgrade() -> None:
    """删除自动命名内部状态，保留已经生成的普通 title 字符串."""
    op.drop_constraint(
        "ck_chat_sessions_title_claim_pair",
        "chat_sessions",
        type_="check",
    )
    op.drop_column("chat_sessions", "title_generated_at")
    op.drop_column("chat_sessions", "title_claimed_at")
    op.drop_column("chat_sessions", "title_claim_token")
