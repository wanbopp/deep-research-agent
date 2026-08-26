"""Add owner-scoped long-term memories with pgvector embeddings.

Revision ID: 5f7c2a9d4e81
Revises: 2c4a7e9b1d63
Create Date: 2026-08-26 15:00:00

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector


revision: str = "5f7c2a9d4e81"
down_revision: str | Sequence[str] | None = "2c4a7e9b1d63"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

MEMORY_EMBEDDING_DIMENSIONS = 1536


def upgrade() -> None:
    """创建 pgvector 扩展和应用拥有的长期记忆表."""
    # vector 是数据库级共享能力，不是 memories 表的私有对象。IF NOT EXISTS 让
    # 已由运维或其他组件启用扩展的数据库也能安全升级。
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        "memories",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("user_id", sa.Uuid(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("kind", sa.String(length=32), nullable=False),
        # VECTOR(1536) 在数据库层拒绝错误维度，防止 provider 配置漂移后写入
        # 无法比较的向量。改变维度必须通过新的 migration 和向量重建完成。
        sa.Column(
            "embedding",
            Vector(MEMORY_EMBEDDING_DIMENSIONS),
            nullable=False,
        ),
        sa.Column("source_thread_id", sa.Uuid(), nullable=False),
        sa.CheckConstraint(
            "kind IN ('preference', 'fact', 'constraint')",
            name="ck_memories_kind",
        ),
        sa.ForeignKeyConstraint(
            ["source_thread_id"],
            ["chat_sessions.id"],
            name="fk_memories_source_thread_id_chat_sessions",
        ),
        sa.ForeignKeyConstraint(
            ["user_id"],
            ["users.id"],
            name="fk_memories_user_id_users",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_memories"),
    )
    op.create_index(
        "ix_memories_source_thread_id",
        "memories",
        ["source_thread_id"],
        unique=False,
    )
    op.create_index(
        "ix_memories_user_id",
        "memories",
        ["user_id"],
        unique=False,
    )
    op.create_index(
        "ix_memories_user_id_kind",
        "memories",
        ["user_id", "kind"],
        unique=False,
    )


def downgrade() -> None:
    """只删除应用拥有的 memories 表，保留共享 vector 扩展."""
    op.drop_index("ix_memories_user_id_kind", table_name="memories")
    op.drop_index("ix_memories_user_id", table_name="memories")
    op.drop_index("ix_memories_source_thread_id", table_name="memories")
    op.drop_table("memories")

    # 故意不执行 DROP EXTENSION vector。扩展属于数据库共享能力，其他服务或
    # 外部表可能仍在使用；migration 只能回滚自己明确拥有的对象。
