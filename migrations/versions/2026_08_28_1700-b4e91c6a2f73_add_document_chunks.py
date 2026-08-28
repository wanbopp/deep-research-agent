"""Add versioned document chunks with pgvector embeddings.

Revision ID: b4e91c6a2f73
Revises: 8c1f4a7d2e90
Create Date: 2026-08-28 17:00:00
"""

from collections.abc import Sequence

from alembic import op
from pgvector.sqlalchemy import Vector
import sqlalchemy as sa

revision: str = "b4e91c6a2f73"
down_revision: str | Sequence[str] | None = "8c1f4a7d2e90"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """创建 owner-scoped、可回查来源的 document_chunks 表."""
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.create_table(
        "document_chunks",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("user_id", sa.Uuid(), nullable=False),
        sa.Column("document_id", sa.Uuid(), nullable=False),
        sa.Column("ordinal", sa.Integer(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("content_sha256", sa.String(length=64), nullable=False),
        sa.Column("token_count", sa.Integer(), nullable=False),
        sa.Column("source_locations", sa.JSON(), nullable=False),
        sa.Column("parser_name", sa.String(length=64), nullable=False),
        sa.Column("parser_version", sa.String(length=32), nullable=False),
        sa.Column("chunker_version", sa.String(length=128), nullable=False),
        sa.Column("embedding_model", sa.String(length=255), nullable=False),
        sa.Column("embedding_version", sa.String(length=64), nullable=False),
        sa.Column("embedding", Vector(1536), nullable=False),
        sa.CheckConstraint("ordinal >= 0", name="ck_document_chunks_ordinal"),
        sa.CheckConstraint("token_count > 0", name="ck_document_chunks_token_count"),
        sa.CheckConstraint("char_length(content_sha256) = 64", name="ck_document_chunks_sha256_length"),
        sa.ForeignKeyConstraint(["document_id"], ["documents.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"]),
        sa.PrimaryKeyConstraint("id", name="pk_document_chunks"),
        sa.UniqueConstraint("document_id", "ordinal", name="uq_document_chunks_document_ordinal"),
    )
    op.create_index("ix_document_chunks_user_id", "document_chunks", ["user_id"])
    op.create_index("ix_document_chunks_document_id", "document_chunks", ["document_id"])
    op.create_index("ix_document_chunks_user_document", "document_chunks", ["user_id", "document_id"])


def downgrade() -> None:
    """删除应用拥有的 chunk 表，但保留共享 vector 扩展."""
    op.drop_index("ix_document_chunks_user_document", table_name="document_chunks")
    op.drop_index("ix_document_chunks_document_id", table_name="document_chunks")
    op.drop_index("ix_document_chunks_user_id", table_name="document_chunks")
    op.drop_table("document_chunks")
