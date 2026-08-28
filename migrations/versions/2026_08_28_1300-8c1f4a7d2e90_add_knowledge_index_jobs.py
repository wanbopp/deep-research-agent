"""Add recoverable document metadata and index jobs.

Revision ID: 8c1f4a7d2e90
Revises: 7a3d1f8c5b20
Create Date: 2026-08-28 13:00:00

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op


revision: str = "8c1f4a7d2e90"
down_revision: str | Sequence[str] | None = "7a3d1f8c5b20"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """扩展 documents，并创建可跨 worker 恢复的 index_jobs."""
    # 先删除旧状态约束，再增加 indexing/deleting。迁移不能修改历史 revision，
    # 因为已经部署过的数据库只会执行本次增量脚本。
    op.drop_constraint("ck_documents_status", "documents", type_="check")

    # 新列先允许为空，以便兼容数据库中可能已经存在的旧 Document。回填使用
    # document UUID 生成稳定且 owner 内唯一的 legacy 值；随后再收紧 NOT NULL。
    op.add_column("documents", sa.Column("size_bytes", sa.BigInteger(), nullable=True))
    op.add_column("documents", sa.Column("content_sha256", sa.String(length=64), nullable=True))
    op.add_column("documents", sa.Column("storage_key", sa.String(length=512), nullable=True))
    op.add_column("documents", sa.Column("failure_code", sa.String(length=64), nullable=True))
    op.add_column(
        "documents",
        sa.Column("indexed_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.execute(
        """
        UPDATE documents
        SET size_bytes = 1,
            content_sha256 = md5(id::text) || md5(id::text),
            storage_key = 'legacy/' || id::text,
            content_type = COALESCE(content_type, 'application/octet-stream'),
            failure_code = CASE
                WHEN status = 'failed' THEN 'LEGACY_FAILURE'
                ELSE NULL
            END
        """
    )
    op.alter_column("documents", "size_bytes", nullable=False)
    op.alter_column("documents", "content_sha256", nullable=False)
    op.alter_column("documents", "storage_key", nullable=False)
    op.alter_column("documents", "content_type", existing_type=sa.String(length=255), nullable=False)

    op.create_check_constraint(
        "ck_documents_status",
        "documents",
        "status IN ('pending', 'indexing', 'ready', 'failed', 'deleting')",
    )
    op.create_check_constraint(
        "ck_documents_size_positive",
        "documents",
        "size_bytes > 0",
    )
    op.create_check_constraint(
        "ck_documents_sha256_length",
        "documents",
        "char_length(content_sha256) = 64",
    )
    op.create_check_constraint(
        "ck_documents_failure_code_state",
        "documents",
        "(status = 'failed' AND failure_code IS NOT NULL) OR "
        "(status <> 'failed' AND failure_code IS NULL)",
    )
    op.create_unique_constraint(
        "uq_documents_user_content_sha256",
        "documents",
        ["user_id", "content_sha256"],
    )
    op.create_unique_constraint(
        "uq_documents_storage_key",
        "documents",
        ["storage_key"],
    )

    op.create_table(
        "index_jobs",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("document_id", sa.Uuid(), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("attempt_count", sa.Integer(), nullable=False),
        sa.Column("max_attempts", sa.Integer(), nullable=False),
        sa.Column("claim_token", sa.Uuid(), nullable=True),
        sa.Column("claimed_by", sa.String(length=128), nullable=True),
        sa.Column("lease_expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("error_code", sa.String(length=64), nullable=True),
        sa.CheckConstraint(
            "status IN ('pending', 'running', 'completed', 'failed')",
            name="ck_index_jobs_status",
        ),
        sa.CheckConstraint(
            "attempt_count >= 0 AND max_attempts > 0 AND attempt_count <= max_attempts",
            name="ck_index_jobs_attempts",
        ),
        sa.CheckConstraint(
            "(status = 'running' AND claim_token IS NOT NULL "
            "AND claimed_by IS NOT NULL AND lease_expires_at IS NOT NULL "
            "AND started_at IS NOT NULL AND finished_at IS NULL) OR "
            "(status <> 'running' AND claim_token IS NULL "
            "AND claimed_by IS NULL AND lease_expires_at IS NULL)",
            name="ck_index_jobs_claim_state",
        ),
        sa.CheckConstraint(
            "(status IN ('completed', 'failed') AND finished_at IS NOT NULL) OR "
            "(status IN ('pending', 'running') AND finished_at IS NULL)",
            name="ck_index_jobs_finished_state",
        ),
        sa.CheckConstraint(
            "(status = 'failed' AND error_code IS NOT NULL) OR "
            "(status <> 'failed' AND error_code IS NULL)",
            name="ck_index_jobs_error_state",
        ),
        sa.ForeignKeyConstraint(
            ["document_id"],
            ["documents.id"],
            name="fk_index_jobs_document_id_documents",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_index_jobs"),
        sa.UniqueConstraint("document_id", name="uq_index_jobs_document_id"),
    )
    op.create_index("ix_index_jobs_document_id", "index_jobs", ["document_id"], unique=False)
    op.create_index("ix_index_jobs_status", "index_jobs", ["status"], unique=False)
    op.create_index(
        "ix_index_jobs_lease_expires_at",
        "index_jobs",
        ["lease_expires_at"],
        unique=False,
    )


def downgrade() -> None:
    """删除索引任务并把 Document 恢复到旧的最小模型."""
    op.drop_index("ix_index_jobs_lease_expires_at", table_name="index_jobs")
    op.drop_index("ix_index_jobs_status", table_name="index_jobs")
    op.drop_index("ix_index_jobs_document_id", table_name="index_jobs")
    op.drop_table("index_jobs")

    op.drop_constraint("uq_documents_storage_key", "documents", type_="unique")
    op.drop_constraint(
        "uq_documents_user_content_sha256",
        "documents",
        type_="unique",
    )
    op.drop_constraint("ck_documents_failure_code_state", "documents", type_="check")
    op.drop_constraint("ck_documents_sha256_length", "documents", type_="check")
    op.drop_constraint("ck_documents_size_positive", "documents", type_="check")
    op.drop_constraint("ck_documents_status", "documents", type_="check")

    # 旧 schema 不认识 indexing/deleting；回退前先把它们归一到旧状态集合。
    op.execute(
        "UPDATE documents SET status = 'pending' "
        "WHERE status IN ('indexing', 'deleting')"
    )
    op.drop_column("documents", "indexed_at")
    op.drop_column("documents", "failure_code")
    op.drop_column("documents", "storage_key")
    op.drop_column("documents", "content_sha256")
    op.drop_column("documents", "size_bytes")
    op.alter_column("documents", "content_type", existing_type=sa.String(length=255), nullable=True)
    op.create_check_constraint(
        "ck_documents_status",
        "documents",
        "status IN ('pending', 'ready', 'failed')",
    )
