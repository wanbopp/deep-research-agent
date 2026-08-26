"""Add the required password hash credential to application users.

Revision ID: 9b8c4d2e1f70
Revises: 6d6d69a03dd8
Create Date: 2026-08-24 16:00:00

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op


# Alembic uses these identifiers to order migrations as a directed chain.
revision: str = "9b8c4d2e1f70"
down_revision: str | Sequence[str] | None = "6d6d69a03dd8"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add a non-null credential column only when no legacy users exist.

    A password hash cannot be reconstructed from an existing user row. Silently writing a
    placeholder would create unusable accounts and disguise a data migration decision. This
    project has not exposed registration yet, so the safe checkpoint policy is explicit:
    migrate only an empty pre-authentication users table; otherwise stop before changing DDL.
    """
    connection = op.get_bind()
    has_existing_users = bool(
        connection.execute(
            sa.text("SELECT EXISTS (SELECT 1 FROM users)"),
        ).scalar_one()
    )
    if has_existing_users:
        raise RuntimeError(
            "Cannot add users.password_hash while legacy user rows exist; migrate or reset those accounts first"
        )

    # With an empty table PostgreSQL can enforce NOT NULL immediately, so there is no
    # temporary nullable state and no fake server default that could leak into future rows.
    op.add_column(
        "users",
        sa.Column("password_hash", sa.String(length=255), nullable=False),
    )


def downgrade() -> None:
    """Remove only the credential column owned by this revision."""
    op.drop_column("users", "password_hash")
