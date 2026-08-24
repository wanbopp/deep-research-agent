"""Run the full Alembic lifecycle against an isolated temporary database.

This smoke intentionally performs DDL, but only inside a randomly named database that
the script creates and removes. It never upgrades or downgrades the configured application
database. The temporary ``checkpoints`` table proves that Alembic ignores external tables.
"""

import json
import os
from pathlib import Path
from time import perf_counter
from uuid import uuid4

import psycopg
from alembic import command
from alembic.config import Config
from psycopg import sql
from psycopg.conninfo import make_conninfo

from app.core.config import settings
from app.infrastructure.database import build_orm_database_url

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANAGED_TABLES = frozenset(
    {
        "users",
        "chat_sessions",
        "documents",
        "research_tasks",
    }
)
EXTERNAL_TABLE = "checkpoints"
CONNECTION_TIMEOUT_SECONDS = 10


def _elapsed_ms(started_at: float) -> float:
    """Return elapsed milliseconds without exposing connection settings."""
    return round((perf_counter() - started_at) * 1000, 2)


def _conninfo(database: str) -> str:
    """Build a psycopg connection string without manual password interpolation."""
    return make_conninfo(
        host=settings.POSTGRES_HOST,
        port=settings.POSTGRES_PORT,
        dbname=database,
        user=settings.POSTGRES_USER,
        password=settings.POSTGRES_PASSWORD,
        connect_timeout=CONNECTION_TIMEOUT_SECONDS,
    )


def _temporary_database_url(database: str) -> str:
    """Return the temporary URL only for the current process environment."""
    return build_orm_database_url(settings).set(database=database).render_as_string(hide_password=False)


def _public_tables(database: str) -> frozenset[str]:
    """Read public table names from the isolated database."""
    with psycopg.connect(_conninfo(database)) as connection:
        rows = connection.execute(
            """
            SELECT tablename
            FROM pg_catalog.pg_tables
            WHERE schemaname = 'public'
            ORDER BY tablename
            """
        ).fetchall()
    return frozenset(str(row[0]) for row in rows)


def _create_database(admin_database: str, test_database: str) -> None:
    """Create the randomly named isolated database."""
    with psycopg.connect(_conninfo(admin_database), autocommit=True) as connection:
        connection.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(test_database)))


def _create_external_table(test_database: str) -> None:
    """Create one representative table Alembic must never manage."""
    with psycopg.connect(_conninfo(test_database)) as connection:
        connection.execute(sql.SQL("CREATE TABLE {} (id integer PRIMARY KEY)").format(sql.Identifier(EXTERNAL_TABLE)))
        connection.commit()


def _drop_database(admin_database: str, test_database: str) -> None:
    """Terminate test connections and remove only the randomly named database."""
    with psycopg.connect(_conninfo(admin_database), autocommit=True) as connection:
        connection.execute(
            """
            SELECT pg_terminate_backend(pid)
            FROM pg_stat_activity
            WHERE datname = %s AND pid <> pg_backend_pid()
            """,
            (test_database,),
        )
        connection.execute(sql.SQL("DROP DATABASE IF EXISTS {}").format(sql.Identifier(test_database)))


def _run_smoke() -> dict[str, object]:
    """Execute upgrade, downgrade, upgrade and no-diff checks."""
    started_at = perf_counter()
    admin_database = settings.POSTGRES_DB
    test_database = f"deep_research_migration_{uuid4().hex[:12]}"
    previous_override = os.environ.get("ALEMBIC_DATABASE_URL")
    database_created = False
    cleanup_ok = False

    try:
        _create_database(admin_database, test_database)
        database_created = True
        _create_external_table(test_database)
        os.environ["ALEMBIC_DATABASE_URL"] = _temporary_database_url(test_database)

        alembic_config = Config(str(PROJECT_ROOT / "alembic.ini"))

        command.upgrade(alembic_config, "head")
        tables_after_first_upgrade = _public_tables(test_database)
        first_upgrade_matches = tables_after_first_upgrade == MANAGED_TABLES | {
            "alembic_version",
            EXTERNAL_TABLE,
        }

        command.downgrade(alembic_config, "base")
        tables_after_downgrade = _public_tables(test_database)
        downgrade_preserves_only_external = tables_after_downgrade == {
            "alembic_version",
            EXTERNAL_TABLE,
        }

        command.upgrade(alembic_config, "head")
        tables_after_second_upgrade = _public_tables(test_database)
        second_upgrade_matches = tables_after_second_upgrade == MANAGED_TABLES | {
            "alembic_version",
            EXTERNAL_TABLE,
        }

        # command.check raises when autogenerate detects a pending schema change.
        command.check(alembic_config)
        no_schema_diff = True

        ok = all(
            (
                first_upgrade_matches,
                downgrade_preserves_only_external,
                second_upgrade_matches,
                no_schema_diff,
            )
        )
        return {
            "ok": ok,
            "first_upgrade_matches": first_upgrade_matches,
            "downgrade_preserves_only_external": downgrade_preserves_only_external,
            "second_upgrade_matches": second_upgrade_matches,
            "external_table_preserved": EXTERNAL_TABLE in tables_after_second_upgrade,
            "no_schema_diff": no_schema_diff,
            "elapsed_ms": _elapsed_ms(started_at),
        }
    finally:
        if previous_override is None:
            os.environ.pop("ALEMBIC_DATABASE_URL", None)
        else:
            os.environ["ALEMBIC_DATABASE_URL"] = previous_override

        if database_created:
            try:
                _drop_database(admin_database, test_database)
            except Exception:
                cleanup_ok = False
            else:
                cleanup_ok = True

        # The success path returns above, but finally still runs. Cleanup failure must not
        # silently pass, because a leaked temporary database is an operational defect.
        if database_created and not cleanup_ok:
            raise RuntimeError("Temporary migration database cleanup failed")


def main() -> int:
    """Print one credential-safe JSON summary."""
    started_at = perf_counter()
    try:
        summary = _run_smoke()
    except Exception as exc:
        print(
            json.dumps(
                {
                    "ok": False,
                    "error_type": type(exc).__name__,
                    "elapsed_ms": _elapsed_ms(started_at),
                }
            )
        )
        return 1

    # Reaching here proves the temporary database was also cleaned up.
    summary["cleanup_ok"] = True
    print(json.dumps(summary))
    return 0 if summary["ok"] is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
