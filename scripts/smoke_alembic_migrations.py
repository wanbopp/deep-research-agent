"""在随机临时数据库中运行完整 Alembic 生命周期验收.

Alembic 是 SQLAlchemy 生态中的数据库迁移工具。它以 Python 脚本（而非裸 SQL）
描述 schema 的每一次变更，并维护 ``alembic_version`` 表记录当前版本，从而支持
向前升级（upgrade）、向后回退（downgrade）以及与 SQLModel/SQLAlchemy 模型定义
的自动差异比对（autogenerate）。本项目把每张业务表的建表、加列、改约束等
DDL 都放在 ``migrations/versions/`` 下的迁移脚本中，由 Alembic 统一编排执行。

教学心智模型：
1. 配置中的 PostgreSQL 数据库只作为"管理入口"，脚本不会迁移其中的业务表；
2. 脚本先创建随机临时数据库，再把 Alembic URL 临时指向它；
3. upgrade -> downgrade -> upgrade 验证迁移既能前进、也能回退、还能再次前进；
4. 模拟的 ``checkpoints`` 表不属于本应用 migration，用它验证表所有权边界；
5. 无论成功或失败，``finally`` 都恢复环境变量并删除临时数据库。

该 smoke 会执行真实 DDL，但所有 DDL 都限制在随机临时数据库内，不会升级或降级
当前配置的应用数据库。
"""

import json
import os
from datetime import UTC, datetime
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

# Alembic Config 需要从项目根目录读取 alembic.ini。使用 __file__ 定位，保证
# 从 PowerShell、PyCharm 或其他工作目录启动脚本时都能找到同一个配置文件。
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Alembic 只拥有这六张业务表。集合精确比较可以同时发现“少创建”和“误创建”。
MANAGED_TABLES = frozenset(
    {
        "users",
        "chat_sessions",
        "documents",
        "document_chunks",
        "index_jobs",
        "memories",
        "research_tasks",
    }
)
# checkpoints 代表未来由 LangGraph checkpointer 自己管理的外部表。Alembic 的
# upgrade、downgrade 和 autogenerate 都不能修改它。
EXTERNAL_TABLE = "checkpoints"
CONNECTION_TIMEOUT_SECONDS = 10
INITIAL_BUSINESS_REVISION = "6d6d69a03dd8"


def _elapsed_ms(started_at: float) -> float:
    """返回耗时毫秒数，不把连接配置放进输出."""
    return round((perf_counter() - started_at) * 1000, 2)


def _conninfo(database: str) -> str:
    """使用结构化参数构造 psycopg 连接串.

    不手工拼接 URL，可以正确转义密码中的特殊字符。返回值只交给驱动使用，
    绝不能打印，否则真实密码会进入控制台或日志。
    """
    return make_conninfo(
        host=settings.POSTGRES_HOST,
        port=settings.POSTGRES_PORT,
        dbname=database,
        user=settings.POSTGRES_USER,
        password=settings.POSTGRES_PASSWORD,
        connect_timeout=CONNECTION_TIMEOUT_SECONDS,
    )


def _temporary_database_url(database: str) -> str:
    """构造仅写入当前进程环境变量的临时 Alembic URL.

    Alembic 需要含真实密码的 URL 才能连接，但该字符串只在内存和当前进程的
    ``ALEMBIC_DATABASE_URL`` 中存在，最终 JSON 摘要不会输出它。
    """
    return build_orm_database_url(settings).set(database=database).render_as_string(hide_password=False)


def _public_tables(database: str) -> frozenset[str]:
    """从 PostgreSQL 系统目录读取临时数据库的 public 表名.

    这里查询数据库实际状态，而不是读取 SQLModel metadata。否则测试只是在
    用声明检查声明，无法证明 migration 真的创建或删除了对应表。
    """
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


def _password_hash_column_matches(database: str) -> bool:
    """直接读取 PostgreSQL catalog，验证 credential 列的真实形状.

    ``alembic check`` 负责发现整体 schema drift；这里再显式检查教学重点，确保
    ``users.password_hash`` 确实是不可空 varchar(255)，而不只是 Python model 中
    看起来正确。
    """
    with psycopg.connect(_conninfo(database)) as connection:
        row = connection.execute(
            """
            SELECT data_type, character_maximum_length, is_nullable
            FROM information_schema.columns
            WHERE table_schema = 'public'
              AND table_name = 'users'
              AND column_name = 'password_hash'
            """
        ).fetchone()

    return row == ("character varying", 255, "NO")


def _password_hash_column_exists(database: str) -> bool:
    """判断失败的 migration 是否意外留下了 credential 列."""
    with psycopg.connect(_conninfo(database)) as connection:
        exists = connection.execute(
            """
            SELECT EXISTS (
                SELECT 1
                FROM information_schema.columns
                WHERE table_schema = 'public'
                  AND table_name = 'users'
                  AND column_name = 'password_hash'
            )
            """
        ).fetchone()
    return exists == (True,)


def _insert_legacy_user(database: str) -> None:
    """在只有首个 revision 的 users 表中模拟一条旧账户记录."""
    now = datetime.now(UTC)
    with psycopg.connect(_conninfo(database)) as connection:
        connection.execute(
            """
            INSERT INTO users (id, created_at, updated_at, email)
            VALUES (%s, %s, %s, %s)
            """,
            (uuid4(), now, now, "legacy-user@example.com"),
        )
        connection.commit()


def _delete_legacy_user(database: str) -> None:
    """删除 smoke 自己创建的旧账户，使正常 migration 可以继续."""
    with psycopg.connect(_conninfo(database)) as connection:
        connection.execute(
            "DELETE FROM users WHERE email = %s",
            ("legacy-user@example.com",),
        )
        connection.commit()


def _create_database(admin_database: str, test_database: str) -> None:
    """通过管理入口创建随机临时数据库.

    PostgreSQL 不允许在普通事务块中执行 ``CREATE DATABASE``，因此连接必须使用
    ``autocommit=True``。数据库名通过 ``sql.Identifier`` 安全引用，不能字符串拼接。
    """
    with psycopg.connect(_conninfo(admin_database), autocommit=True) as connection:
        connection.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(test_database)))


def _create_external_table(test_database: str) -> None:
    """创建一张 Alembic 永远不能管理的模拟外部表.

    只检查五张业务表是否存在还不够：错误的 downgrade 可能顺便删除其他组件的
    表。这个哨兵表必须在 upgrade、downgrade、再次 upgrade 后始终存在。
    """
    with psycopg.connect(_conninfo(test_database)) as connection:
        connection.execute(sql.SQL("CREATE TABLE {} (id integer PRIMARY KEY)").format(sql.Identifier(EXTERNAL_TABLE)))
        connection.commit()


def _drop_database(admin_database: str, test_database: str) -> None:
    """终止临时连接并只删除本次随机数据库.

    PostgreSQL 不允许删除仍有连接占用的数据库，所以先终止目标数据库连接。
    WHERE 条件严格使用随机数据库名，并排除当前管理连接自身。
    """
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
    """执行 upgrade、downgrade、再次 upgrade 与 schema diff 检查."""
    started_at = perf_counter()
    admin_database = settings.POSTGRES_DB
    test_database = f"deep_research_migration_{uuid4().hex[:12]}"
    previous_override = os.environ.get("ALEMBIC_DATABASE_URL")
    database_created = False
    cleanup_ok = False

    try:
        # 步骤 1：创建完全隔离的数据库，并预置一张不属于 Alembic 的表。
        _create_database(admin_database, test_database)
        database_created = True
        _create_external_table(test_database)

        # 步骤 2：只覆盖当前 Python 进程。migrations/env.py 会优先读取该 URL，
        # 因而后续 command.* 全部作用于随机临时库，而非配置的业务数据库。
        os.environ["ALEMBIC_DATABASE_URL"] = _temporary_database_url(test_database)

        alembic_config = Config(str(PROJECT_ROOT / "alembic.ini"))

        # 步骤 3：空库升级到 head。预期得到五张业务表、alembic_version 和
        # 外部 checkpoints；精确集合比较也能发现 migration 意外创建其他表。
        command.upgrade(alembic_config, "head")
        tables_after_first_upgrade = _public_tables(test_database)
        first_upgrade_matches = tables_after_first_upgrade == MANAGED_TABLES | {
            "alembic_version",
            EXTERNAL_TABLE,
        }
        first_password_hash_column_matches = _password_hash_column_matches(test_database)

        # 步骤 4：先只回退第二个 migration，制造一条没有 credential 的旧用户。
        # 再次升级必须在执行 DDL 前明确拒绝，并且事务回滚后不能留下半截列。
        command.downgrade(alembic_config, INITIAL_BUSINESS_REVISION)
        _insert_legacy_user(test_database)
        try:
            command.upgrade(alembic_config, "head")
        except RuntimeError as exc:
            legacy_user_guard_triggered = "legacy user rows exist" in str(exc)
        else:
            legacy_user_guard_triggered = False
        legacy_guard_left_schema_unchanged = not _password_hash_column_exists(test_database)

        # 清除 smoke 数据后，完全相同的 migration 应在空 users 表上正常完成。
        _delete_legacy_user(test_database)
        command.upgrade(alembic_config, "head")

        # 步骤 5：降级到 base。业务表必须按外键依赖逆序删除，但外部表以及
        # Alembic 自己的版本表必须保留。
        command.downgrade(alembic_config, "base")
        tables_after_downgrade = _public_tables(test_database)
        downgrade_preserves_only_external = tables_after_downgrade == {
            "alembic_version",
            EXTERNAL_TABLE,
        }

        # 步骤 6：再次升级，证明 downgrade 没有留下阻止重建的残余对象。
        command.upgrade(alembic_config, "head")
        tables_after_second_upgrade = _public_tables(test_database)
        second_upgrade_matches = tables_after_second_upgrade == MANAGED_TABLES | {
            "alembic_version",
            EXTERNAL_TABLE,
        }
        second_password_hash_column_matches = _password_hash_column_matches(test_database)

        # 步骤 7：比较当前数据库结构与 SQLModel.metadata。若模型已经变化却没有
        # 新 migration，command.check 会抛异常；无异常才表示不存在 schema drift。
        command.check(alembic_config)
        no_schema_diff = True

        ok = all(
            (
                first_upgrade_matches,
                first_password_hash_column_matches,
                legacy_user_guard_triggered,
                legacy_guard_left_schema_unchanged,
                downgrade_preserves_only_external,
                second_upgrade_matches,
                second_password_hash_column_matches,
                no_schema_diff,
            )
        )
        return {
            "ok": ok,
            "first_upgrade_matches": first_upgrade_matches,
            "first_password_hash_column_matches": first_password_hash_column_matches,
            "legacy_user_guard_triggered": legacy_user_guard_triggered,
            "legacy_guard_left_schema_unchanged": legacy_guard_left_schema_unchanged,
            "downgrade_preserves_only_external": downgrade_preserves_only_external,
            "second_upgrade_matches": second_upgrade_matches,
            "second_password_hash_column_matches": second_password_hash_column_matches,
            "external_table_preserved": EXTERNAL_TABLE in tables_after_second_upgrade,
            "no_schema_diff": no_schema_diff,
            "elapsed_ms": _elapsed_ms(started_at),
        }
    finally:
        # 即使 upgrade/check 中途抛异常，也必须先恢复调用者原有环境，避免之后
        # 在同一终端运行 Alembic 时继续误连已经删除的临时数据库。
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

        # Python 在执行 try 中的 return 之前仍会执行 finally。清理失败不能静默
        # 返回 ok=true，否则反复 smoke 会在 PostgreSQL 中泄漏大量测试数据库。
        if database_created and not cleanup_ok:
            raise RuntimeError("临时迁移数据库清理失败")


def main() -> int:
    """打印单行、不含凭据的 JSON 验收摘要."""
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

    # 只有 _run_smoke 正常返回且 finally 清理成功才会到达这里，因此此字段也
    # 是清理路径的证明，而不是在迁移完成前提前写入的乐观值。
    summary["cleanup_ok"] = True
    print(json.dumps(summary))
    return 0 if summary["ok"] is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
