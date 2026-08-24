"""在隔离的真实数据库上运行 Repository 与事务验收测试。

本脚本创建一个随机命名的 PostgreSQL 数据库，使用真实 Alembic 迁移升级，
执行生产 Repository/service 类，最后在 ``finally`` 中删除该数据库。
全程不会写入配置的应用数据库。
"""

import asyncio
import json
import os
import selectors
from pathlib import Path
from time import perf_counter
from uuid import uuid4

import psycopg
from alembic import command
from alembic.config import Config
from psycopg import sql
from psycopg.conninfo import make_conninfo

from app.core.config import Settings, settings
from app.infrastructure.database import build_orm_database_url, create_orm_runtime
from app.models import DocumentStatus, ResearchTaskStatus
from app.repositories import (
    ChatSessionRepository,
    DocumentRepository,
    RepositoryConflictError,
    ResearchTaskRepository,
    UserRepository,
)
from app.services import UserAlreadyExistsError, UserNotFoundError, UserWorkspaceService


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONNECTION_TIMEOUT_SECONDS = 10


def _elapsed_ms(started_at: float) -> float:
    """返回耗时毫秒数，不暴露连接配置。"""
    return round((perf_counter() - started_at) * 1000, 2)


def _conninfo(database: str) -> str:
    """构建不含凭据的 psycopg 连接字符串。"""
    return make_conninfo(
        host=settings.POSTGRES_HOST,
        port=settings.POSTGRES_PORT,
        dbname=database,
        user=settings.POSTGRES_USER,
        password=settings.POSTGRES_PASSWORD,
        connect_timeout=CONNECTION_TIMEOUT_SECONDS,
    )


def _temporary_database_url(database: str) -> str:
    """构建仅用于本次进程的临时 Alembic URL。"""
    return build_orm_database_url(settings).set(database=database).render_as_string(hide_password=False)


def _create_database(admin_database: str, test_database: str) -> None:
    """为本次 smoke 测试创建一个随机命名的数据库。"""
    with psycopg.connect(_conninfo(admin_database), autocommit=True) as connection:
        connection.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(test_database)))


def _drop_database(admin_database: str, test_database: str) -> None:
    """关闭残留连接并仅删除随机测试数据库。"""
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


def _runtime_settings(database: str) -> Settings:
    """创建仅隔离数据库名和连接池大小不同的配置。"""
    config = Settings()
    config.POSTGRES_DB = database

    # Smoke 只需要少量顺序连接，使用 2/0 可以验证真实连接池，同时不占用正式
    # 应用 5/5 预算。生产工厂和 Repository 代码本身仍与应用完全相同。
    config.POSTGRES_ORM_POOL_SIZE = 2
    config.POSTGRES_ORM_MAX_OVERFLOW = 0
    return config


def _selector_loop_factory() -> asyncio.AbstractEventLoop:
    """创建 psycopg 异步模式在 Windows 上所需的 Selector 事件循环。

    Windows 默认使用 ProactorEventLoop，而 psycopg 的异步连接依赖文件描述符
    监听。把 loop factory 只传给本次 ``asyncio.run``，不会修改应用全局策略。
    """
    return asyncio.SelectorEventLoop(selectors.SelectSelector())


async def _exercise_repositories(database: str) -> dict[str, bool | int]:
    """执行真实 CRUD、所有权过滤、service 错误和事务回滚验证。"""
    engine, session_factory = create_orm_runtime(_runtime_settings(database))
    suffix = uuid4().hex
    first_email = f"repository-a-{suffix}@example.com"
    second_email = f"repository-b-{suffix}@example.com"
    rollback_email = f"rollback-{suffix}@example.com"

    try:
        # 第一部分：通过 application service 提交两个完整工作区。service 内部的
        # UserRepository 和 ChatSessionRepository 共享同一个 AsyncSession，退出
        # session.begin() 后才会一起 commit。
        async with session_factory() as session:
            service = UserWorkspaceService(session)
            first_workspace = await service.create_user_workspace(
                email=first_email,
                title="第一个工作区",
            )
            second_workspace = await service.create_user_workspace(
                email=second_email,
                title="第二个工作区",
            )
            second_first_session_id = second_workspace.chat_session_id
            first_second_session_id = await service.create_chat_session(
                user_id=first_workspace.user_id,
                title="后续会话",
            )

        # 第二部分：在一个显式事务中调用另外两个 Repository。这里仍然只有
        # service/smoke 控制 begin，Repository 的 create 只 flush、不 commit。
        async with session_factory() as session:
            documents = DocumentRepository(session)
            tasks = ResearchTaskRepository(session)
            async with session.begin():
                first_document = await documents.create(
                    user_id=first_workspace.user_id,
                    original_filename="first.pdf",
                    content_type="application/pdf",
                    status=DocumentStatus.READY,
                )
                await documents.create(
                    user_id=second_workspace.user_id,
                    original_filename="second.txt",
                    content_type="text/plain",
                )
                first_task = await tasks.create(
                    user_id=first_workspace.user_id,
                    chat_session_id=first_workspace.chat_session_id,
                    topic="Repository 事务边界",
                    status=ResearchTaskStatus.RUNNING,
                )
                await tasks.create(
                    user_id=second_workspace.user_id,
                    chat_session_id=second_workspace.chat_session_id,
                    topic="所有权过滤",
                )

        # 第三部分：用全新的 Session 读取，证明前面的 commit 已经真实落库，而
        # 不是只存在于 SQLAlchemy identity map 中。
        async with session_factory() as session:
            users = UserRepository(session)
            sessions = ChatSessionRepository(session)
            documents = DocumentRepository(session)
            tasks = ResearchTaskRepository(session)

            persisted_user = await users.get_by_id(first_workspace.user_id)
            persisted_session = await sessions.get_by_id(
                first_workspace.chat_session_id,
                user_id=first_workspace.user_id,
            )
            persisted_second_session = await sessions.get_by_id(
                second_first_session_id,
                user_id=second_workspace.user_id,
            )
            persisted_document = await documents.get_by_id(
                first_document.id,
                user_id=first_workspace.user_id,
            )
            persisted_task = await tasks.get_by_id(
                first_task.id,
                user_id=first_workspace.user_id,
            )
            all_users = await users.list_all()
            first_user_sessions = await sessions.list_by_user(first_workspace.user_id)
            first_user_documents = await documents.list_by_user(first_workspace.user_id)
            first_user_tasks = await tasks.list_by_user(first_workspace.user_id)

            repository_crud_ok = all(
                (
                    persisted_user is not None,
                    persisted_session is not None,
                    persisted_second_session is not None,
                    persisted_document is not None,
                    persisted_task is not None,
                    len(all_users) == 2,
                    len(first_user_sessions) == 2,
                    len(first_user_documents) == 1,
                    len(first_user_tasks) == 1,
                )
            )

            # 同一个资源 ID 换成另一个 user_id 后必须返回 None。这是 Repository
            # 对“资源归谁所有”的数据库过滤，不依赖 route 是否记得额外检查。
            ownership_isolated = all(
                (
                    await sessions.get_by_id(
                        first_second_session_id,
                        user_id=second_workspace.user_id,
                    )
                    is None,
                    await documents.get_by_id(
                        first_document.id,
                        user_id=second_workspace.user_id,
                    )
                    is None,
                    await tasks.get_by_id(
                        first_task.id,
                        user_id=second_workspace.user_id,
                    )
                    is None,
                )
            )
            not_found_returns_none = await users.get_by_id(uuid4()) is None

        # 第四部分：service 把 Repository 的结果转换为业务含义。Repository
        # 查询不到只返回 None；“当前用例必须存在用户”由 service 决定并抛业务错误。
        async with session_factory() as session:
            service = UserWorkspaceService(session)
            try:
                await service.create_chat_session(user_id=uuid4())
            except UserNotFoundError:
                missing_user_translated = True
            else:
                missing_user_translated = False

        async with session_factory() as session:
            service = UserWorkspaceService(session)
            try:
                await service.create_user_workspace(email=first_email)
            except UserAlreadyExistsError:
                duplicate_user_translated = True
            else:
                duplicate_user_translated = False

        # 第五部分：故意在同一事务写入两条相同邮箱。第二次 flush 命中唯一约束，
        # 异常离开 begin() 后 PostgreSQL 回滚整个事务，所以第一次 INSERT 也不能留下。
        async with session_factory() as session:
            users = UserRepository(session)
            try:
                async with session.begin():
                    await users.create(email=rollback_email)
                    await users.create(email=rollback_email)
            except RepositoryConflictError:
                second_write_failed = True
            else:
                second_write_failed = False

            # 仍使用同一个 AsyncSession 查询。若事务上下文没有正确 rollback，
            # SQLAlchemy 会抛 PendingRollbackError，而不是成功执行 SELECT。
            rolled_back_user = await users.get_by_email(rollback_email)
            rollback_removed_first_write = rolled_back_user is None
            session_reusable_after_rollback = second_write_failed and rollback_removed_first_write

        return {
            "repository_crud_ok": repository_crud_ok,
            "ownership_isolated": ownership_isolated,
            "not_found_returns_none": not_found_returns_none,
            "service_transaction_committed": persisted_user is not None and persisted_session is not None,
            "service_errors_translated": missing_user_translated and duplicate_user_translated,
            "rollback_removed_first_write": rollback_removed_first_write,
            "session_reusable_after_rollback": session_reusable_after_rollback,
            "first_user_session_count": len(first_user_sessions),
            "second_user_first_session_preserved": persisted_second_session is not None,
        }
    finally:
        # dispose 先关闭 SQLAlchemy pool，外层才能可靠删除临时数据库。
        await engine.dispose()


def _run_smoke() -> dict[str, object]:
    """创建、迁移、测试并删除隔离数据库。"""
    started_at = perf_counter()
    admin_database = settings.POSTGRES_DB
    test_database = f"deep_research_repository_{uuid4().hex[:12]}"
    previous_override = os.environ.get("ALEMBIC_DATABASE_URL")
    database_created = False
    cleanup_ok = False

    try:
        _create_database(admin_database, test_database)
        database_created = True
        os.environ["ALEMBIC_DATABASE_URL"] = _temporary_database_url(test_database)
        command.upgrade(Config(str(PROJECT_ROOT / "alembic.ini")), "head")

        checks = asyncio.run(
            _exercise_repositories(test_database),
            loop_factory=_selector_loop_factory,
        )
        ok = all(value is True or isinstance(value, int) and value > 0 for value in checks.values())
        return {
            "ok": ok,
            **checks,
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

        if database_created and not cleanup_ok:
            raise RuntimeError("临时 Repository 数据库清理失败")


def main() -> int:
    """打印一条不含凭据的 JSON 摘要作为检查记录。"""
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

    summary["cleanup_ok"] = True
    print(json.dumps(summary))
    return 0 if summary["ok"] is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
