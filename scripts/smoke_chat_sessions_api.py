"""在随机真实 PostgreSQL 上验收业务会话 HTTP 所有权边界.

脚本执行正式 Alembic migration、Auth route、JWT、ChatSessionService 和 Repository。
它不执行 LangGraph 或模型请求，因为 10F-B 只建立产品资源与数据库授权边界。
随机邮箱、密码、token、数据库连接信息和资源 UUID 均不会进入最终 JSON 摘要。
"""

import asyncio
import json
import os
import secrets
import selectors
from collections.abc import AsyncIterator
from pathlib import Path
from time import perf_counter
from uuid import UUID, uuid4

import psycopg
from alembic import command
from alembic.config import Config
from asgi_correlation_id import CorrelationIdMiddleware
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from psycopg import sql
from psycopg.conninfo import make_conninfo
from pydantic import SecretStr
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dependencies import get_db_session, get_token_service
from app.api.v1.auth import router as auth_router
from app.api.v1.chat_sessions import router as chat_sessions_router
from app.core.config import Settings, settings
from app.core.exception_handlers import register_exception_handlers
from app.infrastructure.chat_session_ownership import PostgresChatSessionOwnershipVerifier
from app.infrastructure.database import build_orm_database_url, create_orm_runtime
from app.repositories import ChatSessionRepository
from app.services.auth import TokenService
from app.services.chat_session_ownership import ChatSessionNotFoundError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONNECTION_TIMEOUT_SECONDS = 10
TEMPORARY_DATABASE_PREFIX = "deep_research_chat_sessions_"


def _elapsed_ms(started_at: float) -> float:
    """返回不包含请求正文、token 或连接信息的毫秒耗时."""
    return round((perf_counter() - started_at) * 1000, 2)


def _conninfo(database: str) -> str:
    """构造仅供 psycopg 管理临时数据库使用的连接参数.

    Args:
        database: 已存在的 PostgreSQL 数据库名。

    Returns:
        psycopg 接受的结构化 conninfo。该值绝不进入 smoke 输出。
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
    """构造只在本进程中交给 Alembic 的临时数据库 URL."""
    return build_orm_database_url(settings).set(database=database).render_as_string(hide_password=False)


def _create_database(admin_database: str, test_database: str) -> None:
    """在 autocommit 连接中创建随机隔离数据库.

    Args:
        admin_database: 已存在且允许执行 CREATE DATABASE 的管理入口。
        test_database: 带固定安全前缀的随机数据库名。
    """
    if not test_database.startswith(TEMPORARY_DATABASE_PREFIX):
        raise ValueError("refusing to create a database without the smoke prefix")
    with psycopg.connect(_conninfo(admin_database), autocommit=True) as connection:
        connection.execute(
            sql.SQL("CREATE DATABASE {}").format(sql.Identifier(test_database)),
        )


def _drop_database(admin_database: str, test_database: str) -> None:
    """终止临时库残留连接并删除本次 smoke 数据库.

    Args:
        admin_database: 执行清理的管理数据库。
        test_database: 本脚本创建的随机数据库。

    Raises:
        ValueError: 数据库名不带固定 smoke 前缀时拒绝删除。
    """
    if not test_database.startswith(TEMPORARY_DATABASE_PREFIX):
        raise ValueError("refusing to drop a database without the smoke prefix")
    with psycopg.connect(_conninfo(admin_database), autocommit=True) as connection:
        connection.execute(
            """
            SELECT pg_terminate_backend(pid)
            FROM pg_stat_activity
            WHERE datname = %s AND pid <> pg_backend_pid()
            """,
            (test_database,),
        )
        connection.execute(
            sql.SQL("DROP DATABASE IF EXISTS {}").format(
                sql.Identifier(test_database),
            )
        )


def _runtime_settings(database: str) -> Settings:
    """复制应用设置，只替换随机数据库名并缩小 smoke 连接池."""
    config = Settings()
    config.POSTGRES_DB = database
    config.POSTGRES_ORM_POOL_SIZE = 3
    config.POSTGRES_ORM_MAX_OVERFLOW = 0
    return config


def _selector_loop_factory() -> asyncio.AbstractEventLoop:
    """创建 Windows psycopg 异步驱动所需的 Selector event loop."""
    return asyncio.SelectorEventLoop(selectors.SelectSelector())


def _authorization_header(token: str) -> dict[str, str]:
    """构造只在当前进程内使用的 Bearer header.

    Args:
        token: 正式 TokenService 签发的 access token。

    Returns:
        可传给 httpx 的 Authorization header。调用方不得记录该字典。
    """
    return {"Authorization": f"Bearer {token}"}


async def _exercise_chat_session_api(database: str) -> dict[str, bool | int | float]:
    """通过真实 HTTP 和 PostgreSQL 验证业务会话所有权.

    Args:
        database: 已执行正式 Alembic migration 的随机数据库名。

    Returns:
        只含布尔检查、计数和耗时的脱敏摘要。

    Raises:
        Exception: migration 之后的数据库、认证或 API 行为不符合预期时向上传播；
            外层仍会在 finally 中清理临时数据库。
    """
    started_at = perf_counter()
    engine, session_factory = create_orm_runtime(_runtime_settings(database))

    # smoke 使用进程内随机强 secret，不读取或输出开发环境 JWT secret。
    token_service = TokenService(secret_key=SecretStr(secrets.token_urlsafe(48)))

    app = FastAPI()
    register_exception_handlers(app)
    app.add_middleware(CorrelationIdMiddleware)
    app.include_router(auth_router, prefix="/api/v1")
    app.include_router(chat_sessions_router, prefix="/api/v1")

    async def override_db_session() -> AsyncIterator[AsyncSession]:
        """让每个 HTTP 请求获得随机临时库中的独立 AsyncSession."""
        async with session_factory() as session:
            yield session

    def override_token_service() -> TokenService:
        """让注册与后续认证共享同一真实 JWT 签名服务."""
        return token_service

    app.dependency_overrides[get_db_session] = override_db_session
    app.dependency_overrides[get_token_service] = override_token_service

    suffix = uuid4().hex
    password = "Real-Chat-Session-Smoke-2026!"

    try:
        # ASGITransport 不监听 TCP，但请求仍经过 FastAPI 解析、依赖注入、route、
        # application service、Repository 和真实 PostgreSQL transaction。
        transport = ASGITransport(app=app, raise_app_exceptions=False)
        async with AsyncClient(transport=transport, base_url="http://testserver") as client:
            registration_a = await client.post(
                "/api/v1/auth/register",
                json={
                    "email": f"session-a-{suffix}@example.com",
                    "password": password,
                },
            )
            registration_b = await client.post(
                "/api/v1/auth/register",
                json={
                    "email": f"session-b-{suffix}@example.com",
                    "password": password,
                },
            )
            token_a = str(registration_a.json().get("access_token", ""))
            token_b = str(registration_b.json().get("access_token", ""))
            claims_a = token_service.decode_access_token(token_a)
            claims_b = token_service.decode_access_token(token_b)
            headers_a = _authorization_header(token_a)
            headers_b = _authorization_header(token_b)

            # A 创建两个会话，B 创建一个会话。请求 body 没有 user_id；所有权只能
            # 来自当前 Bearer token 对应的 AuthenticatedUser。
            create_a_first = await client.post(
                "/api/v1/chat/sessions",
                headers=headers_a,
                json={"title": "  A first session  "},
            )
            create_a_default = await client.post(
                "/api/v1/chat/sessions",
                headers=headers_a,
                json={},
            )
            create_b = await client.post(
                "/api/v1/chat/sessions",
                headers=headers_b,
                json={"title": "B only session"},
            )

            body_a_first = create_a_first.json()
            body_a_default = create_a_default.json()
            body_b = create_b.json()
            thread_a_first = UUID(str(body_a_first.get("thread_id")))

            list_a = await client.get("/api/v1/chat/sessions", headers=headers_a)
            list_b = await client.get("/api/v1/chat/sessions", headers=headers_b)
            read_a_own = await client.get(
                f"/api/v1/chat/sessions/{thread_a_first}",
                headers=headers_a,
            )

            # 相同真实 UUID 用 B 查询，以及一个完全随机 UUID 用 B 查询，都必须走
            # 同一 404 协议。否则接口会泄漏“这个 UUID 存在但属于别人”。
            read_b_cross_user = await client.get(
                f"/api/v1/chat/sessions/{thread_a_first}",
                headers=headers_b,
            )
            read_b_absent = await client.get(
                f"/api/v1/chat/sessions/{uuid4()}",
                headers=headers_b,
            )
            unauthenticated_list = await client.get("/api/v1/chat/sessions")

            # 即使伪造的 user_id 恰好是另一个真实用户，strict schema 也必须在进入
            # service 前返回 422，不能让客户端选择资源所有者。
            injected_owner = await client.post(
                "/api/v1/chat/sessions",
                headers=headers_a,
                json={"title": "Injected", "user_id": str(claims_b.sub)},
            )
            openapi = (await client.get("/openapi.json")).json()

        # 用独立新 Session 读取真实提交结果，排除旧 identity map 造成的假成功。
        async with session_factory() as session:
            async with session.begin():
                repository = ChatSessionRepository(session)
                stored_a = await repository.list_by_user(claims_a.sub)
                stored_b = await repository.list_by_user(claims_b.sub)
                cross_user_lookup = await repository.get_by_id(
                    thread_a_first,
                    user_id=claims_b.sub,
                )

        # production verifier 长期只保存 sessionmaker，每次调用创建独立 Session。
        # 三条调用分别验证 owner 命中、cross-user 隐藏和随机不存在资源；它们
        # 使用真实 PostgreSQL 查询，不是内存集合或 fake Repository。
        ownership_verifier = PostgresChatSessionOwnershipVerifier(session_factory)
        await ownership_verifier.require_owned(
            session_id=thread_a_first,
            user_id=claims_a.sub,
        )

        cross_user_verifier_rejected = False
        try:
            await ownership_verifier.require_owned(
                session_id=thread_a_first,
                user_id=claims_b.sub,
            )
        except ChatSessionNotFoundError:
            cross_user_verifier_rejected = True

        absent_verifier_rejected = False
        try:
            await ownership_verifier.require_owned(
                session_id=uuid4(),
                user_id=claims_b.sub,
            )
        except ChatSessionNotFoundError:
            absent_verifier_rejected = True

        list_a_sessions = list_a.json().get("sessions", [])
        list_b_sessions = list_b.json().get("sessions", [])
        cross_error = read_b_cross_user.json().get("error", {})
        absent_error = read_b_absent.json().get("error", {})
        paths = openapi.get("paths", {})

        creation_contract_ok = (
            registration_a.status_code == 201
            and registration_b.status_code == 201
            and create_a_first.status_code == 201
            and create_a_default.status_code == 201
            and create_b.status_code == 201
            and body_a_first.get("title") == "A first session"
            and body_a_default.get("title") == "New chat"
            and isinstance(body_b.get("thread_id"), str)
        )
        lists_are_owner_scoped = (
            list_a.status_code == 200
            and list_b.status_code == 200
            and len(list_a_sessions) == 2
            and len(list_b_sessions) == 1
            and {item["thread_id"] for item in list_a_sessions}
            == {body_a_first["thread_id"], body_a_default["thread_id"]}
            and list_b_sessions[0]["thread_id"] == body_b["thread_id"]
        )
        hidden_lookup_contract = (
            read_b_cross_user.status_code == 404
            and read_b_absent.status_code == 404
            and cross_error.get("code") == absent_error.get("code") == "HTTP_ERROR"
            and cross_error.get("message") == absent_error.get("message") == "Chat session was not found"
        )
        openapi_documents_sessions = all(
            path in paths
            for path in (
                "/api/v1/chat/sessions",
                "/api/v1/chat/sessions/{thread_id}",
            )
        )

        return {
            "creation_contract_ok": creation_contract_ok,
            "lists_are_owner_scoped": lists_are_owner_scoped,
            "owner_can_read_session": (
                read_a_own.status_code == 200 and read_a_own.json().get("thread_id") == str(thread_a_first)
            ),
            "cross_user_and_absent_are_indistinguishable": hidden_lookup_contract,
            "unauthenticated_request_rejected": unauthenticated_list.status_code == 401,
            "body_cannot_select_owner": injected_owner.status_code == 422,
            "database_commits_are_owner_scoped": (
                len(stored_a) == 2 and len(stored_b) == 1 and cross_user_lookup is None
            ),
            "shared_verifier_accepts_owner": True,
            "shared_verifier_rejects_cross_user": cross_user_verifier_rejected,
            "shared_verifier_rejects_absent": absent_verifier_rejected,
            "openapi_documents_sessions": openapi_documents_sessions,
            "persisted_session_count": len(stored_a) + len(stored_b),
            "exercise_ms": _elapsed_ms(started_at),
        }
    finally:
        app.dependency_overrides.clear()
        await engine.dispose()


def _run_smoke() -> dict[str, object]:
    """创建、迁移、验收并删除随机临时 PostgreSQL 数据库."""
    started_at = perf_counter()
    admin_database = settings.POSTGRES_DB
    test_database = f"{TEMPORARY_DATABASE_PREFIX}{uuid4().hex[:12]}"
    previous_override = os.environ.get("ALEMBIC_DATABASE_URL")
    database_created = False
    cleanup_ok = False

    try:
        _create_database(admin_database, test_database)
        database_created = True
        os.environ["ALEMBIC_DATABASE_URL"] = _temporary_database_url(test_database)
        command.upgrade(Config(str(PROJECT_ROOT / "alembic.ini")), "head")

        checks = asyncio.run(
            _exercise_chat_session_api(test_database),
            loop_factory=_selector_loop_factory,
        )
        boolean_checks = tuple(value for value in checks.values() if isinstance(value, bool))
        return {
            "ok": bool(boolean_checks) and all(boolean_checks),
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
            raise RuntimeError("temporary Chat session database cleanup failed")


def main() -> int:
    """运行真实 PostgreSQL smoke，并输出脱敏 JSON 摘要."""
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
