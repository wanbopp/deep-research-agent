"""在随机真实 PostgreSQL 上验收注册与登录 HTTP 闭环.

脚本执行正式 Alembic migration、AuthService、FastAPI auth router、Argon2id 和 JWT
代码，只把数据库名与 TokenService 依赖替换为本次 smoke 的隔离资源。所有数据库 DDL
都发生在随机临时库中，JWT secret 只存在于当前进程，最终 JSON 不输出任何 credential。
"""

import asyncio
import json
import os
import secrets
import selectors
from collections.abc import AsyncIterator
from pathlib import Path
from time import perf_counter
from uuid import uuid4

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
from app.core.config import Settings, settings
from app.core.exception_handlers import register_exception_handlers
from app.infrastructure.database import build_orm_database_url, create_orm_runtime
from app.repositories import UserRepository
from app.services.auth import PasswordHasher, TokenService


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONNECTION_TIMEOUT_SECONDS = 10


def _elapsed_ms(started_at: float) -> float:
    """返回不包含请求体、token 或连接信息的毫秒耗时."""
    return round((perf_counter() - started_at) * 1000, 2)


def _conninfo(database: str) -> str:
    """用结构化参数生成仅交给 psycopg 使用的连接字符串."""
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
    """在 autocommit 管理连接中创建随机隔离数据库."""
    with psycopg.connect(_conninfo(admin_database), autocommit=True) as connection:
        connection.execute(
            sql.SQL("CREATE DATABASE {}").format(sql.Identifier(test_database)),
        )


def _drop_database(admin_database: str, test_database: str) -> None:
    """终止残留连接后，只删除本次 smoke 的随机数据库."""
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
    """复制应用配置，只隔离数据库名并缩小本次 smoke 的 ORM pool."""
    config = Settings()
    config.POSTGRES_DB = database
    config.POSTGRES_ORM_POOL_SIZE = 2
    config.POSTGRES_ORM_MAX_OVERFLOW = 0
    return config


def _selector_loop_factory() -> asyncio.AbstractEventLoop:
    """创建 Windows psycopg async 所需的局部 Selector event loop."""
    return asyncio.SelectorEventLoop(selectors.SelectSelector())


async def _exercise_auth_http(database: str) -> dict[str, bool | int | float]:
    """通过真实 HTTP 边界执行注册、登录和安全失败路径."""
    engine, session_factory = create_orm_runtime(_runtime_settings(database))
    password_hasher = PasswordHasher()

    # 这是 smoke 进程独有的高强度 secret。依赖覆盖让正式 auth route 使用这个
    # TokenService，因此无需读取、修改或输出开发/生产环境的 JWT secret。
    token_service = TokenService(
        secret_key=SecretStr(secrets.token_urlsafe(48)),
    )

    app = FastAPI()
    register_exception_handlers(app)
    app.add_middleware(CorrelationIdMiddleware)
    app.include_router(auth_router, prefix="/api/v1")

    async def override_db_session() -> AsyncIterator[AsyncSession]:
        """让每个 HTTP 请求获得独立的真实临时库 Session."""
        async with session_factory() as session:
            yield session

    def override_token_service() -> TokenService:
        """让所有 smoke 请求共享同一真实签名 key."""
        return token_service

    app.dependency_overrides[get_db_session] = override_db_session
    app.dependency_overrides[get_token_service] = override_token_service

    suffix = uuid4().hex
    email = f"auth-{suffix}@example.com"
    unknown_email = f"unknown-{suffix}@example.com"
    password = "Real-Auth-Smoke-Password-2026!"
    wrong_password = "Wrong-Auth-Smoke-Password-2026!"

    try:
        # ASGITransport 仍然通过 FastAPI 的 HTTP 请求解析、dependency、route 和
        # response serialization，只省去监听真实端口；数据库和密码学均为真实实现。
        transport = ASGITransport(app=app, raise_app_exceptions=False)
        async with AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            registration = await client.post(
                "/api/v1/auth/register",
                json={"email": f"  {email.upper()}  ", "password": password},
            )
            registration_body = registration.json()
            registration_token = str(registration_body.get("access_token", ""))

            # 注册成功后立即用正式 TokenService 验签。此时 sub 是可信用户 UUID，
            # 但 9E 仍需再查数据库，才会构造 AuthenticatedUser。
            registration_claims = token_service.decode_access_token(registration_token)

            duplicate = await client.post(
                "/api/v1/auth/register",
                json={"email": email, "password": password},
            )

            login = await client.post(
                "/api/v1/auth/login",
                json={"email": email, "password": password},
            )
            login_body = login.json()
            login_token = str(login_body.get("access_token", ""))
            login_claims = token_service.decode_access_token(login_token)

            wrong_password_started = perf_counter()
            wrong_password_response = await client.post(
                "/api/v1/auth/login",
                json={"email": email, "password": wrong_password},
            )
            wrong_password_ms = _elapsed_ms(wrong_password_started)

            unknown_email_started = perf_counter()
            unknown_email_response = await client.post(
                "/api/v1/auth/login",
                json={"email": unknown_email, "password": wrong_password},
            )
            unknown_email_ms = _elapsed_ms(unknown_email_started)

            openapi = (await client.get("/openapi.json")).json()

        # 使用全新 Session 查询，证明注册事务已经 commit，并且数据库持久化的是
        # 可验证 Argon2id hash，而不是明文或只存在于旧 Session 的对象。
        async with session_factory() as session:
            users = UserRepository(session)
            stored_user = await users.get_by_email(email)
            all_users = await users.list_all()

        stored_credential_is_safe = (
            stored_user is not None
            and stored_user.password_hash.startswith("$argon2id$")
            and stored_user.password_hash != password
            and await password_hasher.verify(
                SecretStr(password),
                stored_user.password_hash,
            )
        )

        registration_succeeded = (
            registration.status_code == 201
            and registration_body.get("token_type") == "bearer"
            and bool(registration_token)
        )
        login_succeeded = login.status_code == 200 and login_body.get("token_type") == "bearer" and bool(login_token)
        tokens_belong_to_same_user = (
            registration_claims.sub == login_claims.sub
            and stored_user is not None
            and registration_claims.sub == stored_user.id
        )
        independently_signed_tokens = registration_claims.jti != login_claims.jti

        duplicate_error = duplicate.json().get("error", {})
        duplicate_rejected = duplicate.status_code == 409 and duplicate_error.get("code") == "EMAIL_ALREADY_REGISTERED"

        wrong_error = wrong_password_response.json().get("error", {})
        unknown_error = unknown_email_response.json().get("error", {})
        invalid_credentials_are_indistinguishable = (
            wrong_password_response.status_code == 401
            and unknown_email_response.status_code == 401
            and wrong_error == unknown_error
            and wrong_password_response.headers.get("WWW-Authenticate") == "Bearer"
            and unknown_email_response.headers.get("WWW-Authenticate") == "Bearer"
        )

        # 时间只作为教学观测值输出，不设置脆弱的绝对差值断言。真正的安全保证来自
        # AuthService 两个分支都调用同一个 PasswordHasher.verify。
        timing_observed = wrong_password_ms > 0 and unknown_email_ms > 0

        paths = openapi.get("paths", {})
        register_responses = paths.get("/api/v1/auth/register", {}).get("post", {}).get("responses", {})
        login_responses = paths.get("/api/v1/auth/login", {}).get("post", {}).get("responses", {})
        openapi_documents_auth = (
            register_responses.get("201", {})
            .get("content", {})
            .get("application/json", {})
            .get("schema", {})
            .get("$ref")
            == "#/components/schemas/TokenResponse"
            and register_responses.get("409", {})
            .get("content", {})
            .get("application/json", {})
            .get("schema", {})
            .get("$ref")
            == "#/components/schemas/ErrorResponse"
            and login_responses.get("401", {})
            .get("content", {})
            .get("application/json", {})
            .get("schema", {})
            .get("$ref")
            == "#/components/schemas/ErrorResponse"
        )

        response_text = "\n".join(
            (
                registration.text,
                duplicate.text,
                login.text,
                wrong_password_response.text,
                unknown_email_response.text,
            )
        )
        responses_hide_credentials = (
            password not in response_text
            and wrong_password not in response_text
            and (stored_user is None or stored_user.password_hash not in response_text)
        )

        return {
            "registration_succeeded": registration_succeeded,
            "login_succeeded": login_succeeded,
            "tokens_belong_to_same_user": tokens_belong_to_same_user,
            "independently_signed_tokens": independently_signed_tokens,
            "duplicate_rejected": duplicate_rejected,
            "invalid_credentials_are_indistinguishable": invalid_credentials_are_indistinguishable,
            "stored_credential_is_safe": stored_credential_is_safe,
            "only_one_user_persisted": len(all_users) == 1,
            "responses_hide_credentials": responses_hide_credentials,
            "openapi_documents_auth": openapi_documents_auth,
            "timing_observed": timing_observed,
            "wrong_password_ms": wrong_password_ms,
            "unknown_email_ms": unknown_email_ms,
        }
    finally:
        app.dependency_overrides.clear()
        await engine.dispose()


def _run_smoke() -> dict[str, object]:
    """创建、迁移、验收并删除随机临时 PostgreSQL 数据库."""
    started_at = perf_counter()
    admin_database = settings.POSTGRES_DB
    test_database = f"deep_research_auth_{uuid4().hex[:12]}"
    previous_override = os.environ.get("ALEMBIC_DATABASE_URL")
    database_created = False
    cleanup_ok = False

    try:
        _create_database(admin_database, test_database)
        database_created = True
        os.environ["ALEMBIC_DATABASE_URL"] = _temporary_database_url(test_database)
        command.upgrade(Config(str(PROJECT_ROOT / "alembic.ini")), "head")

        checks = asyncio.run(
            _exercise_auth_http(test_database),
            loop_factory=_selector_loop_factory,
        )
        boolean_checks = tuple(value for value in checks.values() if isinstance(value, bool))
        ok = bool(boolean_checks) and all(boolean_checks)
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
            raise RuntimeError("临时认证数据库清理失败")


def main() -> int:
    """输出不包含邮箱、密码、哈希、token 或连接凭据的 JSON 摘要."""
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
