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
from datetime import UTC, datetime, timedelta
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
from app.repositories import (
    ChatSessionRepository,
    DocumentRepository,
    ResearchTaskRepository,
    UserRepository,
)
from app.services.auth import PasswordHasher, TokenService


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONNECTION_TIMEOUT_SECONDS = 10


def _elapsed_ms(started_at: float) -> float:
    """返回不包含请求体、token 或连接信息的毫秒耗时."""
    return round((perf_counter() - started_at) * 1000, 2)


def _tamper_token(token: str) -> str:
    """只修改 JWT 签名段的一个字符，构造稳定的篡改样本.

    Args:
        token: 由当前 smoke TokenService 正常签发的三段式 JWT。

    Returns:
        header 和 payload 保持不变、签名必然不同的 token。

    Raises:
        ValueError: 输入不是完整三段 JWT，或签名段为空。

    Notes:
        不修改 payload 可以准确验证“签名不匹配”分支。脚本不会输出原 token 或
        篡改后的 token，二者只在当前进程内作为 HTTP header 使用。
    """
    parts = token.split(".")
    if len(parts) != 3 or not parts[2]:
        raise ValueError("expected a three-part JWT")
    replacement = "A" if parts[2][0] != "A" else "B"
    parts[2] = replacement + parts[2][1:]
    return ".".join(parts)


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
    """通过真实 HTTP 与 PostgreSQL 验收认证、授权和所有权隔离.

    Args:
        database: 已完成 Alembic migration 的随机临时数据库名。所有 ORM Session
            都连接这个库，不会向开发数据库写入 smoke 用户或资源。

    Returns:
        只包含布尔断言、计数与耗时的脱敏摘要。邮箱、密码、hash、JWT 和数据库
        连接凭据均不会进入返回值或最终 JSON。

    Raises:
        Exception: 迁移、真实数据库、Argon2id、JWT 或 FastAPI 调用出现意外故障时
            向上传播，由 main 仅输出异常类型并仍执行数据库清理。
    """
    engine, session_factory = create_orm_runtime(_runtime_settings(database))
    password_hasher = PasswordHasher()

    # 这是 smoke 进程独有的高强度 secret。依赖覆盖让正式 auth route 使用这个
    # TokenService，因此无需读取、修改或输出开发/生产环境的 JWT secret。
    smoke_secret = SecretStr(secrets.token_urlsafe(48))
    token_service = TokenService(
        secret_key=smoke_secret,
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
    second_email = f"auth-second-{suffix}@example.com"
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

            # 第二个真实用户用于 403 和对象级授权验收。两个用户都经过正式注册
            # route、Argon2id、事务与 JWT 签发，不直接向 users 表塞测试行。
            second_registration = await client.post(
                "/api/v1/auth/register",
                json={"email": second_email, "password": password},
            )
            second_registration_body = second_registration.json()
            second_token = str(second_registration_body.get("access_token", ""))
            second_claims = token_service.decode_access_token(second_token)

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

            bearer_headers = {"Authorization": f"Bearer {registration_token}"}
            valid_me = await client.get(
                "/api/v1/auth/me",
                headers=bearer_headers,
            )
            valid_self_scope = await client.get(
                f"/api/v1/auth/users/{registration_claims.sub}",
                headers=bearer_headers,
            )
            forbidden_other_scope = await client.get(
                f"/api/v1/auth/users/{second_claims.sub}",
                headers=bearer_headers,
            )

            # 用相同签名 key 为不存在用户签 token：签名与 claims 都合法，但
            # get_current_user 的数据库确认必须把它拒绝为与其他认证失败相同的 401。
            unknown_user_token = token_service.create_access_token(
                user_id=uuid4(),
            ).access_token

            # 使用过去时钟签发短期 token，再交给当前时钟的正式 dependency 验证，
            # 可以稳定进入 exp 过期分支，而不需要 sleep 或修改系统时间。
            expired_token = (
                TokenService(
                    secret_key=smoke_secret,
                    access_token_ttl=timedelta(minutes=1),
                    clock=lambda: datetime.now(UTC) - timedelta(hours=1),
                )
                .create_access_token(user_id=registration_claims.sub)
                .access_token
            )

            authentication_failures = (
                await client.get("/api/v1/auth/me"),
                await client.get(
                    "/api/v1/auth/me",
                    headers={"Authorization": "Basic not-a-bearer-token"},
                ),
                await client.get(
                    "/api/v1/auth/me",
                    headers={"Authorization": f"Bearer {_tamper_token(registration_token)}"},
                ),
                await client.get(
                    "/api/v1/auth/me",
                    headers={"Authorization": f"Bearer {expired_token}"},
                ),
                await client.get(
                    "/api/v1/auth/me",
                    headers={"Authorization": f"Bearer {unknown_user_token}"},
                ),
            )

            openapi = (await client.get("/openapi.json")).json()

        # 为用户 B 创建三类真实资源。事务成功后再使用全新 Session 分别以 A/B
        # 身份查询，证明所有权限制存在于 Repository SQL，而不是只在 route 做比较。
        async with session_factory() as session:
            async with session.begin():
                chat_sessions = ChatSessionRepository(session)
                documents = DocumentRepository(session)
                research_tasks = ResearchTaskRepository(session)
                owned_chat_session = await chat_sessions.create(
                    user_id=second_claims.sub,
                    title="Authorization smoke",
                )
                owned_document = await documents.create(
                    user_id=second_claims.sub,
                    original_filename="authorization-smoke.txt",
                    content_type="text/plain",
                )
                owned_research_task = await research_tasks.create(
                    user_id=second_claims.sub,
                    topic="Verify object-level authorization",
                    chat_session_id=owned_chat_session.id,
                )

        async with session_factory() as session:
            chat_sessions = ChatSessionRepository(session)
            documents = DocumentRepository(session)
            research_tasks = ResearchTaskRepository(session)

            owner_resource_results = (
                await chat_sessions.get_by_id(
                    owned_chat_session.id,
                    user_id=second_claims.sub,
                ),
                await documents.get_by_id(
                    owned_document.id,
                    user_id=second_claims.sub,
                ),
                await research_tasks.get_by_id(
                    owned_research_task.id,
                    user_id=second_claims.sub,
                ),
            )
            cross_user_resource_results = (
                await chat_sessions.get_by_id(
                    owned_chat_session.id,
                    user_id=registration_claims.sub,
                ),
                await documents.get_by_id(
                    owned_document.id,
                    user_id=registration_claims.sub,
                ),
                await research_tasks.get_by_id(
                    owned_research_task.id,
                    user_id=registration_claims.sub,
                ),
            )

        # 使用全新 Session 查询，证明注册事务已经 commit，并且数据库持久化的是
        # 可验证 Argon2id hash，而不是明文或只存在于旧 Session 的对象。
        async with session_factory() as session:
            users = UserRepository(session)
            stored_user = await users.get_by_email(email)
            stored_second_user = await users.get_by_email(second_email)
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
        second_registration_succeeded = (
            second_registration.status_code == 201
            and second_registration_body.get("token_type") == "bearer"
            and bool(second_token)
            and stored_second_user is not None
            and second_claims.sub == stored_second_user.id
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

        me_body = valid_me.json()
        current_user_resolved_from_database = (
            valid_me.status_code == 200
            and me_body
            == {
                "email": email,
                "user_id": str(registration_claims.sub),
            }
            and "access_token" not in me_body
            and "password_hash" not in me_body
        )
        self_scope_allowed = valid_self_scope.status_code == 200 and valid_self_scope.json() == me_body
        forbidden_error = forbidden_other_scope.json().get("error", {})
        other_user_scope_forbidden = (
            forbidden_other_scope.status_code == 403 and forbidden_error.get("message") == "Insufficient permissions"
        )

        failure_errors = tuple(response.json().get("error", {}) for response in authentication_failures)
        authentication_failures_are_indistinguishable = (
            bool(failure_errors)
            and all(response.status_code == 401 for response in authentication_failures)
            and all(response.headers.get("WWW-Authenticate") == "Bearer" for response in authentication_failures)
            and all(error == failure_errors[0] for error in failure_errors[1:])
            and failure_errors[0].get("message") == "Could not validate credentials"
        )

        owner_can_read_all_resources = all(resource is not None for resource in owner_resource_results)
        cross_user_resources_are_hidden = all(resource is None for resource in cross_user_resource_results)

        paths = openapi.get("paths", {})
        register_responses = paths.get("/api/v1/auth/register", {}).get("post", {}).get("responses", {})
        login_responses = paths.get("/api/v1/auth/login", {}).get("post", {}).get("responses", {})
        me_operation = paths.get("/api/v1/auth/me", {}).get("get", {})
        user_scope_responses = paths.get("/api/v1/auth/users/{user_id}", {}).get("get", {}).get("responses", {})
        security_schemes = openapi.get("components", {}).get("securitySchemes", {})
        http_bearer_scheme_documented = any(
            scheme.get("type") == "http" and str(scheme.get("scheme", "")).casefold() == "bearer"
            for scheme in security_schemes.values()
        )
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
            and bool(me_operation.get("security"))
            and me_operation.get("responses", {})
            .get("200", {})
            .get("content", {})
            .get("application/json", {})
            .get("schema", {})
            .get("$ref")
            == "#/components/schemas/AuthenticatedUser"
            and user_scope_responses.get("403", {})
            .get("content", {})
            .get("application/json", {})
            .get("schema", {})
            .get("$ref")
            == "#/components/schemas/ErrorResponse"
            and http_bearer_scheme_documented
        )

        response_text = "\n".join(
            (
                registration.text,
                duplicate.text,
                login.text,
                wrong_password_response.text,
                unknown_email_response.text,
                valid_me.text,
                valid_self_scope.text,
                forbidden_other_scope.text,
                *(response.text for response in authentication_failures),
            )
        )
        responses_hide_credentials = (
            password not in response_text
            and wrong_password not in response_text
            and (stored_user is None or stored_user.password_hash not in response_text)
        )

        return {
            "registration_succeeded": registration_succeeded,
            "second_registration_succeeded": second_registration_succeeded,
            "login_succeeded": login_succeeded,
            "tokens_belong_to_same_user": tokens_belong_to_same_user,
            "independently_signed_tokens": independently_signed_tokens,
            "duplicate_rejected": duplicate_rejected,
            "invalid_credentials_are_indistinguishable": invalid_credentials_are_indistinguishable,
            "stored_credential_is_safe": stored_credential_is_safe,
            "only_expected_users_persisted": len(all_users) == 2,
            "current_user_resolved_from_database": current_user_resolved_from_database,
            "authentication_failures_are_indistinguishable": authentication_failures_are_indistinguishable,
            "self_scope_allowed": self_scope_allowed,
            "other_user_scope_forbidden": other_user_scope_forbidden,
            "owner_can_read_all_resources": owner_can_read_all_resources,
            "cross_user_resources_are_hidden": cross_user_resources_are_hidden,
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
