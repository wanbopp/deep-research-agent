"""通过真实 PostgreSQL、Redis 和 saver 验收业务会话 DELETE API.

本 smoke 验证的是 10F-E 的 HTTP 适配边界，不执行 LangGraph 或模型请求：

* 随机临时 PostgreSQL 数据库执行正式 Alembic migration；
* production lifespan 创建真实 ORM、Redis guard 与 AsyncPostgresSaver；
* 正式 Auth/ChatSession route 完成注册、创建、删除和错误响应；
* 一次受控 saver 故障只用于验证 503 与 deleting 重试，不伪造成功分支。

最终输出只包含布尔值、计数和耗时，不包含 token、邮箱、UUID、内部 thread key、
数据库连接信息或 checkpoint 内容。
"""

import asyncio
import json
import os
import secrets
import selectors
from pathlib import Path
from time import perf_counter
from uuid import UUID, uuid4

import psycopg
from alembic import command
from alembic.config import Config
from asgi_correlation_id import CorrelationIdMiddleware
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient, Response
from psycopg import sql
from psycopg.conninfo import make_conninfo
from pydantic import SecretStr

from app.api.dependencies import get_token_service
from app.api.v1.auth import router as auth_router
from app.api.v1.chat_sessions import router as chat_sessions_router
from app.core.config import Settings, settings
from app.core.exception_handlers import register_exception_handlers
from app.infrastructure.chat_guard import RedisChatExecutionGuard
from app.infrastructure.database import build_orm_database_url
from app.infrastructure.lifespan import (
    CHAT_GUARD_LEASE_SECONDS,
    get_application_chat_cleanup_service,
    get_application_resources,
    lifespan,
)
from app.services.auth import TokenService
from app.services.chat import ChatService
from app.services.chat_session_cleanup import ChatSessionCleanupService

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONNECTION_TIMEOUT_SECONDS = 10
TOTAL_TIMEOUT_SECONDS = 25.0
TEMPORARY_DATABASE_PREFIX = "deep_research_chat_delete_api_"


class FailingCheckpointStore:
    """确定地抛出一次 saver 故障，用于检查 HTTP 503 边界.

    这不是 fake LLM，也不伪造 checkpoint 删除成功。ORM、Redis guard、HTTP、
    JWT 和后续恢复全部仍使用 production 实现；本对象只代表真实系统中可能发生的
    saver 连接异常，使错误分支能够稳定复现。
    """

    def __init__(self) -> None:
        """初始化可供 smoke 断言的调用计数."""
        self.call_count = 0

    async def adelete_thread(self, thread_id: str) -> None:
        """在接收到内部 key 后抛出不含敏感值的固定故障.

        Args:
            thread_id: cleanup coordinator 生成的内部 key。方法刻意不记录、不返回
                也不拼接这个值。

        Raises:
            RuntimeError: 每次调用都抛出，模拟 saver 当前不可用。
        """
        self.call_count += 1
        raise RuntimeError("injected checkpoint store failure")


def _elapsed_ms(started_at: float) -> float:
    """返回不包含业务数据的单调时钟毫秒数."""
    return round((perf_counter() - started_at) * 1000, 2)


def _conninfo(database: str) -> str:
    """构造只交给 psycopg 使用、绝不写入输出的连接参数."""
    return make_conninfo(
        host=settings.POSTGRES_HOST,
        port=settings.POSTGRES_PORT,
        dbname=database,
        user=settings.POSTGRES_USER,
        password=settings.POSTGRES_PASSWORD,
        connect_timeout=CONNECTION_TIMEOUT_SECONDS,
    )


def _temporary_database_url(database: str) -> str:
    """为 Alembic 构造指向随机临时数据库的 URL."""
    return build_orm_database_url(settings).set(database=database).render_as_string(hide_password=False)


def _create_database(admin_database: str, test_database: str) -> None:
    """在 autocommit 管理连接中创建带安全前缀的随机数据库.

    Args:
        admin_database: 已存在且允许 CREATE DATABASE 的管理库。
        test_database: 本次 smoke 独享的随机数据库名。

    Raises:
        ValueError: 名称不带固定 smoke 前缀时拒绝执行 DDL。
    """
    if not test_database.startswith(TEMPORARY_DATABASE_PREFIX):
        raise ValueError("refusing to create a database without the smoke prefix")
    with psycopg.connect(_conninfo(admin_database), autocommit=True) as connection:
        connection.execute(
            sql.SQL("CREATE DATABASE {}").format(sql.Identifier(test_database)),
        )


def _drop_database(admin_database: str, test_database: str) -> None:
    """终止临时库连接并只删除本次 smoke 创建的数据库."""
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
    """复制真实配置，只替换数据库名并缩小本 smoke 的连接池."""
    config = Settings()
    config.POSTGRES_DB = database
    config.POSTGRES_PSYCOPG_POOL_SIZE = 3
    config.POSTGRES_ORM_POOL_SIZE = 3
    config.POSTGRES_ORM_MAX_OVERFLOW = 0
    return config


def _selector_loop_factory() -> asyncio.AbstractEventLoop:
    """创建 Windows psycopg 异步驱动要求的 Selector event loop."""
    return asyncio.SelectorEventLoop(selectors.SelectSelector())


def _authorization_header(token: str) -> dict[str, str]:
    """构造仅在内存中使用且绝不输出的 Bearer header."""
    return {"Authorization": f"Bearer {token}"}


def _error_signature(response: Response) -> tuple[int, object, object]:
    """提取不包含 request_id 的稳定错误协议部分.

    Args:
        response: 预期包含统一 ErrorResponse JSON 的 HTTP 响应。

    Returns:
        状态码、错误 code 与公开 message。忽略每个请求都不同的 request_id。
    """
    body = response.json()
    error = body.get("error", {})
    return response.status_code, error.get("code"), error.get("message")


async def _register_user(
    client: AsyncClient,
    *,
    email: str,
    password: str,
) -> str:
    """通过正式注册 route 创建用户并返回仅供当前进程使用的 token."""
    response = await client.post(
        "/api/v1/auth/register",
        json={"email": email, "password": password},
    )
    response.raise_for_status()
    token = response.json().get("access_token")
    if not isinstance(token, str) or not token:
        raise RuntimeError("registration response did not contain an access token")
    return token


async def _create_session(
    client: AsyncClient,
    *,
    headers: dict[str, str],
    title: str,
) -> UUID:
    """通过正式 ChatSession route 创建并解析业务会话 UUID."""
    response = await client.post(
        "/api/v1/chat/sessions",
        headers=headers,
        json={"title": title},
    )
    response.raise_for_status()
    return UUID(str(response.json()["thread_id"]))


async def _exercise_delete_api(database: str) -> dict[str, bool | float | int]:
    """使用 production lifespan 验证 DELETE API 的完整外围协议.

    Args:
        database: 已迁移到 Alembic head 的随机数据库。

    Returns:
        只包含布尔检查、计数和耗时的脱敏摘要。
    """
    started_at = perf_counter()
    token_service = TokenService(secret_key=SecretStr(secrets.token_urlsafe(48)))

    app = FastAPI()
    register_exception_handlers(app)
    app.add_middleware(CorrelationIdMiddleware)
    app.include_router(auth_router, prefix="/api/v1")
    app.include_router(chat_sessions_router, prefix="/api/v1")

    # Auth route 和 get_current_user 必须共享同一个随机真实 JWT service。这里只
    # 替换密钥来源，不跳过签名、验签、claims 或数据库用户确认。
    app.dependency_overrides[get_token_service] = lambda: token_service

    async with asyncio.timeout(TOTAL_TIMEOUT_SECONDS):
        async with lifespan(app, config=_runtime_settings(database)):
            resources = get_application_resources(app)
            production_cleanup = get_application_chat_cleanup_service(app)

            transport = ASGITransport(app=app, raise_app_exceptions=False)
            async with AsyncClient(transport=transport, base_url="http://testserver") as client:
                suffix = uuid4().hex
                password = "Real-Delete-API-Smoke-2026!"
                token_a = await _register_user(
                    client,
                    email=f"delete-a-{suffix}@example.com",
                    password=password,
                )
                token_b = await _register_user(
                    client,
                    email=f"delete-b-{suffix}@example.com",
                    password=password,
                )
                headers_a = _authorization_header(token_a)
                headers_b = _authorization_header(token_b)
                user_a = token_service.decode_access_token(token_a).sub

                target_id = await _create_session(
                    client,
                    headers=headers_a,
                    title="Owner deletion target",
                )
                busy_id = await _create_session(
                    client,
                    headers=headers_a,
                    title="Busy deletion target",
                )
                failure_id = await _create_session(
                    client,
                    headers=headers_a,
                    title="Retry deletion target",
                )

                missing_auth = await client.delete(f"/api/v1/chat/sessions/{target_id}")
                cross_user = await client.delete(
                    f"/api/v1/chat/sessions/{target_id}",
                    headers=headers_b,
                )
                absent = await client.delete(
                    f"/api/v1/chat/sessions/{uuid4()}",
                    headers=headers_a,
                )

                # 外部先取得与 production cleanup 相同的 Redis guard key。DELETE
                # 必须快速返回 409，且不能进入 deleting 或 saver 阶段。
                busy_internal_id = ChatService._build_checkpoint_thread_id(user_a, busy_id)
                external_guard = RedisChatExecutionGuard(
                    resources.redis_client,
                    lease_seconds=CHAT_GUARD_LEASE_SECONDS,
                )
                busy_started_at = perf_counter()
                async with external_guard.hold(busy_internal_id):
                    busy = await client.delete(
                        f"/api/v1/chat/sessions/{busy_id}",
                        headers=headers_a,
                    )
                busy_elapsed_ms = _elapsed_ms(busy_started_at)

                # 锁释放后使用同一路由删除，验证前一次 busy 没有污染会话状态。
                busy_retry = await client.delete(
                    f"/api/v1/chat/sessions/{busy_id}",
                    headers=headers_a,
                )

                owner_delete = await client.delete(
                    f"/api/v1/chat/sessions/{target_id}",
                    headers=headers_a,
                )
                repeated = await client.delete(
                    f"/api/v1/chat/sessions/{target_id}",
                    headers=headers_a,
                )

                # 临时把 app.state 指向使用真实 ORM/Redis、但确定失败的 store。
                # route dependency 每次从 app.state 读取，因此该请求会完整经过正式
                # HTTP 和 coordinator，只在 saver 边界抛出故障。
                failing_store = FailingCheckpointStore()
                app.state.chat_session_cleanup_service = ChatSessionCleanupService(
                    session_factory=resources.orm_session_factory,
                    checkpoint_store=failing_store,
                    execution_guard=RedisChatExecutionGuard(
                        resources.redis_client,
                        lease_seconds=CHAT_GUARD_LEASE_SECONDS,
                    ),
                    internal_thread_id_factory=ChatService._build_checkpoint_thread_id,
                )
                cleanup_unavailable = await client.delete(
                    f"/api/v1/chat/sessions/{failure_id}",
                    headers=headers_a,
                )

                # 恢复 lifespan 发布的真实 coordinator。failure_id 已持久化为
                # deleting，重试只能依靠数据库状态继续，而不能依靠故障对象内存。
                app.state.chat_session_cleanup_service = production_cleanup
                failure_retry = await client.delete(
                    f"/api/v1/chat/sessions/{failure_id}",
                    headers=headers_a,
                )

                remaining = await client.get(
                    "/api/v1/chat/sessions",
                    headers=headers_a,
                )
                openapi = (await client.get("/openapi.json")).json()

            delete_operation = openapi["paths"]["/api/v1/chat/sessions/{thread_id}"]["delete"]
            documented_responses = delete_operation["responses"]
            response_204 = documented_responses["204"]

            not_found_signature = (
                404,
                "HTTP_ERROR",
                "Chat session was not found",
            )
            remaining_ids = {item["thread_id"] for item in remaining.json().get("sessions", [])}

            checks: dict[str, bool | float | int] = {
                "missing_auth_is_401": missing_auth.status_code == 401,
                "cross_user_is_hidden_404": _error_signature(cross_user) == not_found_signature,
                "absent_is_same_404": _error_signature(absent) == not_found_signature,
                "busy_is_stable_409": _error_signature(busy)
                == (
                    409,
                    "CHAT_THREAD_BUSY",
                    "Chat thread is already being processed",
                ),
                "busy_is_fail_fast": busy_elapsed_ms < 1000,
                "busy_retry_is_empty_204": (busy_retry.status_code == 204 and busy_retry.content == b""),
                "owner_delete_is_empty_204": (owner_delete.status_code == 204 and owner_delete.content == b""),
                "repeated_delete_is_same_404": _error_signature(repeated) == not_found_signature,
                "cleanup_failure_is_stable_503": _error_signature(cleanup_unavailable)
                == (
                    503,
                    "CHAT_CHECKPOINT_CLEANUP_UNAVAILABLE",
                    "Chat session cleanup is temporarily unavailable",
                ),
                "cleanup_failure_called_store_once": failing_store.call_count == 1,
                "cleanup_retry_is_empty_204": (failure_retry.status_code == 204 and failure_retry.content == b""),
                "deleted_sessions_are_not_listed": all(
                    str(session_id) not in remaining_ids for session_id in (target_id, busy_id, failure_id)
                ),
                "openapi_documents_delete": delete_operation["operationId"].startswith("delete_chat_session"),
                "openapi_documents_statuses": {
                    "204",
                    "401",
                    "404",
                    "409",
                    "503",
                }.issubset(documented_responses),
                "openapi_204_has_no_body": "content" not in response_204,
                "busy_elapsed_ms": busy_elapsed_ms,
            }

        checks["state_removed_after_shutdown"] = (
            not hasattr(app.state, "resources")
            and not hasattr(app.state, "chat_service")
            and not hasattr(app.state, "chat_session_cleanup_service")
        )
        checks["postgres_pool_closed_after_shutdown"] = resources.postgres_pool.closed

    elapsed_ms = _elapsed_ms(started_at)
    checks["within_total_budget"] = elapsed_ms <= TOTAL_TIMEOUT_SECONDS * 1000
    checks["elapsed_ms"] = elapsed_ms
    return checks


def _run_smoke() -> dict[str, object]:
    """创建随机数据库、运行 HTTP Gate，并在 finally 中保证清理."""
    started_at = perf_counter()
    admin_database = settings.POSTGRES_DB
    test_database = f"{TEMPORARY_DATABASE_PREFIX}{uuid4().hex[:10]}"
    previous_override = os.environ.get("ALEMBIC_DATABASE_URL")
    database_created = False
    cleanup_ok = False
    checks: dict[str, bool | float | int] = {}

    try:
        _create_database(admin_database, test_database)
        database_created = True
        os.environ["ALEMBIC_DATABASE_URL"] = _temporary_database_url(test_database)
        command.upgrade(Config(str(PROJECT_ROOT / "alembic.ini")), "head")
        checks = asyncio.run(
            _exercise_delete_api(test_database),
            loop_factory=_selector_loop_factory,
        )
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
        raise RuntimeError("temporary DELETE API database cleanup failed")

    boolean_checks = tuple(value for value in checks.values() if isinstance(value, bool))
    ok = bool(boolean_checks) and all(boolean_checks) and cleanup_ok
    return {
        "ok": ok,
        **checks,
        "cleanup_ok": cleanup_ok,
        "total_elapsed_ms": _elapsed_ms(started_at),
    }


def main() -> int:
    """打印单行脱敏 JSON，并返回 shell 可判断的退出码."""
    started_at = perf_counter()
    try:
        summary = _run_smoke()
    except Exception as error:
        # 基础设施异常可能携带连接信息，因此顶层只公开异常类型。
        print(
            json.dumps(
                {
                    "ok": False,
                    "error_type": type(error).__name__,
                    "elapsed_ms": _elapsed_ms(started_at),
                }
            )
        )
        return 1

    print(json.dumps(summary))
    return 0 if summary["ok"] is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
