"""在真实 PostgreSQL 与文件系统上验收 Lab 17 知识入库闭环.

脚本覆盖正式 migration、multipart API、JWT owner、内容去重、FileStorage、
PostgreSQL worker claim、成功/失败/重试、running 删除冲突和最终清理。Lab 18
尚未实现 parser，因此只注入确定性 IndexProcessor；数据库、事务、HTTP 和文件系统
均使用生产实现。最终摘要不输出 token、用户 ID、文档 ID、hash、storage key 或正文。
"""

import asyncio
import json
import os
import secrets
import selectors
from collections.abc import AsyncIterator
from pathlib import Path
from tempfile import TemporaryDirectory
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

from app.api.dependencies import (
    enforce_auth_rate_limit,
    get_db_session,
    get_token_service,
)
from app.api.v1.auth import router as auth_router
from app.api.v1.knowledge import router as knowledge_router
from app.core.config import Settings, settings
from app.core.exception_handlers import register_exception_handlers
from app.infrastructure.database import build_orm_database_url, create_orm_runtime
from app.infrastructure.file_storage import LocalFileStorage
from app.models import DocumentStatus
from app.models.base import utc_now
from app.repositories import DocumentRepository, IndexJobRepository
from app.services.auth import TokenService
from app.services.index_worker import (
    IndexProcessingError,
    IndexSource,
    IndexWorker,
    IndexWorkerResult,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEMPORARY_DATABASE_PREFIX = "deep_research_knowledge_"
CONNECTION_TIMEOUT_SECONDS = 10
TOTAL_TIMEOUT_SECONDS = 90.0


def _elapsed_ms(started_at: float) -> float:
    """返回不含请求或基础设施内容的毫秒耗时."""
    return round((perf_counter() - started_at) * 1000, 2)


def _conninfo(database: str) -> str:
    """构造只交给 psycopg 使用且不会输出的连接参数."""
    return make_conninfo(
        host=settings.POSTGRES_HOST,
        port=settings.POSTGRES_PORT,
        dbname=database,
        user=settings.POSTGRES_USER,
        password=settings.POSTGRES_PASSWORD,
        connect_timeout=CONNECTION_TIMEOUT_SECONDS,
    )


def _temporary_database_url(database: str) -> str:
    """构造 Alembic 随机数据库 URL，调用方不得记录返回值."""
    return build_orm_database_url(settings).set(database=database).render_as_string(hide_password=False)


def _create_database(admin_database: str, test_database: str) -> None:
    """只允许创建带固定 smoke 前缀的随机数据库."""
    if not test_database.startswith(TEMPORARY_DATABASE_PREFIX):
        raise ValueError("refusing to create a database without the smoke prefix")
    with psycopg.connect(_conninfo(admin_database), autocommit=True) as connection:
        connection.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(test_database)))


def _drop_database(admin_database: str, test_database: str) -> None:
    """终止残留连接并只删除本脚本创建的随机数据库."""
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
        connection.execute(sql.SQL("DROP DATABASE IF EXISTS {}").format(sql.Identifier(test_database)))


def _runtime_settings(database: str) -> Settings:
    """复制真实基础设施设置，只切换随机数据库和连接池预算."""
    config = Settings()
    config.POSTGRES_DB = database
    config.POSTGRES_ORM_POOL_SIZE = 5
    config.POSTGRES_ORM_MAX_OVERFLOW = 0
    return config


def _selector_loop_factory() -> asyncio.AbstractEventLoop:
    """创建 Windows psycopg 异步驱动需要的 Selector event loop."""
    return asyncio.SelectorEventLoop(selectors.SelectSelector())


class _SuccessfulProcessor:
    """Lab 17 的边界处理器：验证输入后正常返回，不实现 Lab 18 解析."""

    def __init__(self) -> None:
        self.calls = 0

    async def process(self, source: IndexSource) -> None:
        """确认 worker 取得真实文件字节和可信元数据."""
        if not source.content or source.content_type != "text/plain":
            raise IndexProcessingError("INVALID_SMOKE_SOURCE")
        self.calls += 1


class _FailingProcessor:
    """返回稳定业务错误码，用于验证失败状态与人工重试."""

    async def process(self, source: IndexSource) -> None:
        """不记录正文，只产生可安全持久化的错误码."""
        if not source.content:
            raise RuntimeError("smoke source unexpectedly empty")
        raise IndexProcessingError("PARSER_NOT_READY")


async def _exercise(database: str, storage_root: Path) -> dict[str, bool | int | float]:
    """执行 HTTP、数据库 worker 与文件清理组合验收."""
    started_at = perf_counter()
    engine, session_factory = create_orm_runtime(_runtime_settings(database))
    storage = LocalFileStorage(storage_root)
    token_service = TokenService(secret_key=SecretStr(secrets.token_urlsafe(48)))

    app = FastAPI()
    register_exception_handlers(app)
    app.add_middleware(CorrelationIdMiddleware)
    app.include_router(auth_router, prefix="/api/v1")
    app.include_router(knowledge_router, prefix="/api/v1")
    app.state.file_storage = storage

    async def override_db_session() -> AsyncIterator[AsyncSession]:
        """为每个 smoke HTTP 请求创建独立真实 Session."""
        async with session_factory() as session:
            yield session

    def override_token_service() -> TokenService:
        """让注册与后续认证共享本进程随机 JWT secret."""
        return token_service

    async def bypass_auth_rate_limit() -> None:
        """本 smoke 不重复验证 Lab 15 Redis 限流，只保留真实认证和数据库链."""

    app.dependency_overrides[get_db_session] = override_db_session
    app.dependency_overrides[get_token_service] = override_token_service
    app.dependency_overrides[enforce_auth_rate_limit] = bypass_auth_rate_limit

    # 缩小本脚本上传上限以低成本覆盖 413；finally 会恢复全局设置。
    previous_max_upload = settings.KNOWLEDGE_MAX_UPLOAD_BYTES
    settings.KNOWLEDGE_MAX_UPLOAD_BYTES = 64
    suffix = uuid4().hex
    password = "Knowledge-Smoke-Password-2026!"

    try:
        transport = ASGITransport(app=app, raise_app_exceptions=False)
        async with AsyncClient(transport=transport, base_url="http://testserver") as client:
            register_a = await client.post(
                "/api/v1/auth/register",
                json={"email": f"knowledge-a-{suffix}@example.com", "password": password},
            )
            register_b = await client.post(
                "/api/v1/auth/register",
                json={"email": f"knowledge-b-{suffix}@example.com", "password": password},
            )
            token_a = str(register_a.json().get("access_token", ""))
            token_b = str(register_b.json().get("access_token", ""))
            user_a = token_service.decode_access_token(token_a).sub
            headers_a = {"Authorization": f"Bearer {token_a}"}
            headers_b = {"Authorization": f"Bearer {token_b}"}

            content = b"owner-scoped knowledge"
            upload_a = await client.post(
                "/api/v1/knowledge/documents",
                headers=headers_a,
                files={"file": ("../../notes.txt", content, "text/plain")},
            )
            upload_a_body = upload_a.json()
            document_a = UUID(str(upload_a_body["document"]["document_id"]))
            duplicate_a = await client.post(
                "/api/v1/knowledge/documents",
                headers=headers_a,
                files={"file": ("renamed.txt", content, "text/plain")},
            )
            cross_user = await client.get(
                f"/api/v1/knowledge/documents/{document_a}",
                headers=headers_b,
            )
            list_a = await client.get("/api/v1/knowledge/documents", headers=headers_a)
            invalid_type = await client.post(
                "/api/v1/knowledge/documents",
                headers=headers_a,
                files={"file": ("image.png", b"not-an-image", "image/png")},
            )
            empty = await client.post(
                "/api/v1/knowledge/documents",
                headers=headers_a,
                files={"file": ("empty.txt", b"", "text/plain")},
            )
            too_large = await client.post(
                "/api/v1/knowledge/documents",
                headers=headers_a,
                files={"file": ("large.txt", b"x" * 65, "text/plain")},
            )

            # 两个真实 worker 同时轮询同一 pending job。FOR UPDATE SKIP LOCKED
            # 与已提交 running 状态应保证 processor 总共只执行一次。
            successful_processor = _SuccessfulProcessor()
            worker_first = IndexWorker(
                session_factory=session_factory,
                storage=storage,
                processor=successful_processor,
                worker_id="knowledge-smoke-a",
                lease_seconds=30,
            )
            worker_second = IndexWorker(
                session_factory=session_factory,
                storage=storage,
                processor=successful_processor,
                worker_id="knowledge-smoke-b",
                lease_seconds=30,
            )
            worker_results = await asyncio.gather(
                worker_first.run_once(),
                worker_second.run_once(),
            )
            single_claim_calls = successful_processor.calls
            ready_a = await client.get(
                f"/api/v1/knowledge/documents/{document_a}",
                headers=headers_a,
            )

            # 相同内容在另一个 owner namespace 中应创建独立 Document/IndexJob，
            # 而不是复用 A 的记录或泄漏 A 已经上传过。随后先完成 B 的 pending job，
            # 避免它干扰下面针对失败重试的单任务场景。
            upload_b = await client.post(
                "/api/v1/knowledge/documents",
                headers=headers_b,
                files={"file": ("notes.txt", content, "text/plain")},
            )
            owner_b_result = await worker_first.run_once()
            list_b = await client.get("/api/v1/knowledge/documents", headers=headers_b)

            failed_upload = await client.post(
                "/api/v1/knowledge/documents",
                headers=headers_a,
                files={"file": ("failure.txt", b"failure-source", "text/plain")},
            )
            failed_document_id = UUID(str(failed_upload.json()["document"]["document_id"]))
            failing_worker = IndexWorker(
                session_factory=session_factory,
                storage=storage,
                processor=_FailingProcessor(),
                worker_id="knowledge-smoke-failure",
                lease_seconds=30,
            )
            failed_result = await failing_worker.run_once()
            failed_state = await client.get(
                f"/api/v1/knowledge/documents/{failed_document_id}",
                headers=headers_a,
            )
            retry = await client.post(
                f"/api/v1/knowledge/documents/{failed_document_id}/retry",
                headers=headers_a,
            )
            retried_result = await worker_first.run_once()

            busy_upload = await client.post(
                "/api/v1/knowledge/documents",
                headers=headers_a,
                files={"file": ("busy.txt", b"busy-source", "text/plain")},
            )
            busy_document_id = UUID(str(busy_upload.json()["document"]["document_id"]))

            # 手工执行 production Repository claim，模拟正在工作的另一个进程。
            async with session_factory() as session:
                async with session.begin():
                    job = await IndexJobRepository(session).claim_next(
                        worker_id="knowledge-smoke-holder",
                        now=utc_now(),
                        lease_seconds=30,
                    )
                    if job is None or job.document_id != busy_document_id:
                        raise RuntimeError("expected busy smoke job")
                    document = await DocumentRepository(session).get_internal_for_update(job.document_id)
                    if document is None:
                        raise RuntimeError("expected busy smoke document")
                    await DocumentRepository(session).set_status(document, status=DocumentStatus.INDEXING)

            busy_delete = await client.delete(
                f"/api/v1/knowledge/documents/{busy_document_id}",
                headers=headers_a,
            )

            # 只把租约改为过去，模拟 worker 崩溃。删除服务随后应允许清理，不需要
            # 访问或信任旧 worker 的进程内状态。
            async with session_factory() as session:
                async with session.begin():
                    job = await IndexJobRepository(session).get_for_update(document_id=busy_document_id)
                    if job is None:
                        raise RuntimeError("expected claimed busy job")
                    job.lease_expires_at = utc_now().replace(year=2025)
                    session.add(job)

            expired_delete = await client.delete(
                f"/api/v1/knowledge/documents/{busy_document_id}",
                headers=headers_a,
            )
            delete_ready = await client.delete(
                f"/api/v1/knowledge/documents/{document_a}",
                headers=headers_a,
            )
            read_deleted = await client.get(
                f"/api/v1/knowledge/documents/{document_a}",
                headers=headers_a,
            )
            openapi = (await client.get("/openapi.json")).json()

        async with session_factory() as session:
            async with session.begin():
                remaining_a = await DocumentRepository(session).list_by_user(user_a)

        ready_body = ready_a.json()
        failed_body = failed_state.json()
        retry_body = retry.json()
        completed_count = sum(result == IndexWorkerResult.COMPLETED for result in worker_results)
        idle_count = sum(result == IndexWorkerResult.IDLE for result in worker_results)

        return {
            "upload_created": upload_a.status_code == 201 and upload_a_body["deduplicated"] is False,
            "filename_sanitized": upload_a_body["document"]["original_filename"] == "notes.txt",
            "internal_storage_fields_hidden": all(
                field not in json.dumps(upload_a_body)
                for field in ("storage_key", "content_sha256", "claim_token", "claimed_by")
            ),
            "owner_duplicate_reused": duplicate_a.status_code == 201
            and duplicate_a.json()["deduplicated"] is True
            and duplicate_a.json()["document"]["document_id"] == str(document_a),
            "cross_owner_same_content_isolated": upload_b.status_code == 201
            and upload_b.json()["deduplicated"] is False
            and owner_b_result == IndexWorkerResult.COMPLETED,
            "cross_user_hidden": cross_user.status_code == 404,
            "lists_are_owner_scoped": len(list_a.json()["documents"]) == 1 and len(list_b.json()["documents"]) == 1,
            "unsupported_type_rejected": invalid_type.status_code == 415,
            "empty_rejected": empty.status_code == 422,
            "oversize_rejected": too_large.status_code == 413,
            "single_worker_claimed": completed_count == 1 and idle_count == 1 and single_claim_calls == 1,
            "worker_completed_document": ready_body["status"] == "ready"
            and ready_body["index_job"]["status"] == "completed",
            "failure_is_stable": failed_result == IndexWorkerResult.FAILED
            and failed_body["status"] == "failed"
            and failed_body["failure_code"] == "PARSER_NOT_READY",
            "retry_completed": retry.status_code == 200
            and retry_body["status"] == "pending"
            and retried_result == IndexWorkerResult.COMPLETED,
            "active_lease_blocks_delete": busy_delete.status_code == 409,
            "expired_lease_allows_delete": expired_delete.status_code == 204,
            "ready_delete_completed": delete_ready.status_code == 204 and read_deleted.status_code == 404,
            "remaining_owner_documents_match": len(remaining_a) == 1,
            "openapi_registered": "/api/v1/knowledge/documents" in openapi["paths"],
            "within_total_budget": _elapsed_ms(started_at) <= TOTAL_TIMEOUT_SECONDS * 1000,
            "elapsed_ms": _elapsed_ms(started_at),
        }
    finally:
        settings.KNOWLEDGE_MAX_UPLOAD_BYTES = previous_max_upload
        await engine.dispose()


def _run_smoke() -> dict[str, object]:
    """迁移随机数据库、运行组合验收并无条件清理数据库和文件目录."""
    started_at = perf_counter()
    admin_database = settings.POSTGRES_DB
    test_database = f"{TEMPORARY_DATABASE_PREFIX}{uuid4().hex[:10]}"
    previous_override = os.environ.get("ALEMBIC_DATABASE_URL")
    database_created = False
    database_cleanup_ok = False

    with TemporaryDirectory(prefix="deep-research-knowledge-") as temporary_directory:
        try:
            _create_database(admin_database, test_database)
            database_created = True
            os.environ["ALEMBIC_DATABASE_URL"] = _temporary_database_url(test_database)
            command.upgrade(Config(str(PROJECT_ROOT / "alembic.ini")), "head")
            checks = asyncio.run(
                _exercise(test_database, Path(temporary_directory)),
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
                    database_cleanup_ok = False
                else:
                    database_cleanup_ok = True

    if database_created and not database_cleanup_ok:
        raise RuntimeError("temporary knowledge database cleanup failed")
    boolean_checks = tuple(value for value in checks.values() if isinstance(value, bool))
    return {
        "ok": bool(boolean_checks) and all(boolean_checks) and database_cleanup_ok,
        **checks,
        "database_cleanup_ok": database_cleanup_ok,
        "total_elapsed_ms": _elapsed_ms(started_at),
    }


def main() -> int:
    """打印安全单行 JSON，并返回 shell 友好的退出码."""
    started_at = perf_counter()
    try:
        summary = _run_smoke()
    except Exception as error:
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
