"""可跨进程领取并收敛文档索引任务的 worker 边界."""

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.logging import logger
from app.models import DocumentStatus
from app.models.base import utc_now
from app.repositories import DocumentRepository, IndexJobRepository
from app.services.file_storage import FileStorage, FileStorageError


@dataclass(frozen=True, slots=True)
class IndexSource:
    """从可信数据库元数据和 FileStorage 组合出的 worker 输入."""

    document_id: UUID
    user_id: UUID
    original_filename: str
    content_type: str
    content_sha256: str
    content: bytes


class IndexProcessor(Protocol):
    """Lab 18/19 解析、分块和 embedding 管线需要实现的接口."""

    async def process(self, source: IndexSource) -> None:
        """处理一份原始文档；正常返回表示本次索引全部完成."""
        ...


class IndexProcessingError(RuntimeError):
    """处理器提供稳定错误码的可预期失败."""

    def __init__(self, error_code: str) -> None:
        """保存不含文档正文、路径或 provider body 的稳定错误码."""
        self.error_code = error_code
        super().__init__("Document indexing failed")


class IndexWorkerResult(StrEnum):
    """一次 worker 轮询的安全结果摘要."""

    IDLE = "idle"
    COMPLETED = "completed"
    FAILED = "failed"
    STALE = "stale"


class IndexWorker:
    """用 PostgreSQL 租约领取任务，并把处理结果安全收敛回数据库."""

    def __init__(
        self,
        *,
        session_factory: async_sessionmaker[AsyncSession],
        storage: FileStorage,
        processor: IndexProcessor,
        worker_id: str,
        lease_seconds: float,
    ) -> None:
        """保存可跨轮询复用的无请求状态依赖.

        Args:
            session_factory: lifespan 创建的 ORM Session 工厂；每个领取/收尾阶段
                都创建独立 Session，处理器执行期间不长期占用数据库连接。
            storage: 原始文件存储协议。
            processor: Lab 18/19 提供的实际索引管线。
            worker_id: 当前 worker 实例的内部标识，只写数据库 claim，不进公开响应。
            lease_seconds: worker 崩溃后允许其他实例接管的最长等待时间。
        """
        if not worker_id.strip():
            raise ValueError("worker_id must not be empty")
        if lease_seconds <= 0:
            raise ValueError("lease_seconds must be greater than 0")
        self._session_factory = session_factory
        self._storage = storage
        self._processor = processor
        self._worker_id = worker_id
        self._lease_seconds = lease_seconds

    async def run_once(self) -> IndexWorkerResult:
        """领取并处理至多一个 job，适合 CLI、队列 consumer 或定时轮询调用.

        Returns:
            idle 表示当前没有可领取工作；completed/failed 表示本次 claim 已按
            当前 token 收敛；stale 表示租约被接管或文档进入删除流程，旧 worker
            的结果被 fencing token 拒绝。
        """
        # 阶段 1：短事务领取任务并把 Document 标记 indexing。事务提交以后，
        # 即使进程立即崩溃，running + lease 仍是其他 worker 的恢复坐标。
        async with self._session_factory() as session:
            async with session.begin():
                jobs = IndexJobRepository(session)
                documents = DocumentRepository(session)
                job = await jobs.claim_next(
                    worker_id=self._worker_id,
                    now=utc_now(),
                    lease_seconds=self._lease_seconds,
                )
                if job is None:
                    return IndexWorkerResult.IDLE
                document = await documents.get_internal_for_update(job.document_id)
                if document is None or document.status == DocumentStatus.DELETING:
                    return IndexWorkerResult.STALE
                await documents.set_status(document, status=DocumentStatus.INDEXING)

                job_id = job.id
                claim_token = job.claim_token
                storage_key = document.storage_key
                source_metadata = (
                    document.id,
                    document.user_id,
                    document.original_filename,
                    document.content_type,
                    document.content_sha256,
                )

        if claim_token is None:
            raise RuntimeError("Claimed index job is missing its fencing token")

        # 阶段 2：事务外读取文件并运行耗时处理器。这样 parse/embedding 不会占用
        # ORM 连接或行锁；并发正确性由 claim token 与 lease 负责。
        try:
            content = await self._storage.read(storage_key)
            source = IndexSource(
                document_id=source_metadata[0],
                user_id=source_metadata[1],
                original_filename=source_metadata[2],
                content_type=source_metadata[3],
                content_sha256=source_metadata[4],
                content=content,
            )
            await self._processor.process(source)
        except FileStorageError as error:
            logger.warning(
                "knowledge_index_failed",
                stage="storage_read",
                error_type=type(error).__name__,
            )
            return await self._finish_failed(
                job_id=job_id,
                claim_token=claim_token,
                error_code="SOURCE_FILE_UNAVAILABLE",
            )
        except IndexProcessingError as error:
            logger.warning(
                "knowledge_index_failed",
                stage="processor",
                error_code=error.error_code,
            )
            return await self._finish_failed(
                job_id=job_id,
                claim_token=claim_token,
                error_code=error.error_code,
            )
        except Exception as error:
            # 不记录 str(error) 或 source 内容。第三方 parser/provider 异常可能
            # 包含文件片段、URL 或响应 body，服务端只需要异常类型定位类别。
            logger.exception(
                "knowledge_index_failed",
                stage="processor",
                error_type=type(error).__name__,
            )
            return await self._finish_failed(
                job_id=job_id,
                claim_token=claim_token,
                error_code="INDEX_PROCESSING_FAILED",
            )

        # 阶段 3：只有仍持有同一 fencing token 的 worker 可以写 completed/ready。
        now = utc_now()
        async with self._session_factory() as session:
            async with session.begin():
                jobs = IndexJobRepository(session)
                documents = DocumentRepository(session)
                claimed_job = await jobs.get_claim_for_update(
                    job_id=job_id,
                    claim_token=claim_token,
                )
                if claimed_job is None:
                    return IndexWorkerResult.STALE
                document = await documents.get_internal_for_update(claimed_job.document_id)
                if document is None or document.status == DocumentStatus.DELETING:
                    return IndexWorkerResult.STALE
                await jobs.mark_completed(claimed_job, now=now)
                await documents.set_status(
                    document,
                    status=DocumentStatus.READY,
                    indexed_at=now,
                )
        return IndexWorkerResult.COMPLETED

    async def _finish_failed(
        self,
        *,
        job_id: UUID,
        claim_token: UUID,
        error_code: str,
    ) -> IndexWorkerResult:
        """按 fencing token 写入失败；过期 worker 的错误不能覆盖新结果."""
        now = utc_now()
        async with self._session_factory() as session:
            async with session.begin():
                jobs = IndexJobRepository(session)
                documents = DocumentRepository(session)
                claimed_job = await jobs.get_claim_for_update(
                    job_id=job_id,
                    claim_token=claim_token,
                )
                if claimed_job is None:
                    return IndexWorkerResult.STALE
                document = await documents.get_internal_for_update(claimed_job.document_id)
                if document is None or document.status == DocumentStatus.DELETING:
                    return IndexWorkerResult.STALE
                await jobs.mark_failed(claimed_job, now=now, error_code=error_code)
                await documents.set_status(
                    document,
                    status=DocumentStatus.FAILED,
                    failure_code=error_code,
                )
        return IndexWorkerResult.FAILED


__all__ = [
    "IndexProcessingError",
    "IndexProcessor",
    "IndexSource",
    "IndexWorker",
    "IndexWorkerResult",
]
