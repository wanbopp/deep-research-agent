"""知识文档上传、查询、重试与删除的应用服务."""

from hashlib import sha256
from pathlib import PurePosixPath
from typing import Protocol
from uuid import UUID, uuid4

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import logger
from app.models import Document, DocumentStatus, IndexJob, IndexJobStatus
from app.models.base import utc_now
from app.repositories import (
    DocumentRepository,
    IndexJobRepository,
    RepositoryConflictError,
)
from app.schemas.knowledge import (
    DocumentListResponse,
    DocumentResponse,
    DocumentUploadResponse,
    IndexJobResponse,
)
from app.services.file_storage import FileStorage, FileStorageError


class AsyncReadable(Protocol):
    """KnowledgeService 读取上传正文所需的最小异步接口."""

    async def read(self, size: int = -1) -> bytes:
        """读取至多 size 字节；到达结尾时返回空 bytes."""
        ...


class KnowledgeServiceError(RuntimeError):
    """知识文档可预期业务错误的基类."""


class KnowledgeDocumentNotFoundError(KnowledgeServiceError):
    """文档不存在或不属于当前用户."""


class KnowledgeDocumentBusyError(KnowledgeServiceError):
    """文档正在索引或删除，当前操作不能安全执行."""


class KnowledgeDocumentNotRetryableError(KnowledgeServiceError):
    """文档没有处于允许人工重试的失败状态."""


class KnowledgeUnsupportedMediaTypeError(KnowledgeServiceError):
    """上传 MIME 不在服务端 allowlist 中."""


class KnowledgeDocumentTooLargeError(KnowledgeServiceError):
    """上传正文超过配置的最大字节数."""


class KnowledgeEmptyDocumentError(KnowledgeServiceError):
    """上传正文为空."""


class KnowledgeInvalidFilenameError(KnowledgeServiceError):
    """文件名为空、过长或无法安全规范化."""


class KnowledgeStorageUnavailableError(KnowledgeServiceError):
    """原始文件存储无法完成当前操作."""


class KnowledgeStateError(RuntimeError):
    """数据库出现不符合 Document/IndexJob 组合不变量的状态."""


class KnowledgeService:
    """协调请求级数据库事务和共享 FileStorage.

    Service 保存的是当前请求独享的 ``AsyncSession``。它可以在一个请求内顺序
    开启多个短事务，但不能跨请求共享，也不能交给并发 worker 使用。
    """

    def __init__(
        self,
        session: AsyncSession,
        *,
        storage: FileStorage,
        allowed_content_types: frozenset[str],
        max_upload_bytes: int,
        read_chunk_bytes: int,
        index_max_attempts: int,
    ) -> None:
        """保存上传策略、请求级事务和文件存储能力.

        Args:
            session: 当前 HTTP 请求独享的 ORM Session。
            storage: lifespan 共享或无状态的 FileStorage adapter。
            allowed_content_types: 规范化小写 MIME allowlist。
            max_upload_bytes: 单文件最大字节数；service 在读取时立即执行限制。
            read_chunk_bytes: 从 UploadFile 等异步流单次读取的字节数。
            index_max_attempts: 新 IndexJob 允许的最大领取次数。
        """
        if not allowed_content_types:
            raise ValueError("allowed_content_types must not be empty")
        for name, value in (
            ("max_upload_bytes", max_upload_bytes),
            ("read_chunk_bytes", read_chunk_bytes),
            ("index_max_attempts", index_max_attempts),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be greater than 0")

        self._session = session
        self._storage = storage
        self._allowed_content_types = allowed_content_types
        self._max_upload_bytes = max_upload_bytes
        self._read_chunk_bytes = read_chunk_bytes
        self._index_max_attempts = index_max_attempts

    async def upload(
        self,
        *,
        user_id: UUID,
        filename: str | None,
        content_type: str | None,
        source: AsyncReadable,
    ) -> DocumentUploadResponse:
        """保存原始文件，并在一个数据库事务中创建 Document 与 IndexJob.

        Args:
            user_id: 认证 dependency 提供的可信用户 UUID。
            filename: multipart 元数据中的展示文件名；只保存规范化 basename。
            content_type: multipart MIME；规范化后必须命中服务端 allowlist。
                参数来源于 multipart 请求中该文件 part 声明的 MIME 类型；经过规范化之后，必须命中服务端允许的类型白名单（allowlist），否则会抛出 KnowledgeUnsupportedMediaTypeError。
            source: UploadFile 或其他实现异步 read 的有界输入流。

        Returns:
            新建或 owner 内幂等复用的文档与索引任务。

        Raises:
            KnowledgeUnsupportedMediaTypeError: MIME 不受支持。
            KnowledgeDocumentTooLargeError: 正文超过最大字节数。
            KnowledgeEmptyDocumentError: 正文为空。
            KnowledgeInvalidFilenameError: 文件名不安全或不符合长度限制。
            KnowledgeStorageUnavailableError: 原始文件无法安全保存或补偿清理。
            KnowledgeDocumentBusyError: 相同内容正在删除，不能建立重复记录。
        """
        safe_filename = self._normalize_filename(filename)
        normalized_content_type = self._normalize_content_type(content_type)
        content, content_sha256 = await self._read_bounded(source)

        # 第一次查询避免普通重复上传产生无用文件。数据库唯一约束仍是并发竞态的
        # 最终防线，因此后面还会处理两个请求同时通过这里的情况。
        async with self._session.begin():
            existing = await DocumentRepository(self._session).get_by_content_hash(
                user_id=user_id,
                content_sha256=content_sha256,
                include_deleting=True,
            )
            if existing is not None:
                if existing.status == DocumentStatus.DELETING:
                    raise KnowledgeDocumentBusyError
                job = await IndexJobRepository(self._session).get_by_document_id(
                    existing.id,
                    user_id=user_id,
                )
                return DocumentUploadResponse(
                    document=self._to_document_response(existing, self._require_job(job)),
                    deduplicated=True,
                )

        storage_key = self._build_storage_key(user_id=user_id)
        try:
            await self._storage.put(storage_key, content)
        except FileStorageError as error:
            raise KnowledgeStorageUnavailableError from error

        try:
            async with self._session.begin():
                documents = DocumentRepository(self._session)
                jobs = IndexJobRepository(self._session)
                document = await documents.create(
                    user_id=user_id,
                    original_filename=safe_filename,
                    content_type=normalized_content_type,
                    size_bytes=len(content),
                    content_sha256=content_sha256,
                    storage_key=storage_key,
                )
                job = await jobs.create(
                    document_id=document.id,
                    max_attempts=self._index_max_attempts,
                )
        except RepositoryConflictError:
            # 两个并发上传可能都先得到 miss 并分别写文件，随后只有一个能通过
            # owner+hash 唯一约束。失败者先删除自己的随机对象，再读取胜出记录。
            await self._compensate_storage(storage_key)
            async with self._session.begin():
                existing = await DocumentRepository(self._session).get_by_content_hash(
                    user_id=user_id,
                    content_sha256=content_sha256,
                    include_deleting=True,
                )
                if existing is None:
                    raise
                if existing.status == DocumentStatus.DELETING:
                    raise KnowledgeDocumentBusyError
                existing_job = await IndexJobRepository(self._session).get_by_document_id(
                    existing.id,
                    user_id=user_id,
                )
                return DocumentUploadResponse(
                    document=self._to_document_response(existing, self._require_job(existing_job)),
                    deduplicated=True,
                )
        except Exception:
            await self._compensate_storage(storage_key)
            raise

        return DocumentUploadResponse(
            document=self._to_document_response(document, job),
            deduplicated=False,
        )

    async def list_owned(self, *, user_id: UUID) -> DocumentListResponse:
        """列出当前用户文档，并用一次批量查询加载各自 IndexJob."""
        async with self._session.begin():
            documents = await DocumentRepository(self._session).list_by_user(user_id)
            jobs = await IndexJobRepository(self._session).list_by_document_ids(
                tuple(document.id for document in documents)
            )

        jobs_by_document = {job.document_id: job for job in jobs}
        return DocumentListResponse(
            documents=tuple(
                self._to_document_response(
                    document,
                    self._require_job(jobs_by_document.get(document.id)),
                )
                for document in documents
            )
        )

    async def get_owned(self, *, document_id: UUID, user_id: UUID) -> DocumentResponse:
        """按资源 ID 和可信 owner 查询文档；跨用户与不存在使用相同错误."""
        async with self._session.begin():
            document = await DocumentRepository(self._session).get_by_id(document_id, user_id=user_id)
            if document is None:
                raise KnowledgeDocumentNotFoundError
            job = await IndexJobRepository(self._session).get_by_document_id(document.id, user_id=user_id)
        return self._to_document_response(document, self._require_job(job))

    async def retry_owned(self, *, document_id: UUID, user_id: UUID) -> DocumentResponse:
        """把未耗尽次数的 failed job 放回 pending 队列."""
        async with self._session.begin():
            documents = DocumentRepository(self._session)
            jobs = IndexJobRepository(self._session)
            if (
                await documents.get_by_id(
                    document_id,
                    user_id=user_id,
                    include_deleting=True,
                )
                is None
            ):
                raise KnowledgeDocumentNotFoundError
            # 全系统统一按 IndexJob -> Document 加锁；worker claim/finalize 和删除
            # 采用相同顺序，避免两个事务各持有一把锁并等待对方。
            job = self._require_job(await jobs.get_for_update(document_id=document_id))
            document = await documents.get_for_update(document_id, user_id=user_id)
            if document is None:
                raise KnowledgeDocumentNotFoundError
            if (
                document.status != DocumentStatus.FAILED
                or job.status != IndexJobStatus.FAILED
                or job.attempt_count >= job.max_attempts
            ):
                raise KnowledgeDocumentNotRetryableError

            await jobs.reset_failed(job)
            await documents.set_status(document, status=DocumentStatus.PENDING)

        return self._to_document_response(document, job)

    async def delete_owned(self, *, document_id: UUID, user_id: UUID) -> None:
        """以持久 deleting tombstone 协调数据库与文件系统删除.

        第一次事务提交删除意图；文件删除失败时 tombstone 保留，客户端稍后重试。
        文件删除成功后第二次事务物理删除 Document，IndexJob 由外键级联删除。
        """
        async with self._session.begin():
            documents = DocumentRepository(self._session)
            jobs = IndexJobRepository(self._session)
            if (
                await documents.get_by_id(
                    document_id,
                    user_id=user_id,
                    include_deleting=True,
                )
                is None
            ):
                raise KnowledgeDocumentNotFoundError
            job = self._require_job(await jobs.get_for_update(document_id=document_id))
            document = await documents.get_for_update(document_id, user_id=user_id)
            if document is None:
                raise KnowledgeDocumentNotFoundError
            if job.status == IndexJobStatus.RUNNING and job.lease_expires_at is not None:
                if job.lease_expires_at > utc_now():
                    raise KnowledgeDocumentBusyError
            storage_key = document.storage_key
            if document.status != DocumentStatus.DELETING:
                await documents.mark_deleting(document)

        try:
            await self._storage.delete(storage_key)
        except FileStorageError as error:
            raise KnowledgeStorageUnavailableError from error

        async with self._session.begin():
            documents = DocumentRepository(self._session)
            document = await documents.get_for_update(document_id, user_id=user_id)
            if document is None:
                return
            if document.status != DocumentStatus.DELETING:
                raise KnowledgeStateError("Document must be deleting before final cleanup")
            await documents.delete(document)

    async def _read_bounded(self, source: AsyncReadable) -> tuple[bytes, str]:
        """分块读取上传流，同时执行大小限制和 SHA-256 计算."""
        chunks: list[bytes] = []
        digest = sha256()
        total_bytes = 0
        while True:
            chunk = await source.read(self._read_chunk_bytes)
            if not chunk:
                break
            total_bytes += len(chunk)
            if total_bytes > self._max_upload_bytes:
                raise KnowledgeDocumentTooLargeError
            digest.update(chunk)
            chunks.append(chunk)

        if total_bytes == 0:
            raise KnowledgeEmptyDocumentError
        return b"".join(chunks), digest.hexdigest()

    def _normalize_content_type(self, content_type: str | None) -> str:
        """去除 MIME 参数并按小写匹配 allowlist."""
        normalized = (content_type or "").split(";", maxsplit=1)[0].strip().lower()
        if normalized not in self._allowed_content_types:
            raise KnowledgeUnsupportedMediaTypeError
        return normalized

    @staticmethod
    def _normalize_filename(filename: str | None) -> str:
        """只保留展示 basename；绝不把用户文件名用作 storage key."""
        normalized = PurePosixPath((filename or "").replace("\\", "/")).name.strip()
        if not normalized or normalized in {".", ".."} or "\x00" in normalized or len(normalized) > 255:
            raise KnowledgeInvalidFilenameError
        return normalized

    @staticmethod
    def _build_storage_key(*, user_id: UUID) -> str:
        """用可信 owner 与随机对象 ID 构造不可由文件名影响的内部 key."""
        return f"documents/{user_id.hex}/{uuid4().hex}"

    async def _compensate_storage(self, storage_key: str) -> None:
        """数据库事务失败后尽力删除本次创建的原始对象."""
        try:
            await self._storage.delete(storage_key)
        except FileStorageError as error:
            logger.error(
                "knowledge_upload_compensation_failed",
                error_type=type(error).__name__,
            )
            raise KnowledgeStorageUnavailableError from error

    @staticmethod
    def _require_job(job: IndexJob | None) -> IndexJob:
        """确保每个可见 Document 都满足“一份当前 job”的数据库不变量."""
        if job is None:
            raise KnowledgeStateError("Document is missing its index job")
        return job

    @staticmethod
    def _to_document_response(document: Document, job: IndexJob) -> DocumentResponse:
        """移除 owner、hash、storage key 和 claim 等内部字段后构造公开模型."""
        retryable = job.status == IndexJobStatus.FAILED and job.attempt_count < job.max_attempts
        return DocumentResponse(
            document_id=document.id,
            original_filename=document.original_filename,
            content_type=document.content_type,
            size_bytes=document.size_bytes,
            status=document.status,
            failure_code=document.failure_code,
            indexed_at=document.indexed_at,
            created_at=document.created_at,
            updated_at=document.updated_at,
            index_job=IndexJobResponse(
                job_id=job.id,
                status=job.status,
                attempt_count=job.attempt_count,
                max_attempts=job.max_attempts,
                retryable=retryable,
                error_code=job.error_code,
                started_at=job.started_at,
                finished_at=job.finished_at,
            ),
        )


__all__ = [
    "AsyncReadable",
    "KnowledgeDocumentBusyError",
    "KnowledgeDocumentNotFoundError",
    "KnowledgeDocumentNotRetryableError",
    "KnowledgeDocumentTooLargeError",
    "KnowledgeEmptyDocumentError",
    "KnowledgeInvalidFilenameError",
    "KnowledgeService",
    "KnowledgeServiceError",
    "KnowledgeStateError",
    "KnowledgeStorageUnavailableError",
    "KnowledgeUnsupportedMediaTypeError",
]
