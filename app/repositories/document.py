"""文档元数据访问边界."""

from datetime import datetime
from uuid import UUID

from sqlmodel import col, select

from app.models import Document, DocumentStatus, utc_now
from app.repositories.base import RepositoryBase


class DocumentRepository(RepositoryBase):
    """封装文档写入、owner-scoped 查询和状态变化."""

    async def create(
        self,
        *,
        user_id: UUID,
        original_filename: str,
        content_type: str,
        size_bytes: int,
        content_sha256: str,
        storage_key: str,
    ) -> Document:
        """在当前事务中创建 pending 文档元数据.

        Repository 只执行 ``flush``，不提交事务。调用方必须把 Document 与其
        IndexJob 放进同一个 ``session.begin()``，避免只创建一半业务事实。
        """
        document = Document(
            user_id=user_id,
            original_filename=original_filename,
            content_type=content_type,
            size_bytes=size_bytes,
            content_sha256=content_sha256,
            storage_key=storage_key,
            status=DocumentStatus.PENDING,
        )
        return await self._persist(document, resource="document")

    async def get_by_id(
        self,
        document_id: UUID,
        *,
        user_id: UUID,
        include_deleting: bool = False,
    ) -> Document | None:
        """按文档 ID 与可信用户 ID 查询，跨用户和不存在统一返回 None."""
        statement = select(Document).where(
            Document.id == document_id,
            Document.user_id == user_id,
        )
        if not include_deleting:
            statement = statement.where(Document.status != DocumentStatus.DELETING)
        result = await self._session.execute(statement)
        return result.scalar_one_or_none()

    async def get_by_content_hash(
        self,
        *,
        user_id: UUID,
        content_sha256: str,
        include_deleting: bool = False,
    ) -> Document | None:
        """在当前 owner namespace 内查找相同内容，供幂等上传使用."""
        statement = select(Document).where(
            Document.user_id == user_id,
            Document.content_sha256 == content_sha256,
        )
        if not include_deleting:
            statement = statement.where(Document.status != DocumentStatus.DELETING)
        result = await self._session.execute(statement)
        return result.scalar_one_or_none()

    async def get_for_update(
        self,
        document_id: UUID,
        *,
        user_id: UUID,
    ) -> Document | None:
        """锁定 owner-scoped 文档行，供删除或人工重试执行状态转换."""
        statement = (
            select(Document)
            .where(
                Document.id == document_id,
                Document.user_id == user_id,
            )
            .with_for_update()
        )
        result = await self._session.execute(statement)
        return result.scalar_one_or_none()

    async def get_internal_for_update(self, document_id: UUID) -> Document | None:
        """供可信后台 worker 按内部外键锁行；HTTP route 不得调用此方法."""
        statement = select(Document).where(Document.id == document_id).with_for_update()
        result = await self._session.execute(statement)
        return result.scalar_one_or_none()

    async def list_by_user(self, user_id: UUID) -> tuple[Document, ...]:
        """稳定列出用户可见文档；删除 tombstone 不重新暴露给普通列表."""
        statement = (
            select(Document)
            .where(
                Document.user_id == user_id,
                Document.status != DocumentStatus.DELETING,
            )
            .order_by(col(Document.created_at), col(Document.id))
        )
        result = await self._session.execute(statement)
        return tuple(result.scalars().all())

    async def mark_deleting(self, document: Document) -> None:
        """持久化删除意图，使跨文件系统清理可以安全重试."""
        document.status = DocumentStatus.DELETING
        document.failure_code = None
        document.updated_at = utc_now()
        self._session.add(document)
        await self._session.flush()

    async def set_status(
        self,
        document: Document,
        *,
        status: DocumentStatus,
        failure_code: str | None = None,
        indexed_at: datetime | None = None,
    ) -> None:
        """在 worker 持有行锁时同步文档状态与稳定错误码."""
        document.status = status
        document.failure_code = failure_code
        document.indexed_at = indexed_at
        document.updated_at = utc_now()
        self._session.add(document)
        await self._session.flush()

    async def delete(self, document: Document) -> None:
        """删除已完成外部文件清理的文档行，IndexJob 由外键级联删除."""
        await self._session.delete(document)
        await self._session.flush()


__all__ = ["DocumentRepository"]
