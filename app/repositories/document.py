"""文档元数据访问边界."""

from uuid import UUID

from sqlmodel import select

from app.models import Document, DocumentStatus
from app.repositories.base import RepositoryBase


class DocumentRepository(RepositoryBase):
    """封装文档元数据的创建和用户所有权查询."""

    async def create(
        self,
        *,
        user_id: UUID,
        original_filename: str,
        content_type: str | None = None,
        status: DocumentStatus = DocumentStatus.PENDING,
    ) -> Document:
        """在当前事务中创建文档元数据."""
        document = Document(
            user_id=user_id,
            original_filename=original_filename,
            content_type=content_type,
            status=status,
        )
        return await self._persist(document, resource="document")

    async def get_by_id(self, document_id: UUID, *, user_id: UUID) -> Document | None:
        """同时按文档 ID 和用户 ID 查询，落实资源所有权过滤."""
        statement = select(Document).where(
            Document.id == document_id,
            Document.user_id == user_id,
        )
        result = await self._session.execute(statement)
        return result.scalar_one_or_none()

    async def list_by_user(self, user_id: UUID) -> tuple[Document, ...]:
        """稳定地列出一个用户拥有的全部文档."""
        statement = select(Document).where(Document.user_id == user_id).order_by("created_at", "id")
        result = await self._session.execute(statement)
        return tuple(result.scalars().all())
