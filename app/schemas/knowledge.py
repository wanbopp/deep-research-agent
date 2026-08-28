"""知识文档与索引任务的公开 API 模型."""

from uuid import UUID

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field

from app.models import DocumentStatus, IndexJobStatus
from app.models.document import (
    MAX_DOCUMENT_CONTENT_TYPE_LENGTH,
    MAX_DOCUMENT_FAILURE_CODE_LENGTH,
    MAX_DOCUMENT_FILENAME_LENGTH,
)
from app.models.index_job import MAX_INDEX_JOB_ERROR_CODE_LENGTH


class _StrictKnowledgeModel(BaseModel):
    """统一禁止未知字段和响应对象原地修改."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class IndexJobResponse(_StrictKnowledgeModel):
    """客户端可观察的索引任务状态，不公开 worker 或 claim token."""

    job_id: UUID
    status: IndexJobStatus
    attempt_count: int = Field(ge=0)
    max_attempts: int = Field(gt=0)
    retryable: bool
    error_code: str | None = Field(default=None, max_length=MAX_INDEX_JOB_ERROR_CODE_LENGTH)
    started_at: AwareDatetime | None = None
    finished_at: AwareDatetime | None = None


class DocumentResponse(_StrictKnowledgeModel):
    """用户可见的文档元数据与当前索引状态."""

    document_id: UUID
    original_filename: str = Field(min_length=1, max_length=MAX_DOCUMENT_FILENAME_LENGTH)
    content_type: str = Field(min_length=1, max_length=MAX_DOCUMENT_CONTENT_TYPE_LENGTH)
    size_bytes: int = Field(gt=0)
    status: DocumentStatus
    failure_code: str | None = Field(default=None, max_length=MAX_DOCUMENT_FAILURE_CODE_LENGTH)
    indexed_at: AwareDatetime | None = None
    created_at: AwareDatetime
    updated_at: AwareDatetime
    index_job: IndexJobResponse


class DocumentUploadResponse(_StrictKnowledgeModel):
    """上传结果；deduplicated 表示复用了 owner 内已有内容."""

    document: DocumentResponse
    deduplicated: bool


class DocumentListResponse(_StrictKnowledgeModel):
    """当前用户拥有的稳定排序文档列表."""

    documents: tuple[DocumentResponse, ...] = ()


__all__ = [
    "DocumentListResponse",
    "DocumentResponse",
    "DocumentUploadResponse",
    "IndexJobResponse",
]
