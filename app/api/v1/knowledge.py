"""认证用户的知识文档上传、查询、重试与删除 API."""

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, File, Response, UploadFile, status

from app.api.dependencies import CurrentUserDependency, get_knowledge_service
from app.schemas.base import ErrorResponse
from app.schemas.knowledge import (
    DocumentListResponse,
    DocumentResponse,
    DocumentUploadResponse,
    IndexJobResponse,
)
from app.services.knowledge import KnowledgeService

router = APIRouter(prefix="/knowledge/documents", tags=["knowledge"])

KnowledgeServiceDependency = Annotated[
    KnowledgeService,
    Depends(get_knowledge_service),
]


@router.post(
    "",
    response_model=DocumentUploadResponse,
    status_code=status.HTTP_201_CREATED,
    responses={
        status.HTTP_401_UNAUTHORIZED: {"model": ErrorResponse, "description": "Authentication required"},
        status.HTTP_413_CONTENT_TOO_LARGE: {"model": ErrorResponse, "description": "Document is too large"},
        status.HTTP_415_UNSUPPORTED_MEDIA_TYPE: {
            "model": ErrorResponse,
            "description": "Document MIME type is not supported",
        },
        status.HTTP_503_SERVICE_UNAVAILABLE: {
            "model": ErrorResponse,
            "description": "File storage is temporarily unavailable",
        },
    },
)
async def upload_document(
    file: Annotated[UploadFile, File(description="需要加入当前用户知识库的原始文件")],
    service: KnowledgeServiceDependency,
    current_user: CurrentUserDependency,
) -> DocumentUploadResponse:
    """上传原始文件并立即创建 pending 索引任务.

    Args:
        file: Starlette 管理的 multipart 临时文件。service 只依赖其异步 read 接口。
        service: 当前请求的事务与 FileStorage 协调器。
        current_user: JWT 验签并经数据库确认的可信 owner。

    Returns:
        文档、索引任务和 owner 内内容去重标记。返回 201 只表示上传事务完成，
        不表示 parse、chunk、embedding 或索引已经完成。
    """
    return await service.upload(
        user_id=current_user.user_id,
        filename=file.filename,
        content_type=file.content_type,
        source=file,
    )


@router.get("", response_model=DocumentListResponse)
async def list_documents(
    service: KnowledgeServiceDependency,
    current_user: CurrentUserDependency,
) -> DocumentListResponse:
    """列出当前认证用户拥有且未进入 deleting 的文档."""
    return await service.list_owned(user_id=current_user.user_id)


@router.get("/{document_id}", response_model=DocumentResponse)
async def read_document(
    document_id: UUID,
    service: KnowledgeServiceDependency,
    current_user: CurrentUserDependency,
) -> DocumentResponse:
    """按 document_id 与可信 user_id 读取文档，跨用户时统一表现为 404."""
    return await service.get_owned(
        document_id=document_id,
        user_id=current_user.user_id,
    )


@router.get("/{document_id}/index-job", response_model=IndexJobResponse)
async def read_document_index_job(
    document_id: UUID,
    service: KnowledgeServiceDependency,
    current_user: CurrentUserDependency,
) -> IndexJobResponse:
    """读取文档当前索引任务，不公开 worker ID、租约或 fencing token."""
    document = await service.get_owned(
        document_id=document_id,
        user_id=current_user.user_id,
    )
    return document.index_job


@router.post(
    "/{document_id}/retry",
    response_model=DocumentResponse,
    responses={
        status.HTTP_409_CONFLICT: {
            "model": ErrorResponse,
            "description": "The current document state cannot be retried",
        }
    },
)
async def retry_document_index(
    document_id: UUID,
    service: KnowledgeServiceDependency,
    current_user: CurrentUserDependency,
) -> DocumentResponse:
    """把当前用户未耗尽次数的 failed job 重新放回 pending 队列."""
    return await service.retry_owned(
        document_id=document_id,
        user_id=current_user.user_id,
    )


@router.delete(
    "/{document_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    response_class=Response,
    responses={
        status.HTTP_409_CONFLICT: {
            "model": ErrorResponse,
            "description": "The document is currently being indexed",
        },
        status.HTTP_503_SERVICE_UNAVAILABLE: {
            "model": ErrorResponse,
            "description": "File storage cleanup is temporarily unavailable",
        },
    },
)
async def delete_document(
    document_id: UUID,
    service: KnowledgeServiceDependency,
    current_user: CurrentUserDependency,
) -> Response:
    """删除当前用户文档、原始文件与级联 IndexJob."""
    await service.delete_owned(
        document_id=document_id,
        user_id=current_user.user_id,
    )
    return Response(status_code=status.HTTP_204_NO_CONTENT)


__all__ = ["router"]
