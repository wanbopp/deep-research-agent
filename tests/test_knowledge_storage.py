"""知识文件存储与公开上传协议的聚焦门禁."""

from pathlib import Path

import pytest

from app.infrastructure.file_storage import LocalFileStorage
from app.main import app
from app.services.file_storage import (
    FileStorageConflictError,
    FileStorageError,
)


@pytest.mark.anyio
async def test_local_storage_is_atomic_idempotent_and_rejects_unsafe_keys(tmp_path: Path) -> None:
    """本地 adapter 不得覆盖对象、逃逸根目录，删除必须可重试."""
    storage = LocalFileStorage(tmp_path / "knowledge")
    key = "documents/owner/object"

    await storage.put(key, b"knowledge-content")
    assert await storage.exists(key) is True
    assert await storage.read(key) == b"knowledge-content"

    with pytest.raises(FileStorageConflictError):
        await storage.put(key, b"replacement-must-not-win")
    assert await storage.read(key) == b"knowledge-content"

    for unsafe_key in ("../escape", "/absolute", "documents\\escape", ""):
        with pytest.raises(FileStorageError):
            await storage.put(unsafe_key, b"unsafe")

    await storage.delete(key)
    await storage.delete(key)
    assert await storage.exists(key) is False


def test_openapi_documents_multipart_upload_and_index_status() -> None:
    """OpenAPI 必须公开 multipart 上传及独立索引状态查询."""
    paths = app.openapi()["paths"]
    upload = paths["/api/v1/knowledge/documents"]["post"]
    index_status = paths["/api/v1/knowledge/documents/{document_id}/index-job"]["get"]

    assert "multipart/form-data" in upload["requestBody"]["content"]
    assert upload["responses"]["201"]["content"]["application/json"]["schema"]
    assert index_status["responses"]["200"]["content"]["application/json"]["schema"]
