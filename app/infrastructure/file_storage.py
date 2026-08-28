"""开发环境使用的本地原始文件存储适配器."""

import asyncio
import os
from pathlib import Path, PurePosixPath
from uuid import uuid4

from app.services.file_storage import (
    FileStorageConflictError,
    FileStorageError,
)


class LocalFileStorage:
    """把服务端 storage key 安全映射到指定根目录.

    adapter 不接收原始文件名。KnowledgeService 会生成只含固定目录、用户 UUID
    与随机对象 UUID 的 key；这里仍执行路径校验，形成第二层防护。
    """

    def __init__(self, root: Path) -> None:
        """创建存储根目录并保存其规范化绝对路径.

        Args:
            root: 开发环境文件根目录。相对路径按应用启动目录解析。

        Raises:
            FileStorageError: 根目录无法创建或不是目录。
        """
        try:
            resolved_root = root.expanduser().resolve()
            resolved_root.mkdir(parents=True, exist_ok=True)
        except OSError as error:
            raise FileStorageError("File storage root is unavailable") from error
        if not resolved_root.is_dir():
            raise FileStorageError("File storage root is unavailable")
        self._root = resolved_root

    async def put(self, key: str, content: bytes) -> None:
        """通过临时文件与原子 hard-link 发布对象，拒绝覆盖已有 key.

        Args:
            key: 由服务端生成并通过路径校验的相对对象 key。
            content: 已在 KnowledgeService 中完成大小限制的文件字节。

        Raises:
            FileStorageConflictError: 目标 key 已经存在。
            FileStorageError: 目录创建、写入、刷盘或原子发布失败。

        Notes:
            阻塞文件系统调用通过 ``asyncio.to_thread`` 移出事件循环。临时文件
            与目标文件位于同一目录，hard-link 发布不会出现“先看到半个文件”。
        """
        path = self._resolve_key(key)
        await asyncio.to_thread(self._put_sync, path, content)

    async def read(self, key: str) -> bytes:
        """读取对象；驱动错误转换为不携带本机路径的稳定异常."""
        path = self._resolve_key(key)
        try:
            return await asyncio.to_thread(path.read_bytes)
        except OSError as error:
            raise FileStorageError("Stored file is unavailable") from error

    async def delete(self, key: str) -> None:
        """幂等删除对象，便于数据库/文件系统三阶段清理重试."""
        path = self._resolve_key(key)
        try:
            await asyncio.to_thread(path.unlink, missing_ok=True)
        except OSError as error:
            raise FileStorageError("Stored file could not be deleted") from error

    async def exists(self, key: str) -> bool:
        """在线程池中检查对象是否存在."""
        path = self._resolve_key(key)
        return await asyncio.to_thread(path.is_file)

    def _resolve_key(self, key: str) -> Path:
        """验证 storage key 并保证最终路径仍位于根目录内."""
        if not key or "\\" in key or "\x00" in key:
            raise FileStorageError("Invalid storage key")

        pure_key = PurePosixPath(key)
        if pure_key.is_absolute() or any(part in {"", ".", ".."} for part in pure_key.parts):
            raise FileStorageError("Invalid storage key")

        path = self._root.joinpath(*pure_key.parts).resolve()
        if not path.is_relative_to(self._root):
            raise FileStorageError("Invalid storage key")
        return path

    @staticmethod
    def _put_sync(path: Path, content: bytes) -> None:
        """同步写临时文件并以不覆盖方式原子发布."""
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = path.parent / f".{path.name}.{uuid4().hex}.tmp"
        try:
            with temporary_path.open("xb") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            try:
                os.link(temporary_path, path)
            except FileExistsError as error:
                raise FileStorageConflictError("Storage key already exists") from error
            except OSError as error:
                raise FileStorageError("Stored file could not be published") from error
        except FileStorageConflictError:
            raise
        except OSError as error:
            raise FileStorageError("Stored file could not be written") from error
        finally:
            temporary_path.unlink(missing_ok=True)


__all__ = ["LocalFileStorage"]
