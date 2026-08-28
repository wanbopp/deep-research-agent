"""原始知识文件的存储能力协议与稳定错误."""

from typing import Protocol


class FileStorageError(RuntimeError):
    """文件存储无法完成安全读写时的稳定应用错误."""


class FileStorageConflictError(FileStorageError):
    """服务端生成的 storage key 意外发生冲突."""


class FileStorage(Protocol):
    """KnowledgeService 所依赖的最小原始文件存储能力."""

    async def put(self, key: str, content: bytes) -> None:
        """原子保存完整文件；已存在的 key 必须拒绝覆盖."""
        ...

    async def read(self, key: str) -> bytes:
        """读取指定服务端 key 的原始字节."""
        ...

    async def delete(self, key: str) -> None:
        """幂等删除指定对象；对象不存在也视为目标状态已满足."""
        ...

    async def exists(self, key: str) -> bool:
        """判断指定对象是否存在."""
        ...


__all__ = [
    "FileStorage",
    "FileStorageConflictError",
    "FileStorageError",
]
