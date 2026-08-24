"""异步 Repository 的公共会话与写入边界."""

from typing import TypeVar

from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession


ModelT = TypeVar("ModelT")


class RepositoryError(RuntimeError):
    """所有持久化层领域错误的基类.

    Repository 不依赖 FastAPI，因此这里不能抛 ``HTTPException``。上层 service
    可以捕获这些错误并转换为业务错误，route 再决定最终的 HTTP 状态码。
    """


class RepositoryConflictError(RepositoryError):
    """数据库约束拒绝一次写入，例如唯一键或外键冲突."""

    def __init__(self, resource: str) -> None:
        """记录资源种类，但不把邮箱等敏感输入写进异常文本."""
        self.resource = resource
        super().__init__(f"Database constraint rejected {resource}")


class RepositoryBase:
    """保存一次工作单元共享的 ``AsyncSession``.

    一个 application service 可以创建多个 Repository，并把同一个 Session
    传给它们。这样所有 Repository 的 SQL 都落在同一个数据库事务中。
    """

    def __init__(self, session: AsyncSession) -> None:
        """接收调用方管理生命周期和事务的 Session."""
        self._session = session

    async def _persist(self, instance: ModelT, *, resource: str) -> ModelT:
        """把新对象写入当前事务并刷新数据库生成的字段.

        ``flush`` 会把待执行的 INSERT 发送给 PostgreSQL，因此可以提前发现
        唯一键、外键等约束错误；但它不会提交事务。真正的 commit 仍由外层
        ``async with session.begin()`` 在所有操作成功后统一完成。

        Raises:
            RepositoryConflictError: PostgreSQL 拒绝违反约束的 INSERT。异常继续
                向外传播后，``session.begin()`` 会负责 rollback。
        """
        self._session.add(instance)
        try:
            await self._session.flush()
        except IntegrityError as exc:
            # flush 失败后 Session 处于 failed transaction 状态。这里不能吞掉异常
            # 或继续查询；必须让异常离开事务上下文，由 begin() 完成 rollback。
            raise RepositoryConflictError(resource) from exc

        # refresh 从数据库重新读取当前行。现在主要用于取得数据库规范化后的值；
        # 将来增加 server_default 字段时，调用方也能直接拿到完整对象。
        await self._session.refresh(instance)
        return instance
