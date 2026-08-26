"""长期记忆存储协议与稳定应用错误.

本模块只描述应用层需要的能力，不导入 FastAPI、SQLAlchemy、pgvector、Redis
或 LangGraph。未来 MemoryService 和 Agent 依赖该协议，PostgreSQL/pgvector
只是在 infrastructure 层提供的一种实现。
"""

from typing import Protocol
from uuid import UUID

from app.schemas.memory import MemoryCreate, MemoryItem, MemoryQuery


class MemoryUnavailableError(RuntimeError):
    """长期记忆后端无法安全完成操作.

    错误文本保持固定，不包含数据库连接、向量、查询正文、用户 ID、记忆内容或
    底层驱动消息。调用方以后可以按具体用例决定降级为空记忆还是终止请求。
    """

    def __init__(self) -> None:
        """创建可以安全跨越应用层边界的稳定异常."""
        super().__init__("Memory backend is unavailable")


class MemoryStore(Protocol):
    """定义长期记忆持久化适配器必须提供的能力.

    Protocol 不提供代码实现，也不要求具体类继承。只要对象提供相同异步方法
    签名，Pyright 就能把它视为 MemoryStore，实现依赖倒置。
    """

    async def search(
        self,
        *,
        user_id: UUID,
        query: MemoryQuery,
    ) -> tuple[MemoryItem, ...]:
        """在一个可信用户 namespace 内检索相关记忆.

        Args:
            user_id: 认证 dependency 建立的可信用户 UUID。adapter 必须把它作为
                数据库过滤条件，不能只在查询后用 Python 过滤。
            query: 不包含用户归属的检索文本、数量上限和可选分类集合。

        Returns:
            按相关性排序的不可变元组，最多返回 ``query.limit`` 条。没有结果时
            返回空元组，不把普通未命中当作异常。

        Raises:
            MemoryUnavailableError: 后端无法可靠完成用户隔离查询。
        """
        ...

    async def add(
        self,
        *,
        user_id: UUID,
        memory: MemoryCreate,
    ) -> MemoryItem:
        """把候选记忆绑定到可信用户并持久化.

        Args:
            user_id: 来自认证链的可信用户 UUID，不能从 memory 或模型输出读取。
            memory: 已通过长度、分类和来源 thread 校验的候选记忆。

        Returns:
            带权威 UUID、user_id 和审计时间的持久化结果。

        Raises:
            MemoryUnavailableError: 后端无法安全完成写入。

        Notes:
            12A 只定义边界。12B adapter 写入前还要验证 source_thread_id 属于同一
            user_id，避免把其他用户的会话当作记忆来源。
        """
        ...

    async def delete(
        self,
        *,
        user_id: UUID,
        memory_id: UUID,
    ) -> None:
        """在可信用户作用域内幂等删除一条记忆.

        Args:
            user_id: 认证链建立的可信用户 UUID。
            memory_id: 待删除记忆的业务 UUID。

        Raises:
            MemoryUnavailableError: 后端无法完成删除。

        Notes:
            删除条件必须同时包含 user_id 和 memory_id。不存在或属于其他用户时
            都视为幂等完成，避免通过返回值枚举其他用户是否拥有该 memory_id。
        """
        ...


__all__ = ["MemoryStore", "MemoryUnavailableError"]
