"""Chat thread execution guard contracts."""

from contextlib import AbstractAsyncContextManager
from typing import Protocol


class ChatThreadBusyError(RuntimeError):
    """同一内部 thread 已经有一次 Agent 执行正在进行."""


class ChatExecutionGuardUnavailableError(RuntimeError):
    """执行 guard 后端不可用，无法安全地启动 Agent."""


class ChatExecutionGuard(Protocol):
    """定义 ChatService 所依赖的执行权接口.

    ChatService 只知道“申请指定内部 thread 的执行权”，不应该知道
    Redis key、Lua 脚本、连接池或锁 token 等基础设施细节。

    此处只是定义了一个规范，所有符合这种声明都属于ChatExecutionGuard
    ChatService中的guard在lifespan中已经初始化完成 RedisChatExecutionGuard
    """

    def hold(
        self,
        internal_thread_id: str,
    ) -> AbstractAsyncContextManager[None]:
        """返回管理指定内部 thread 执行权的异步上下文管理器.

        Args:
            internal_thread_id: 由可信 user_id 与公开 thread_id 构造的内部标识。

        Returns:
            异步上下文管理器。进入时申请执行权，退出时释放执行权。

        Raises:
            ChatThreadBusyError: 相同内部 thread 已经有执行正在进行。
            ChatExecutionGuardUnavailableError: guard 后端无法完成安全协调。

        """
        ...
