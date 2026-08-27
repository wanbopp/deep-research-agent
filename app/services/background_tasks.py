"""后台任务提交协议.

应用服务依赖这个小接口，不直接调用 ``asyncio.create_task()``。应用层只需要
“立即提交且异常可观察”的语义；shutdown 等待和取消属于 infrastructure 生命周期，
因此不扩大本 Protocol，也不让 ChatService 管理进程关闭。
"""

from collections.abc import Awaitable, Callable
from typing import Protocol

# 使用 factory 而不是已经创建的 coroutine。submitter 可以先完成参数检查，再在
# 自己管理的任务中调用 factory；即使提交失败，也不会遗留从未 await 的 coroutine。
BackgroundOperationFactory = Callable[[], Awaitable[None]]


class BackgroundTaskSubmitter(Protocol):
    """定义应用层提交一次非阻塞后台操作所需的最小能力."""

    def submit(
        self,
        operation_factory: BackgroundOperationFactory,
        *,
        name: str,
    ) -> None:
        """提交操作并立即返回，不把后台结果耦合到当前响应.

        Args:
            operation_factory: 无参数异步操作工厂。调用后产生本次任务自己的
                awaitable，不能复用已执行或已关闭的 coroutine。
            name: 由服务端代码给出的稳定任务类别，用于安全日志和调试；不能包含
                用户输入、thread ID、Prompt 或记忆正文。

        Raises:
            ValueError: name 为空。
            RuntimeError: 当前线程没有运行中的 asyncio event loop。
        """
        ...


__all__ = ["BackgroundOperationFactory", "BackgroundTaskSubmitter"]
