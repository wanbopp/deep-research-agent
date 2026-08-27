"""基于 asyncio 的进程内后台任务提交器."""

import asyncio

from app.core.logging import logger
from app.services.background_tasks import BackgroundOperationFactory


class AsyncioBackgroundTaskSubmitter:
    """跟踪本 worker 内的后台任务，并消费每个任务的最终异常.

    当前实现只提供 12D 需要的提交、强引用和异常观察。任务集合不能解决跨 worker
    协调，也还没有 shutdown drain；这些生命周期边界会在 12E 显式增加。
    """

    def __init__(self) -> None:
        """创建空的任务集合；构造阶段不要求 event loop 已经运行."""
        # asyncio 只保存 Task 的弱引用。这里保留强引用，避免没有其他引用的后台
        # 任务在完成前被垃圾回收；done callback 会负责及时移除。
        self._tasks: set[asyncio.Task[None]] = set()

    @property
    def active_count(self) -> int:
        """返回当前 worker 中仍未完成的受管任务数量."""
        return len(self._tasks)

    def submit(
        self,
        operation_factory: BackgroundOperationFactory,
        *,
        name: str,
    ) -> None:
        """在当前 event loop 创建任务并安装安全的完成回调.

        Args:
            operation_factory: 后台操作工厂。它在受管 Task 内调用，因此同步抛出的
                异常也会变成 Task 异常并由完成回调消费。
            name: 不含请求数据的稳定任务类别。

        Raises:
            ValueError: name 去除空白后为空。
            RuntimeError: 调用位置不在运行中的 asyncio event loop 内。
        """
        task_name = name.strip()
        if not task_name:
            raise ValueError("background task name must not be empty")

        async def run_operation() -> None:
            """把 factory 调用也纳入 Task 的异常边界."""
            await operation_factory()

        # create_task 只安排执行，不等待后台操作，因此 HTTP/流式响应不依赖记忆
        # 提取是否成功。没有运行中 event loop 时应立即失败，暴露错误接线。
        task = asyncio.create_task(
            run_operation(),
            name=task_name,
        )
        self._tasks.add(task)
        task.add_done_callback(self._consume_result)

    def _consume_result(self, task: asyncio.Task[None]) -> None:
        """释放强引用并消费完成、失败或取消状态.

        读取 ``task.exception()`` 不会重新抛出异常，但会把它标记为已观察，避免
        event loop 在稍后输出难以关联的 ``Task exception was never retrieved``。
        日志只记录固定任务名和异常类型，不记录异常文本或业务正文。
        """
        self._tasks.discard(task)

        if task.cancelled():
            logger.info(
                "background_task_cancelled",
                task_name=task.get_name(),
            )
            return

        error = task.exception()
        if error is None:
            logger.info(
                "background_task_completed",
                task_name=task.get_name(),
            )
            return

        logger.error(
            "background_task_failed",
            task_name=task.get_name(),
            error_type=type(error).__name__,
        )


__all__ = ["AsyncioBackgroundTaskSubmitter"]
