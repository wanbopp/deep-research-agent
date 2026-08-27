"""基于 asyncio 的进程内后台任务提交器."""

import asyncio

from app.core.logging import logger
from app.services.background_tasks import BackgroundOperationFactory


class BackgroundTaskSubmissionClosedError(RuntimeError):
    """应用已进入 shutdown，不能再接收依赖底层资源的新任务."""


class AsyncioBackgroundTaskSubmitter:
    """跟踪本 worker 的后台任务，并管理提交、异常消费和 shutdown 收敛.

    task 集合只负责当前进程的资源生命周期，不提供跨 worker 互斥。会话命名的
    分布式单赢家语义由 PostgreSQL claim 负责，二者不能互相替代。
    """

    def __init__(self) -> None:
        """创建空的任务集合；构造阶段不要求 event loop 已经运行."""
        # asyncio 只保存 Task 的弱引用。这里保留强引用，避免没有其他引用的后台
        # 任务在完成前被垃圾回收；done callback 会负责及时移除。
        self._tasks: set[asyncio.Task[None]] = set()
        self._accepting = True

    @property
    def active_count(self) -> int:
        """返回当前 worker 中仍未完成的受管任务数量."""
        return len(self._tasks)

    @property
    def accepting(self) -> bool:
        """返回提交器当前是否仍接受新任务."""
        return self._accepting

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
            BackgroundTaskSubmissionClosedError: 应用已开始 shutdown。
            RuntimeError: 调用位置不在运行中的 asyncio event loop 内。
        """
        task_name = name.strip()
        if not task_name:
            raise ValueError("background task name must not be empty")
        if not self._accepting:
            # factory 尚未调用，因此拒绝不会产生“coroutine was never awaited”。
            raise BackgroundTaskSubmissionClosedError("background task submitter is shutting down")

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

    async def shutdown(self, *, timeout_seconds: float) -> None:
        """停止接收任务，并在预算内等待或取消已有任务.

        Args:
            timeout_seconds: 允许任务自然结束的非负秒数。零表示立即取消。

        Raises:
            ValueError: timeout_seconds 小于零。
            CancelledError: 外层 shutdown 自身被取消；方法仍会先取消并回收任务。

        Notes:
            本方法可重复调用。先关闭 accepting，再获取 task 快照，保证同一 event
            loop 中不会在快照后插入新任务。超时任务会 cancel，并通过 gather 消费
            最终状态，随后才能安全关闭 ORM、Redis 和 provider 资源。
        """
        if timeout_seconds < 0:
            raise ValueError("timeout_seconds must not be negative")

        self._accepting = False
        tasks = tuple(self._tasks)
        if not tasks:
            return

        try:
            _done, pending = await asyncio.wait(
                tasks,
                timeout=timeout_seconds,
            )
        except asyncio.CancelledError:
            # shutdown 本身被取消时也不能放任子任务继续访问即将关闭的资源。
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise

        if pending:
            logger.warning(
                "background_task_shutdown_timeout",
                pending_count=len(pending),
                timeout_seconds=timeout_seconds,
            )
            for task in pending:
                task.cancel()

        # gather 同时等待取消清理的 finally，并消费结果。done callback 仍负责
        # 固定事件日志和从强引用集合移除，二者执行顺序不影响最终 active_count。
        await asyncio.gather(*tasks, return_exceptions=True)
        self._tasks.difference_update(tasks)

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


__all__ = [
    "AsyncioBackgroundTaskSubmitter",
    "BackgroundTaskSubmissionClosedError",
]
