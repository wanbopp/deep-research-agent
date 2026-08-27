"""会话标题 schema 与后台任务生命周期的聚焦测试."""

import asyncio

import pytest
from pydantic import ValidationError

from app.infrastructure.background_tasks import (
    AsyncioBackgroundTaskSubmitter,
    BackgroundTaskSubmissionClosedError,
)
from app.schemas.chat_title import ChatSessionTitleResult


def test_chat_title_result_normalizes_and_rejects_placeholder() -> None:
    """模型标题应被压成单行，并拒绝系统默认占位值."""
    result = ChatSessionTitleResult(title="  PostgreSQL\n原子 Claim  ")

    assert result.title == "PostgreSQL 原子 Claim"

    with pytest.raises(ValidationError):
        ChatSessionTitleResult(title="New chat")


@pytest.mark.anyio
async def test_background_submitter_drains_cancels_and_closes() -> None:
    """Shutdown 应等待短任务、取消遗留任务，并永久拒绝后续提交."""
    drained = asyncio.Event()
    draining_submitter = AsyncioBackgroundTaskSubmitter()

    async def short_operation() -> None:
        """让出一次调度后正常结束，证明 shutdown 会执行 drain."""
        await asyncio.sleep(0)
        drained.set()

    draining_submitter.submit(short_operation, name="short-operation")
    await draining_submitter.shutdown(timeout_seconds=1.0)

    assert drained.is_set()
    assert draining_submitter.active_count == 0
    assert not draining_submitter.accepting

    cancelled = asyncio.Event()
    started = asyncio.Event()
    cancelling_submitter = AsyncioBackgroundTaskSubmitter()

    async def long_operation() -> None:
        """模拟不会在 shutdown 预算内自然结束的后台 I/O."""
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            # finally 是资源释放位置；测试证明 cancel 后仍会执行清理代码。
            cancelled.set()

    cancelling_submitter.submit(long_operation, name="long-operation")
    await started.wait()
    await cancelling_submitter.shutdown(timeout_seconds=0.0)

    assert cancelled.is_set()
    assert cancelling_submitter.active_count == 0
    assert not cancelling_submitter.accepting

    with pytest.raises(BackgroundTaskSubmissionClosedError):
        cancelling_submitter.submit(short_operation, name="late-operation")
