"""使用两个真实 Redis 客户端验证 Chat execution guard 的分布式语义.

本 smoke 不调用 Graph、LLM 或工具。它只验证基础设施适配器本身：

1. 客户端 A 持有 key 时，客户端 B 对同 key 必须立即得到 busy；
2. B 在 A 持锁期间仍能取得不同 key，证明不是全局锁；
3. 正常退出、代码块异常和 asyncio task 取消后，同 key 都能再次取得；
4. Redis 中的锁名只包含摘要，不包含内部 thread ID 原文。

最终 JSON 不输出 Redis 地址、密码、原始内部 key、摘要 key 或 owner token。
"""

import asyncio
import json
import selectors
from time import perf_counter
from uuid import uuid4

from redis.asyncio import Redis

from app.core.config import settings
from app.infrastructure.chat_guard import RedisChatExecutionGuard
from app.services.chat_guard import ChatThreadBusyError

LEASE_SECONDS = 30.0
TOTAL_TIMEOUT_SECONDS = 15.0
BUSY_BUDGET_SECONDS = 0.5


class _ExpectedBlockError(RuntimeError):
    """只用于证明异常离开 async with 后仍会释放锁."""


def _elapsed_ms(started_at: float) -> float:
    """返回安全的毫秒耗时."""
    return round((perf_counter() - started_at) * 1000, 2)


def _create_redis_client() -> Redis:
    """创建一个指向当前真实 Redis 的独立异步客户端.

    Returns:
        使用 Git-ignored 环境配置的惰性 Redis 客户端。两个独立实例用于模拟
        两个 worker；真正连接会在首次命令时按需建立。
    """
    return Redis(
        host=settings.REDIS_HOST,
        port=settings.REDIS_PORT,
        db=settings.REDIS_DB,
        password=settings.REDIS_PASSWORD or None,
        socket_connect_timeout=5.0,
        socket_timeout=5.0,
        decode_responses=True,
    )


def _selector_loop_factory() -> asyncio.AbstractEventLoop:
    """创建与项目 Windows 异步基础设施一致的 Selector event loop."""
    return asyncio.SelectorEventLoop(selectors.SelectSelector())


async def _exercise_guard() -> dict[str, bool | float]:
    """运行同 key 冲突、不同 key 并发以及三类释放路径.

    Returns:
        只包含布尔判定和耗时的安全摘要。

    Raises:
        Exception: Redis 连接或 guard 行为异常。顶层只公开异常类型，并且本方法
            的 finally 仍关闭两个客户端。
    """
    started_at = perf_counter()
    first_client = _create_redis_client()
    second_client = _create_redis_client()
    first_guard = RedisChatExecutionGuard(
        first_client,
        lease_seconds=LEASE_SECONDS,
    )
    second_guard = RedisChatExecutionGuard(
        second_client,
        lease_seconds=LEASE_SECONDS,
    )

    # 使用随机内部 key，避免并行运行的两次 smoke 相互干扰。值不会进入输出。
    same_key = f"user:{uuid4().hex}:thread:{uuid4().hex}"
    different_key = f"user:{uuid4().hex}:thread:{uuid4().hex}"
    exception_key = f"user:{uuid4().hex}:thread:{uuid4().hex}"
    cancellation_key = f"user:{uuid4().hex}:thread:{uuid4().hex}"

    try:
        async with asyncio.timeout(TOTAL_TIMEOUT_SECONDS):
            # A 与 B 是两个独立 Redis client。B 能看见 A 的锁，证明协调状态不在
            # Python 对象或当前 event loop 中，而在共享 Redis。
            async with first_guard.hold(same_key):
                busy_started_at = perf_counter()
                try:
                    async with second_guard.hold(same_key):
                        raise AssertionError("busy guard unexpectedly entered")
                except ChatThreadBusyError:
                    same_key_rejected = True
                busy_elapsed_ms = _elapsed_ms(busy_started_at)

                # 不同内部 key 映射为不同摘要锁名，应在 A 持锁时照常进入。
                async with second_guard.hold(different_key):
                    different_key_entered = True

                # Redis key 不应包含原始内部 thread ID。这里只在锁仍存在时扫描，
                # 并立即转换为布尔值；key 名本身不会进入最终摘要。
                redis_keys = await second_client.keys("deep-research:chat-execution:*")
                raw_internal_key_not_exposed = all(same_key not in str(redis_key) for redis_key in redis_keys)

            # 正常离开后，第二客户端必须能重新取得同 key。
            async with second_guard.hold(same_key):
                normal_release_reacquired = True

            # 代码块内部异常仍会触发 guard finally，异常本身继续传播给调用方。
            try:
                async with first_guard.hold(exception_key):
                    raise _ExpectedBlockError
            except _ExpectedBlockError:
                pass
            async with second_guard.hold(exception_key):
                exception_release_reacquired = True

            # 创建一个长期等待的 holder task，在确认它已进入 guard 后取消。任务的
            # CancelledError 会穿过 async with，触发同一个 release finally。
            holder_entered = asyncio.Event()
            wait_forever = asyncio.Event()

            async def hold_until_cancelled() -> None:
                """持有 cancellation_key，直到 smoke 主任务主动取消."""
                async with first_guard.hold(cancellation_key):
                    holder_entered.set()
                    await wait_forever.wait()

            holder_task = asyncio.create_task(hold_until_cancelled())
            await holder_entered.wait()
            holder_task.cancel()
            cancellation_propagated = False
            try:
                await holder_task
            except asyncio.CancelledError:
                cancellation_propagated = True

            async with second_guard.hold(cancellation_key):
                cancellation_release_reacquired = True

        return {
            "same_key_rejected": same_key_rejected,
            "busy_failed_fast": busy_elapsed_ms <= BUSY_BUDGET_SECONDS * 1000,
            "different_key_entered": different_key_entered,
            "raw_internal_key_not_exposed": raw_internal_key_not_exposed,
            "normal_release_reacquired": normal_release_reacquired,
            "exception_release_reacquired": exception_release_reacquired,
            "cancellation_propagated": cancellation_propagated,
            "cancellation_release_reacquired": cancellation_release_reacquired,
            "busy_elapsed_ms": busy_elapsed_ms,
            "elapsed_ms": _elapsed_ms(started_at),
        }
    finally:
        await first_client.aclose()
        await second_client.aclose()


def _run_smoke() -> dict[str, object]:
    """在兼容 Windows 的 event loop 中运行真实 Redis guard smoke."""
    started_at = perf_counter()
    checks = asyncio.run(
        _exercise_guard(),
        loop_factory=_selector_loop_factory,
    )
    boolean_checks = tuple(value for value in checks.values() if isinstance(value, bool))
    return {
        "ok": bool(boolean_checks) and all(boolean_checks),
        **checks,
        "total_elapsed_ms": _elapsed_ms(started_at),
    }


def main() -> int:
    """打印单行安全 JSON，并返回适合 PowerShell/CI 的退出码."""
    started_at = perf_counter()
    try:
        summary = _run_smoke()
    except Exception as error:
        # Redis 驱动异常可能包含地址等诊断信息，顶层只公开异常类名。
        print(
            json.dumps(
                {
                    "ok": False,
                    "error_type": type(error).__name__,
                    "elapsed_ms": _elapsed_ms(started_at),
                }
            )
        )
        return 1

    print(json.dumps(summary))
    return 0 if summary["ok"] is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
