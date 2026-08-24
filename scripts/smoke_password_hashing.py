"""使用真实 Argon2id 验收 PasswordHasher 的安全和异步边界.

这不是 fake 或 mock：脚本会真实计算两个带独立随机 salt 的 Argon2id 哈希，再执行
正确密码、错误密码与畸形哈希验证。输出只包含布尔结论和耗时，绝不打印密码或完整
哈希，因此既能作为教学观察点，也不会把 credential 留在终端历史中。
"""

import asyncio
import json
from time import perf_counter

from pydantic import SecretStr

from app.services.auth import PasswordHasher


def _elapsed_ms(started_at: float) -> float:
    """返回适合 smoke 摘要的毫秒耗时."""
    return round((perf_counter() - started_at) * 1000, 2)


async def _observe_event_loop(task: asyncio.Task[str]) -> int:
    """统计 Argon2 运行期间事件循环还能获得多少次调度机会.

    ``PasswordHasher.hash`` 内部若直接执行同步 Argon2，事件循环会卡住，当前协程
    直到哈希结束后才可能运行，计数通常为 0。正确使用 ``to_thread.run_sync`` 后，
    哈希工作在线程中进行，而事件循环可以继续调度这个观察协程。
    """
    scheduler_turns = 0
    while not task.done():
        scheduler_turns += 1
        # 极短 sleep 主动交还控制权，同时避免纯 sleep(0) 形成高 CPU 忙循环。
        await asyncio.sleep(0.001)
    return scheduler_turns


async def _run_smoke() -> dict[str, object]:
    """执行真实哈希、验证、随机 salt 与异步调度检查."""
    started_at = perf_counter()
    hasher = PasswordHasher()

    # SecretStr 的 repr 会隐藏值。明文只在 PasswordHasher 内部短暂解包，下面所有
    # JSON 字段都只报告判断结果，不输出 get_secret_value() 或编码哈希。
    password = SecretStr("Real-Argon2-Smoke-Password")
    wrong_password = SecretStr("Wrong-Argon2-Smoke-Password")

    first_hash_task = asyncio.create_task(hasher.hash(password))
    scheduler_turns = await _observe_event_loop(first_hash_task)
    first_hash = await first_hash_task
    second_hash = await hasher.hash(password)

    algorithm_is_argon2id = first_hash.startswith("$argon2id$") and second_hash.startswith("$argon2id$")
    hashes_use_random_salts = first_hash != second_hash
    correct_password_matches = await hasher.verify(password, first_hash) and await hasher.verify(
        password,
        second_hash,
    )
    wrong_password_rejected = not await hasher.verify(wrong_password, first_hash)
    malformed_hash_rejected = not await hasher.verify(password, "not-a-password-hash")
    plaintext_not_stored = first_hash != password.get_secret_value()
    event_loop_progressed = scheduler_turns > 0

    ok = all(
        (
            algorithm_is_argon2id,
            hashes_use_random_salts,
            correct_password_matches,
            wrong_password_rejected,
            malformed_hash_rejected,
            plaintext_not_stored,
            event_loop_progressed,
        )
    )
    return {
        "ok": ok,
        "algorithm_is_argon2id": algorithm_is_argon2id,
        "hashes_use_random_salts": hashes_use_random_salts,
        "correct_password_matches": correct_password_matches,
        "wrong_password_rejected": wrong_password_rejected,
        "malformed_hash_rejected": malformed_hash_rejected,
        "plaintext_not_stored": plaintext_not_stored,
        "event_loop_progressed": event_loop_progressed,
        "scheduler_turns": scheduler_turns,
        "elapsed_ms": _elapsed_ms(started_at),
    }


def main() -> int:
    """运行异步 smoke 并打印单行、安全的 JSON 摘要."""
    started_at = perf_counter()
    try:
        summary = asyncio.run(_run_smoke())
    except Exception as exc:
        # 失败输出也只暴露异常类型，不输出可能携带实现细节的异常文本。
        print(
            json.dumps(
                {
                    "ok": False,
                    "error_type": type(exc).__name__,
                    "elapsed_ms": _elapsed_ms(started_at),
                }
            )
        )
        return 1

    print(json.dumps(summary))
    return 0 if summary["ok"] is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
