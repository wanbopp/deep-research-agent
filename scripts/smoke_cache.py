"""使用已配置的真实 Redis 服务验收缓存适配器.

本 smoke 不调用模型，也不输出 Redis 地址、密码、原始 key、缓存值或驱动异常
文本。脚本拥有自己创建的 client，因此负责最终清理；生产 ``RedisCache`` 只借用
lifespan 中的共享 client。
"""

import asyncio
import json
import selectors
from time import perf_counter
from uuid import uuid4

from redis.asyncio import Redis

from app.core.config import settings
from app.infrastructure.cache import RedisCache
from app.services.cache import build_cache_key

ENTRY_TTL_SECONDS = 1
DELETE_TTL_SECONDS = 30
EXPIRY_WAIT_SECONDS = 1.2
TOTAL_TIMEOUT_SECONDS = 8.0


def _elapsed_ms(started_at: float) -> float:
    """返回用于安全摘要的毫秒耗时."""
    return round((perf_counter() - started_at) * 1000, 2)


def _selector_loop_factory() -> asyncio.AbstractEventLoop:
    """创建与项目其他基础设施 smoke 一致的事件循环."""
    return asyncio.SelectorEventLoop(selectors.SelectSelector())


def _create_redis_client() -> Redis:
    """创建一个只由当前 smoke 拥有的惰性 client.

    Returns:
        使用 Git 忽略环境配置并解码字符串响应的异步 client。第一次 Redis
        命令才会真正建立网络连接。
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


async def _exercise_real_cache() -> dict[str, bool | float]:
    """验证 miss、hit、过期、删除、参数校验和 client 所有权.

    Returns:
        只包含行为布尔值和耗时的脱敏映射。

    Raises:
        Exception: 连接或语义错误传播给 ``main``，最终只输出异常类名。
            ``finally`` 仍会尝试清理 key 并关闭 smoke 自己的 client。
    """
    started_at = perf_counter()
    redis_client = _create_redis_client()
    cache = RedisCache(redis_client)

    # 随机 identity 防止并发执行的 smoke 共享同一条目。原始 identity 和生成的
    # 摘要 key 只保留在当前进程中，不进入最终输出。
    key = build_cache_key(
        namespace="cache_smoke",
        version="v1",
        identity=uuid4().hex,
    )
    try:
        async with asyncio.timeout(TOTAL_TIMEOUT_SECONDS):
            await cache.delete(key)
            initial_miss = await cache.get(key) is None

            await cache.set(key, "ttl-value", ttl_seconds=ENTRY_TTL_SECONDS)
            hit_matches = await cache.get(key) == "ttl-value"

            # 这里故意真实等待。与使用手动时钟的 InMemory 测试不同，这一步用于
            # 证明过期行为确实由 Redis 服务端执行。
            await asyncio.sleep(EXPIRY_WAIT_SECONDS)
            expired_to_miss = await cache.get(key) is None

            await cache.set(key, "delete-value", ttl_seconds=DELETE_TTL_SECONDS)
            await cache.delete(key)
            await cache.delete(key)
            delete_is_idempotent = await cache.get(key) is None

            invalid_ttl_rejected = False
            try:
                await cache.set(key, "invalid", ttl_seconds=0)
            except ValueError:
                invalid_ttl_rejected = True

            # RedisCache 没有关闭借用的 client：完成所有适配器操作后，资源所有者
            # 仍然可以直接使用同一个 client。
            borrowed_client_still_usable = bool(await redis_client.ping())

        return {
            "initial_miss": initial_miss,
            "hit_matches": hit_matches,
            "expired_to_miss": expired_to_miss,
            "delete_is_idempotent": delete_is_idempotent,
            "invalid_ttl_rejected": invalid_ttl_rejected,
            "borrowed_client_still_usable": borrowed_client_still_usable,
            "within_total_budget": _elapsed_ms(started_at) <= TOTAL_TIMEOUT_SECONDS * 1000,
            "elapsed_ms": _elapsed_ms(started_at),
        }
    finally:
        # 当前脚本同时拥有测试 key 和 client，因此必须负责两者的清理。适配器
        # 绝不能执行这些资源清理，因为生产环境中的共享 client 属于 lifespan。
        cleanup_errors: list[Exception] = []
        try:
            # 在连接仍可用时先删除 key。若 delete 与 close 并发执行，会在同一个
            # client 内部制造资源关闭竞态。
            await redis_client.delete(key)
        except Exception as error:
            cleanup_errors.append(error)

        try:
            await redis_client.aclose()
        except Exception as error:
            cleanup_errors.append(error)

        # 即使缓存语义全部通过，清理失败也必须让 smoke 失败。详细驱动错误留在
        # 内部，main 最终只公开 RuntimeError 类型。
        if cleanup_errors:
            raise RuntimeError("cache smoke cleanup failed")


def _run_smoke() -> dict[str, object]:
    """运行异步 smoke，并汇总全部布尔检查结果."""
    checks = asyncio.run(
        _exercise_real_cache(),
        loop_factory=_selector_loop_factory,
    )
    boolean_checks = tuple(value for value in checks.values() if isinstance(value, bool))
    return {
        "ok": bool(boolean_checks) and all(boolean_checks),
        **checks,
        "cleanup_ok": True,
    }


def main() -> int:
    """打印一行脱敏 JSON，并返回适合 CI 的退出码."""
    started_at = perf_counter()
    try:
        summary = _run_smoke()
    except Exception as error:
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
