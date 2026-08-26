"""基于进程内存或共享 Redis client 的缓存适配器.

应用层契约位于 :mod:`app.services.cache`。本模块负责把该契约翻译为具体
基础设施操作，同时避免业务 service 直接依赖 redis-py 类型。
"""

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from time import monotonic
from typing import NoReturn

from redis.asyncio import Redis
from redis.exceptions import RedisError

from app.core.logging import logger
from app.services.cache import CacheUnavailableError


@dataclass(frozen=True, slots=True)
class _MemoryCacheEntry:
    """保存一个不透明字符串值及其单调时钟过期点."""

    value: str
    expires_at: float


class InMemoryCache:
    """在单个 Python 进程中实现 Cache 协议.

    该适配器适用于确定性测试和本地单进程场景。它不能替代生产 Redis，
    因为每个 worker 都拥有独立字典，彼此无法看到对方的缓存内容。
    """

    def __init__(
        self,
        *,
        clock: Callable[[], float] = monotonic,
    ) -> None:
        """创建空缓存，并允许注入单调时钟.

        Args:
            clock: 返回持续递增的秒数。生产环境使用 :func:`time.monotonic`；
                测试可以注入手动推进的时钟，无需真实等待即可观察 TTL。

        Notes:
            异步锁保护“读取、检查过期、删除”这一完整序列。如果没有锁，并发
            task 可能在检查 deadline 与删除条目之间读取或覆盖同一个条目。
        """
        self._clock = clock
        self._entries: dict[str, _MemoryCacheEntry] = {}
        self._lock = asyncio.Lock()

    async def get(self, key: str) -> str | None:
        """返回仍有效的值；不存在或已过期时返回 ``None``.

        Args:
            key: 应用层构造的安全缓存键。

        Returns:
            TTL 有效时返回不透明字符串，否则返回 ``None``。

        Notes:
            过期条目在读取时惰性删除。这个小型确定性适配器无需后台清理任务；
            否则还要额外管理后台任务的启动、停止和异常生命周期。
        """
        async with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return None

            if entry.expires_at <= self._clock():
                self._entries.pop(key, None)
                return None

            return entry.value

    async def set(self, key: str, value: str, *, ttl_seconds: int) -> None:
        """保存值直到单调时钟 TTL 截止点.

        Args:
            key: 应用层构造的安全缓存键。
            value: 已序列化的数据；适配器只把它当作不透明字符串。
            ttl_seconds: 以秒为单位的正数生存时间。

        Raises:
            ValueError: ``ttl_seconds`` 为零或负数。
        """
        _validate_ttl(ttl_seconds)
        expires_at = self._clock() + ttl_seconds

        async with self._lock:
            self._entries[key] = _MemoryCacheEntry(
                value=value,
                expires_at=expires_at,
            )

    async def delete(self, key: str) -> None:
        """从当前进程幂等删除一个缓存条目.

        Args:
            key: 应用层构造的安全缓存键。

        删除不存在的 key 仍视为成功。这样符合应用层协议，也不会向业务层
        泄漏字典或 Redis 的底层删除计数。
        """
        async with self._lock:
            self._entries.pop(key, None)


class RedisCache:
    """使用 lifespan 所有的 Redis client 实现 Cache 协议.

    适配器只借用 client，故意不提供 ``close`` 方法。FastAPI lifespan 是共享
    连接池的唯一所有者，也只有它能在 shutdown 阶段关闭 client。
    """

    def __init__(self, redis_client: Redis) -> None:
        """保存共享异步 Redis client 的引用.

        Args:
            redis_client: 由 ``ApplicationResources`` 持有且配置了
                ``decode_responses=True`` 的 client。构造过程不会访问网络。
        """
        self._redis_client = redis_client

    async def get(self, key: str) -> str | None:
        """从 Redis 读取已经解码的字符串.

        Args:
            key: 应用层构造的安全缓存键。

        Returns:
            命中时返回解码后的字符串；未命中或过期时返回 ``None``。

        Raises:
            CacheUnavailableError: Redis 操作失败，或共享 client 未配置字符串解码。
        """
        try:
            value = await self._redis_client.get(key)
        except RedisError as error:
            _raise_cache_unavailable(operation="get", error=error)

        if value is None:
            return None
        if not isinstance(value, str):
            # production factory 承诺 decode_responses=True。若这里返回 bytes，
            # 就会静默破坏 Cache[str] 边界；因此把错误配置视为后端不可用，
            # 不能让 bytes 继续流入业务 service。
            logger.warning(
                "cache_backend_operation_failed",
                operation="get",
                error_type="UnexpectedResponseType",
            )
            raise CacheUnavailableError

        return value

    async def set(self, key: str, value: str, *, ttl_seconds: int) -> None:
        """把值和 Redis 过期时间作为一个原子操作写入.

        Args:
            key: 应用层构造的安全缓存键。
            value: 已序列化的应用数据。
            ttl_seconds: 以秒为单位的正数生存时间。

        Raises:
            ValueError: ``ttl_seconds`` 为零或负数；发送 Redis 命令前即拒绝。
            CacheUnavailableError: Redis 拒绝写入或执行写入失败。

        Notes:
            ``SET key value EX seconds`` 是一条原子命令。如果拆成 ``SET`` 和
            ``EXPIRE``，进程或连接在两条命令之间失败时，可能留下永不过期的旧值。
        """
        _validate_ttl(ttl_seconds)

        try:
            stored = await self._redis_client.set(
                name=key,
                value=value,
                ex=ttl_seconds,
            )
        except RedisError as error:
            _raise_cache_unavailable(operation="set", error=error)

        if stored is not True:
            logger.warning(
                "cache_backend_operation_failed",
                operation="set",
                error_type="WriteRejected",
            )
            raise CacheUnavailableError

    async def delete(self, key: str) -> None:
        """幂等删除一个 Redis 条目.

        Args:
            key: 应用层构造的安全缓存键。

        Raises:
            CacheUnavailableError: Redis 无法完成缓存失效。

        redis-py 返回的整数删除计数会被刻意忽略。业务 service 只关心命令是否
        安全完成，不存在的 key 不属于应用错误。
        """
        try:
            await self._redis_client.delete(key)
        except RedisError as error:
            _raise_cache_unavailable(operation="delete", error=error)


def _validate_ttl(ttl_seconds: int) -> None:
    """拒绝无法产生有效缓存条目的 TTL.

    Args:
        ttl_seconds: 请求的缓存生存时间，单位为整数秒。

    Raises:
        ValueError: ``ttl_seconds`` 为零或负数。
    """
    if ttl_seconds <= 0:
        raise ValueError("ttl_seconds must be greater than 0")


def _raise_cache_unavailable(*, operation: str, error: RedisError) -> NoReturn:
    """记录脱敏 Redis 故障并抛出稳定的应用层异常.

    Args:
        operation: 固定适配器操作名：get、set 或 delete。
        error: redis-py 异常；这里只读取异常类名。

    Raises:
        CacheUnavailableError: 始终抛出。这里禁止异常链，避免上层随后记录
            traceback 时泄漏地址、原始 key 或 provider 文本。
    """
    logger.warning(
        "cache_backend_operation_failed",
        operation=operation,
        error_type=type(error).__name__,
    )
    raise CacheUnavailableError from None


__all__ = ["InMemoryCache", "RedisCache"]
