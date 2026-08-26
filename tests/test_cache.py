"""缓存边界与缓存键构造测试."""

from typing import cast
from unittest.mock import AsyncMock

import pytest
from redis.asyncio import Redis
from redis.exceptions import RedisError

from app.infrastructure.cache import InMemoryCache, RedisCache
from app.services.cache import CacheUnavailableError, build_cache_key


class _ManualClock:
    """为 TTL 测试提供可手动推进的确定性单调时钟."""

    def __init__(self) -> None:
        self.current = 100.0

    def __call__(self) -> float:
        """返回当前模拟的单调时间戳."""
        return self.current

    def advance(self, seconds: float) -> None:
        """推进模拟时间，不让测试因真实等待而变慢."""
        self.current += seconds


def test_cache_key_is_stable_versioned_and_secret_safe() -> None:
    """用一个聚焦测试覆盖缓存键的安全与隔离属性."""
    sensitive_identity = " user@example.com:private search text "
    base_key = build_cache_key(
        namespace="chat_session_list",
        version="v1",
        identity=sensitive_identity,
    )

    # 稳定规范化防止无意义的边界空格或等价 Unicode 表示生成重复缓存条目。
    equivalent_key = build_cache_key(
        namespace="chat_session_list",
        version="v1",
        identity=sensitive_identity.strip(),
    )
    assert equivalent_key == base_key

    # 每个协议维度都会隔离缓存。当序列化数据不兼容时，提升版本号即可形成
    # 低成本失效边界，无需扫描并删除全部旧 key。
    assert (
        build_cache_key(
            namespace="chat_session_detail",
            version="v1",
            identity=sensitive_identity,
        )
        != base_key
    )
    assert (
        build_cache_key(
            namespace="chat_session_list",
            version="v2",
            identity=sensitive_identity,
        )
        != base_key
    )
    assert (
        build_cache_key(
            namespace="chat_session_list",
            version="v1",
            identity="another-user@example.com:private search text",
        )
        != base_key
    )

    assert base_key.startswith("deep-research:cache:v1:chat_session_list:")
    assert len(base_key.rsplit(":", maxsplit=1)[-1]) == 64
    assert "user@example.com" not in base_key
    assert "private search text" not in base_key

    # 不安全的动态段和空 identity 必须在构造阶段拒绝，避免 key 语义含糊或
    # 通过 Redis key namespace 暴露原始数据。
    invalid_arguments = (
        {"namespace": "", "version": "v1", "identity": "owner"},
        {"namespace": "User:List", "version": "v1", "identity": "owner"},
        {"namespace": "user_list", "version": "", "identity": "owner"},
        {"namespace": "user_list", "version": "v1", "identity": "   "},
    )
    for arguments in invalid_arguments:
        with pytest.raises(ValueError):
            build_cache_key(**arguments)

    # 面向应用层的错误文本刻意保持固定，不携带地址、原始 key、缓存值或
    # provider 异常细节。
    assert str(CacheUnavailableError()) == "Cache backend is unavailable"


@pytest.mark.anyio
async def test_in_memory_cache_supports_hit_expiry_and_idempotent_delete() -> None:
    """确定性适配器应覆盖完整的 Cache 生命周期."""
    clock = _ManualClock()
    cache = InMemoryCache(clock=clock)
    key = build_cache_key(
        namespace="cache_test",
        version="v1",
        identity="owner-1",
    )

    assert await cache.get(key) is None

    await cache.set(key, "serialized-value", ttl_seconds=2)
    assert await cache.get(key) == "serialized-value"

    # 恰好到达 deadline 即视为过期。适配器在返回普通 miss 所用的 None 时，
    # 还会同步删除陈旧条目。
    clock.advance(2)
    assert await cache.get(key) is None

    await cache.set(key, "replacement", ttl_seconds=10)
    await cache.delete(key)
    await cache.delete(key)
    assert await cache.get(key) is None

    # 非法 TTL 属于调用方错误，必须在修改存储前拒绝。
    with pytest.raises(ValueError, match="ttl_seconds"):
        await cache.set(key, "invalid", ttl_seconds=0)


@pytest.mark.anyio
async def test_redis_cache_translates_driver_failure_without_leaking_details() -> None:
    """用一个确定性失败检查保护适配器的安全边界."""
    redis_mock = AsyncMock(spec=Redis)
    redis_mock.get.side_effect = RedisError("sensitive backend diagnostic with host, raw key, and provider text")
    cache = RedisCache(cast(Redis, redis_mock))

    with pytest.raises(CacheUnavailableError) as exc_info:
        await cache.get("sensitive-raw-key")

    assert str(exc_info.value) == "Cache backend is unavailable"
    assert exc_info.value.__cause__ is None

    # RedisCache 在发送命令前校验 TTL。它属于调用方错误而非基础设施故障，
    # 因此应保持 ValueError，且不能触发 redis_client.set。
    with pytest.raises(ValueError, match="ttl_seconds"):
        await cache.set("key", "value", ttl_seconds=0)
    redis_mock.set.assert_not_called()
