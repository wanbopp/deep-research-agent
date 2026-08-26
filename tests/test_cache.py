"""Cache boundary and key-construction checks."""

from typing import cast
from unittest.mock import AsyncMock

import pytest
from redis.asyncio import Redis
from redis.exceptions import RedisError

from app.infrastructure.cache import InMemoryCache, RedisCache
from app.services.cache import CacheUnavailableError, build_cache_key


class _ManualClock:
    """Provide deterministic monotonic time for TTL checks."""

    def __init__(self) -> None:
        self.current = 100.0

    def __call__(self) -> float:
        """Return the current simulated monotonic timestamp."""
        return self.current

    def advance(self, seconds: float) -> None:
        """Move simulated time forward without slowing down the test."""
        self.current += seconds


def test_cache_key_is_stable_versioned_and_secret_safe() -> None:
    """One focused test covers the security and isolation properties of cache keys."""
    sensitive_identity = " user@example.com:private search text "
    base_key = build_cache_key(
        namespace="chat_session_list",
        version="v1",
        identity=sensitive_identity,
    )

    # Stable normalization prevents harmless boundary whitespace or equivalent
    # Unicode representation from creating duplicate cache entries.
    equivalent_key = build_cache_key(
        namespace="chat_session_list",
        version="v1",
        identity=sensitive_identity.strip(),
    )
    assert equivalent_key == base_key

    # Every protocol dimension isolates the entry.  A version bump is therefore a
    # cheap invalidation boundary when serialized data becomes incompatible.
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

    # Unsafe dynamic segments and identities are rejected before they can become
    # ambiguous or reveal raw data through the Redis key namespace.
    invalid_arguments = (
        {"namespace": "", "version": "v1", "identity": "owner"},
        {"namespace": "User:List", "version": "v1", "identity": "owner"},
        {"namespace": "user_list", "version": "", "identity": "owner"},
        {"namespace": "user_list", "version": "v1", "identity": "   "},
    )
    for arguments in invalid_arguments:
        with pytest.raises(ValueError):
            build_cache_key(**arguments)

    # The application-facing error is intentionally stable and contains no backend
    # address, raw key, cached value, or provider exception details.
    assert str(CacheUnavailableError()) == "Cache backend is unavailable"


@pytest.mark.anyio
async def test_in_memory_cache_supports_hit_expiry_and_idempotent_delete() -> None:
    """The deterministic adapter should implement the complete Cache lifecycle."""
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

    # Exactly reaching the deadline counts as expired.  The adapter also removes
    # the stale entry while returning the same None used for an ordinary miss.
    clock.advance(2)
    assert await cache.get(key) is None

    await cache.set(key, "replacement", ttl_seconds=10)
    await cache.delete(key)
    await cache.delete(key)
    assert await cache.get(key) is None

    # Invalid TTL is a caller error and is rejected before mutating storage.
    with pytest.raises(ValueError, match="ttl_seconds"):
        await cache.set(key, "invalid", ttl_seconds=0)


@pytest.mark.anyio
async def test_redis_cache_translates_driver_failure_without_leaking_details() -> None:
    """The one deterministic failure check protects the adapter's security boundary."""
    redis_mock = AsyncMock(spec=Redis)
    redis_mock.get.side_effect = RedisError("sensitive backend diagnostic with host, raw key, and provider text")
    cache = RedisCache(cast(Redis, redis_mock))

    with pytest.raises(CacheUnavailableError) as exc_info:
        await cache.get("sensitive-raw-key")

    assert str(exc_info.value) == "Cache backend is unavailable"
    assert exc_info.value.__cause__ is None

    # RedisCache validates TTL before sending a command.  This is a caller error,
    # not an infrastructure outage, and therefore remains ValueError.
    with pytest.raises(ValueError, match="ttl_seconds"):
        await cache.set("key", "value", ttl_seconds=0)
    redis_mock.set.assert_not_called()
