"""限流应用契约与安全 key 的聚焦测试."""

import pytest

from app.services.rate_limit import (
    RateLimitBackendUnavailableError,
    RateLimitDecision,
    RateLimitExceededError,
    RateLimitPolicy,
    build_rate_limit_key,
    enforce_rate_limit,
)


class _SequenceLimiter:
    """按顺序返回预置判断的最小确定性限流器."""

    def __init__(self, decisions: list[RateLimitDecision]) -> None:
        self._decisions = decisions

    async def acquire(
        self,
        *,
        policy: RateLimitPolicy,
        identity: str,
    ) -> RateLimitDecision:
        """返回下一条判断；参数访问用于证明协议签名完整."""
        assert policy.name == "agent"
        assert identity == "user:sensitive-user-id"
        return self._decisions.pop(0)


@pytest.mark.anyio
async def test_rate_limit_contract_is_validated_secret_safe_and_explicit() -> None:
    """用一个测试覆盖配置、key、允许和明确拒绝四个应用边界."""
    policy = RateLimitPolicy(name="agent", limit=2, window_seconds=60)
    identity = "user:sensitive-user-id"
    key = build_rate_limit_key(policy_name=policy.name, identity=identity)

    assert key.startswith("deep-research:rate-limit:v1:agent:")
    assert len(key.rsplit(":", maxsplit=1)[-1]) == 64
    assert identity not in key

    for invalid_policy in (
        {"name": "", "limit": 1, "window_seconds": 60},
        {"name": "Agent:Dynamic", "limit": 1, "window_seconds": 60},
        {"name": "agent", "limit": 0, "window_seconds": 60},
        {"name": "agent", "limit": 1, "window_seconds": 0},
    ):
        with pytest.raises(ValueError):
            RateLimitPolicy(**invalid_policy)

    with pytest.raises(ValueError, match="identity"):
        build_rate_limit_key(policy_name="agent", identity="   ")

    limiter = _SequenceLimiter(
        [
            RateLimitDecision(
                allowed=True,
                limit=2,
                remaining=1,
                retry_after_seconds=60,
            ),
            RateLimitDecision(
                allowed=False,
                limit=2,
                remaining=0,
                retry_after_seconds=37,
            ),
        ]
    )
    allowed = await enforce_rate_limit(
        limiter,
        policy=policy,
        identity=identity,
    )
    assert allowed.remaining == 1

    with pytest.raises(RateLimitExceededError) as exc_info:
        await enforce_rate_limit(
            limiter,
            policy=policy,
            identity=identity,
        )
    assert exc_info.value.retry_after_seconds == 37

    # 两种异常文本保持固定，不包含 identity、Redis 地址、key 或驱动诊断。
    assert str(exc_info.value) == "Rate limit exceeded"
    assert str(RateLimitBackendUnavailableError()) == "Rate limit backend is unavailable"
