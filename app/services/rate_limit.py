"""应用级请求限流规则、结果与协议.

本模块只描述“某个可信身份能否继续执行一次受保护操作”，不导入 Redis、
FastAPI 或任何具体 HTTP 类型。这样 API dependency 可以负责身份来源，基础设施
适配器可以负责原子计数，而高成本 Agent 路由只接收最终的允许或拒绝结果。
"""

from dataclasses import dataclass
from hashlib import sha256
import re
from typing import Protocol
import unicodedata


_RATE_LIMIT_KEY_PREFIX = "deep-research:rate-limit"
_SAFE_POLICY_NAME = re.compile(r"^[a-z][a-z0-9_-]{0,63}$")


class RateLimitExceededError(RuntimeError):
    """表示 Redis 已确认当前身份在窗口内耗尽额度."""

    def __init__(self, *, retry_after_seconds: int) -> None:
        """创建可安全映射为 HTTP 429 的应用异常.

        Args:
            retry_after_seconds: 客户端至少等待多少秒再重试。该值来自 Redis 中
                当前计数窗口的剩余 TTL，而不是 Web 进程自己的时钟。

        Raises:
            ValueError: 等待时间不是正整数。
        """
        if retry_after_seconds <= 0:
            raise ValueError("retry_after_seconds must be greater than 0")

        self.retry_after_seconds = retry_after_seconds
        super().__init__("Rate limit exceeded")


class RateLimitBackendUnavailableError(RuntimeError):
    """表示共享后端无法可靠判断当前额度.

    该错误和“已超限”不同：Redis 故障并不能证明用户耗尽额度。高成本 Agent
    入口应把它映射为 503 并 fail-closed，避免绕过限流继续产生模型费用。
    """

    def __init__(self) -> None:
        """创建不包含 Redis 地址、key 或驱动文本的稳定异常."""
        super().__init__("Rate limit backend is unavailable")


class RateLimitIdentityUnavailableError(RuntimeError):
    """表示请求边界无法建立可用于限流的可信身份."""

    def __init__(self) -> None:
        """创建可安全映射为 503 的身份边界错误."""
        super().__init__("Rate limit identity is unavailable")


@dataclass(frozen=True, slots=True)
class RateLimitPolicy:
    """描述一个由应用代码控制的固定窗口限流策略.

    Attributes:
        name: 稳定的小写策略名，只用于区分认证入口和 Agent 入口等用途。
        limit: 单个身份在一个窗口内最多允许的请求数。
        window_seconds: 固定窗口长度，单位为秒。
    """

    name: str
    limit: int
    window_seconds: int

    def __post_init__(self) -> None:
        """在任何 Redis 命令发送前拒绝不安全策略配置."""
        if not _SAFE_POLICY_NAME.fullmatch(self.name):
            raise ValueError(
                "name must start with a lowercase letter and contain only "
                "lowercase letters, digits, underscores, or hyphens"
            )
        if self.limit <= 0:
            raise ValueError("limit must be greater than 0")
        if self.window_seconds <= 0:
            raise ValueError("window_seconds must be greater than 0")


@dataclass(frozen=True, slots=True)
class RateLimitPolicies:
    """集中保存应用当前启用的两类限流策略."""

    auth: RateLimitPolicy
    agent: RateLimitPolicy


@dataclass(frozen=True, slots=True)
class RateLimitDecision:
    """返回一次原子计数后的限流判断.

    Attributes:
        allowed: 本次请求是否仍在额度内。
        limit: 当前策略的窗口上限。
        remaining: 本次计数后仍可使用的次数，最小为零。
        retry_after_seconds: 当前 Redis 窗口还剩多少秒。
    """

    allowed: bool
    limit: int
    remaining: int
    retry_after_seconds: int


class RateLimiter(Protocol):
    """定义应用服务所依赖的共享限流能力."""

    async def acquire(
        self,
        *,
        policy: RateLimitPolicy,
        identity: str,
    ) -> RateLimitDecision:
        """为可信身份原子消费一次请求额度.

        Args:
            policy: 由应用启动配置构造的受控策略，route 不能动态创建它。
            identity: API 边界建立的内部身份原文，例如规范化 IP 或可信 user_id。
                适配器只能把其摘要写入 Redis key。

        Returns:
            本次消费后的允许状态、剩余额度和窗口剩余时间。

        Raises:
            ValueError: identity 标准化后为空。
            RateLimitBackendUnavailableError: 共享后端无法安全完成原子判断。
        """
        ...


async def enforce_rate_limit(
    limiter: RateLimiter,
    *,
    policy: RateLimitPolicy,
    identity: str,
) -> RateLimitDecision:
    """消费额度，并把明确拒绝转换为统一应用异常.

    Args:
        limiter: 由 lifespan 发布的共享限流适配器。
        policy: 当前入口使用的固定策略。
        identity: 已在可信边界建立的内部身份原文。

    Returns:
        允许时返回完整判断，便于未来输出剩余额度响应头。

    Raises:
        RateLimitExceededError: Redis 已确认本次请求超过窗口额度。
        RateLimitBackendUnavailableError: 后端无法判断额度，异常原样传播。
    """
    decision = await limiter.acquire(policy=policy, identity=identity)
    if not decision.allowed:
        raise RateLimitExceededError(
            retry_after_seconds=decision.retry_after_seconds,
        )
    return decision


def build_rate_limit_key(
    *,
    policy_name: str,
    identity: str,
) -> str:
    """构造不泄漏原始身份的版本化 Redis key.

    Args:
        policy_name: 受控的策略名，例如 ``auth`` 或 ``agent``。
        identity: 来自可信边界的规范化身份原文。

    Returns:
        形如 ``deep-research:rate-limit:v1:<policy>:<sha256>`` 的安全 key。

    Raises:
        ValueError: policy name 不安全，或 identity 标准化后为空。

    Security:
        摘要只解决 Redis key 可观察性问题，不提供授权。认证 Agent 入口仍必须先
        验证 token 并从数据库建立可信 user_id，不能把客户端 thread_id 当身份。
    """
    if not _SAFE_POLICY_NAME.fullmatch(policy_name):
        raise ValueError("policy_name is not a safe key segment")

    normalized_identity = unicodedata.normalize("NFC", identity).strip()
    if not normalized_identity:
        raise ValueError("identity must not be empty")

    digest = sha256(normalized_identity.encode("utf-8")).hexdigest()
    return f"{_RATE_LIMIT_KEY_PREFIX}:v1:{policy_name}:{digest}"


__all__ = [
    "RateLimitBackendUnavailableError",
    "RateLimitDecision",
    "RateLimitExceededError",
    "RateLimitIdentityUnavailableError",
    "RateLimitPolicies",
    "RateLimitPolicy",
    "RateLimiter",
    "build_rate_limit_key",
    "enforce_rate_limit",
]
