"""使用共享 Redis client 实现跨 worker 原子请求限流."""

from typing import NoReturn

from redis.asyncio import Redis
from redis.exceptions import RedisError

from app.core.logging import logger
from app.services.rate_limit import (
    RateLimitBackendUnavailableError,
    RateLimitDecision,
    RateLimitPolicy,
    build_rate_limit_key,
)


# 一段 Lua 脚本在 Redis 单线程执行期间不可被其他命令插入，因此"递增、首次设置
# TTL、读取剩余 TTL"属于同一个原子判断。若拆成 Python 的 INCR + EXPIRE，进程在
# 两条命令之间退出时会留下永不过期计数，多个 worker 也可能观察到不一致窗口。
#
# 固定窗口计数原理：
#   1. INCRBY key 1  → 原子递增计数器；key 不存在时 Redis 自动从 0 开始，返回 1。
#   2. 如果 current == 1（本轮窗口第一次请求），用 EXPIRE 设置窗口过期时间。
#      后续请求不再重复设置 TTL，保证窗口边界由第一次请求锚定。
#   3. TTL key       → 读取 key 剩余存活秒数。
#      如果 TTL < 0（极端情况：key 存在但 TTL 丢失），补设一次 TTL 并返回窗口长度。
#   4. 返回 {current, ttl} 给 Python 层判定是否超限。
#
# 判定逻辑在 Python 层：current <= limit 则允许，否则拒绝。
# 窗口剩余时间 ttl 直接来自 Redis，而非 Python 进程时钟，因此跨 worker 一致。
_FIXED_WINDOW_SCRIPT = """
local current = redis.call('INCRBY', KEYS[1], 1)
if current == 1 then
    redis.call('EXPIRE', KEYS[1], ARGV[1])
end

local ttl = redis.call('TTL', KEYS[1])
if ttl < 0 then
    redis.call('EXPIRE', KEYS[1], ARGV[1])
    ttl = tonumber(ARGV[1])
end

return {current, ttl}
"""


class RedisRateLimiter:
    """借用 lifespan 共享 Redis client 实现固定窗口限流.

    本适配器不提供 ``close``：它不拥有连接池，shutdown 仍由
    ``ApplicationResources`` 统一关闭共享 client。
    """

    def __init__(self, redis_client: Redis) -> None:
        """保存共享异步 Redis client.

        Args:
            redis_client: lifespan 创建并持有的 redis-py 异步 client。构造过程
                不发送网络请求，也不会复制连接池。
        """
        self._redis_client = redis_client

    async def acquire(
        self,
        *,
        policy: RateLimitPolicy,
        identity: str,
    ) -> RateLimitDecision:
        """在 Redis 中原子消费一次固定窗口额度.

        Args:
            policy: 应用启动阶段创建的受控策略。
            identity: API 层建立的可信内部身份。原文只在当前进程短暂存在，写入
                Redis 前会由 ``build_rate_limit_key`` 转换为 SHA-256 摘要。

        Returns:
            计数后的允许状态、剩余额度与 Redis 窗口 TTL。

        Raises:
            ValueError: identity 为空或 policy name 不安全。
            RateLimitBackendUnavailableError: Redis 命令失败或返回不可解释结果。
        """
        key = build_rate_limit_key(
            policy_name=policy.name,
            identity=identity,
        )
        # key 形如 deep-research:rate-limit:v1:auth:<sha256(IP)> 或
        #      deep-research:rate-limit:v1:agent:<sha256(user_id)>
        # 不同 policy name + 不同 identity 产生完全隔离的 Redis key，互不干扰。

        try:
            # eval() 将 Lua 脚本发送到 Redis 原子执行：
            #   KEYS[1] = key（限流计数器）
            #   ARGV[1] = policy.window_seconds（窗口长度，单位秒）
            # 返回值是 Lua 脚本的 {current, ttl} 两元素列表。
            result = await self._redis_client.eval(
                _FIXED_WINDOW_SCRIPT,
                1,
                key,
                policy.window_seconds,
            )
        except RedisError as error:
            # Redis 故障时 fail-closed：不能证明有额度就拒绝，防止绕过限流产生模型费用。
            _raise_rate_limit_unavailable(error)

        current, ttl = _parse_script_result(result)
        # 判定逻辑：
        #   current <= limit  → allowed=True（额度内）
        #   current > limit   → allowed=False（已超限，返回 429）
        # remaining 是窗口内还能用多少次；retry_after 是窗口还剩多少秒，
        # 客户端可据此设置 Retry-After header。
        return RateLimitDecision(
            allowed=current <= policy.limit,
            limit=policy.limit,
            remaining=max(policy.limit - current, 0),
            retry_after_seconds=max(ttl, 1),
        )


def _parse_script_result(result: object) -> tuple[int, int]:
    """把 Lua 返回值收窄为正整数计数和 TTL.

    Args:
        result: redis-py 返回的动态结果，正常形态为两个整数的 list。

    Returns:
        ``(current, ttl)``；二者都大于零。

    Raises:
        RateLimitBackendUnavailableError: 返回结构无法证明当前额度状态。

    Notes:
        这里选择 fail-closed。不可解释响应不能被当作“仍有额度”，否则高成本
        Agent 请求会在基础设施异常期间绕过保护继续调用模型。
    """
    if not isinstance(result, (list, tuple)) or len(result) != 2:
        _raise_unexpected_response()

    current = result[0]
    ttl = result[1]
    if not isinstance(current, int) or not isinstance(ttl, int) or current <= 0 or ttl <= 0:
        _raise_unexpected_response()

    return current, ttl


def _raise_unexpected_response() -> NoReturn:
    """记录脱敏响应类型错误并抛出稳定后端异常."""
    logger.warning(
        "rate_limit_backend_operation_failed",
        operation="acquire",
        error_type="UnexpectedResponseType",
    )
    raise RateLimitBackendUnavailableError


def _raise_rate_limit_unavailable(error: RedisError) -> NoReturn:
    """把 redis-py 异常转换为不泄漏基础设施细节的应用异常.

    Args:
        error: Redis 驱动异常；日志只读取异常类名。

    Raises:
        RateLimitBackendUnavailableError: 始终抛出，并刻意删除异常链，避免上层
            traceback 泄漏地址、认证信息或完整 Redis key。
    """
    logger.warning(
        "rate_limit_backend_operation_failed",
        operation="acquire",
        error_type=type(error).__name__,
    )
    raise RateLimitBackendUnavailableError from None


__all__ = ["RedisRateLimiter"]
