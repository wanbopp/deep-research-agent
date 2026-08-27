"""业务 ChatSession 列表缓存的应用层规则.

本模块只负责列表缓存的 key 语义和尽力失效策略，不导入 Redis、SQLAlchemy
或 FastAPI。创建服务与删除协调器复用同一函数，避免两处 key 规则随时间漂移。
"""

from typing import Literal
from uuid import UUID

from app.core.logging import logger
from app.services.cache import Cache, CacheUnavailableError, build_cache_key

CHAT_SESSION_LIST_CACHE_NAMESPACE = "chat_session_list"
CHAT_SESSION_LIST_CACHE_VERSION = "v1"

ChatSessionCacheInvalidationReason = Literal[
    "created",
    "deleting",
    "invalid_payload",
    "title_generated",
]


def build_chat_session_list_cache_key(user_id: UUID) -> str:
    """为一个可信用户构造列表缓存 key.

    Args:
        user_id: 来自认证链或 owner-scoped 数据库行的可信 UUID，不能使用请求
            body、Prompt、模型输出或工具参数替代。

    Returns:
        只包含固定前缀、版本、namespace 和 SHA-256 摘要的稳定 key。

    Notes:
        用户 UUID 不以原文进入 Redis key。摘要只减少运维观察面中的身份暴露，
        真正的用户隔离仍依赖可信 user_id 和 Repository 的 owner-scoped SQL。
    """
    return build_cache_key(
        namespace=CHAT_SESSION_LIST_CACHE_NAMESPACE,
        version=CHAT_SESSION_LIST_CACHE_VERSION,
        identity=str(user_id),
    )


async def invalidate_chat_session_list_cache(
    cache: Cache,
    *,
    user_id: UUID,
    reason: ChatSessionCacheInvalidationReason,
) -> None:
    """尽力删除一个用户的列表缓存，不回滚已提交业务事务.

    Args:
        cache: 应用层缓存协议。production 注入 RedisCache，确定性测试可注入
            InMemoryCache 或明确失败的替代实现。
        user_id: 需要失效列表缓存的可信用户 UUID。
        reason: 受控失效原因，只用于不含身份和 key 的结构化日志。

    Returns:
        没有返回值。缓存删除成功或缓存暂时不可用都正常结束。

    Notes:
        create 或 deleting 事务到达这里时已经提交。Redis 只是性能副本，不能因
        失效失败撤销 PostgreSQL 事实；短 TTL 会限制最坏陈旧窗口。adapter 已记录
        脱敏驱动错误，这里再记录业务阶段，便于区分创建、删除和损坏值清理。
    """
    key = build_chat_session_list_cache_key(user_id)
    try:
        await cache.delete(key)
    except CacheUnavailableError:
        logger.warning(
            "chat_session_list_cache_invalidation_skipped",
            reason=reason,
        )


__all__ = [
    "CHAT_SESSION_LIST_CACHE_NAMESPACE",
    "CHAT_SESSION_LIST_CACHE_VERSION",
    "build_chat_session_list_cache_key",
    "invalidate_chat_session_list_cache",
]
