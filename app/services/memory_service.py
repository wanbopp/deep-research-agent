"""长期记忆的应用策略、缓存和失败降级编排."""

from uuid import UUID

from app.core.logging import logger
from app.schemas.memory import (
    MemoryCreate,
    MemoryItem,
    MemoryQuery,
    MemorySearchResult,
    MemorySearchStatus,
)
from app.services.cache import Cache
from app.services.memory import MemoryStore, MemoryUnavailableError
from app.services.memory_cache import MemorySearchCache
from app.services.memory_policy import (
    CredentialMemoryPolicy,
    MemoryContentPolicy,
)

MEMORY_UNAVAILABLE_ERROR_CODE = "MEMORY_UNAVAILABLE"


class MemoryService:
    """在 MemoryStore 之上实现安全、可观察的长期记忆用例.

    Service 不知道 SQL、pgvector、OpenAI SDK 或 redis-py。它只组合应用层
    ``MemoryStore``、``Cache`` 和内容策略，因此未来 Agent node 依赖的是稳定
    业务入口，而不是一组基础设施客户端。
    """

    def __init__(
        self,
        store: MemoryStore,
        cache: Cache,
        *,
        search_cache_ttl_seconds: int,
        generation_ttl_seconds: int,
        content_policy: MemoryContentPolicy | None = None,
    ) -> None:
        """构造可跨请求共享的无状态 MemoryService.

        Args:
            store: 负责真实 Embedding 和 owner-scoped PostgreSQL 操作的存储协议。
            cache: Redis 或进程内 Cache 协议；故障时搜索允许回源。
            search_cache_ttl_seconds: 单个查询结果允许保存的秒数。
            generation_ttl_seconds: 用户缓存 namespace 版本的秒数。
            content_policy: 写入前的本地敏感内容策略；默认使用高精度 credential
                规则。策略不应执行网络请求或保存候选正文。

        Raises:
            ValueError: 缓存 TTL 不满足正数和先后关系。
        """
        self._store = store
        self._content_policy = content_policy or CredentialMemoryPolicy()
        self._search_cache = MemorySearchCache(
            cache,
            search_ttl_seconds=search_cache_ttl_seconds,
            generation_ttl_seconds=generation_ttl_seconds,
        )

    async def search(
        self,
        *,
        user_id: UUID,
        query: MemoryQuery,
    ) -> MemorySearchResult:
        """优先读取缓存，必要时回源，并显式区分正常空结果和故障.

        Args:
            user_id: 认证链提供的可信用户 UUID。它独立于 query 传入，不能由
                Prompt、模型输出或客户端查询字段覆盖。
            query: 已校验的文本、可选 kind 集合和结果上限。

        Returns:
            ``available`` 表示 Store 正常完成，即使 items 为空；``degraded``
            表示真实 MemoryStore 不可用，Agent 可选择不带长期记忆继续工作。

        Notes:
            Redis 故障只造成 cache miss。MemoryStore 故障则必须进入 degraded，
            不能返回普通空元组掩盖系统状态。
        """
        lookup = await self._search_cache.lookup(
            user_id=user_id,
            query=query,
        )
        if lookup.is_hit:
            return MemorySearchResult(items=lookup.items or ())

        try:
            items = await self._store.search(user_id=user_id, query=query)
        except MemoryUnavailableError as exc:
            return self._degraded_search_result(exc)

        # 即使 adapter 承诺 owner-scoped，Service 仍在跨层边界复核。缓存污染、
        # adapter 回归或错误实现都不能把其他用户的记忆交给 Agent。
        if len(items) > query.limit or any(item.user_id != user_id for item in items):
            logger.error(
                "memory_store_contract_violated",
                operation="search",
            )
            return self._degraded_search_result(MemoryUnavailableError())

        await self._search_cache.store_if_current(
            user_id=user_id,
            lookup=lookup,
            items=items,
        )
        return MemorySearchResult(items=items)

    async def add(
        self,
        *,
        user_id: UUID,
        memory: MemoryCreate,
    ) -> MemoryItem:
        """先执行本地敏感策略，再持久化并切换缓存 generation.

        Args:
            user_id: 认证链建立的可信用户 UUID。
            memory: 不含用户归属和向量的严格候选记忆。

        Returns:
            Store 持久化后的权威 ``MemoryItem``。

        Raises:
            MemoryRejectedError: 候选高度疑似包含 credential；此时 Store 和真实
                Embedding provider 都不会被调用。
            MemorySourceNotFoundError: 来源会话不可访问。
            MemoryUnavailableError: 真实 provider 或 PostgreSQL 写入失败，或 Store
                返回的数据违反应用契约。
        """
        # 内容策略必须位于第一条外部 I/O 之前。否则即使数据库最终拒绝，秘密也
        # 已经发送给 Embedding provider，无法通过事务 rollback 撤回。
        self._content_policy.ensure_allowed(memory.content)

        item = await self._store.add(user_id=user_id, memory=memory)
        item_matches_request = (
            item.user_id == user_id
            and item.content == memory.content
            and item.kind is memory.kind
            and item.source_thread_id == memory.source_thread_id
        )
        if not item_matches_request:
            logger.error(
                "memory_store_contract_violated",
                operation="add",
            )
            raise MemoryUnavailableError() from None

        # PostgreSQL 已提交成功。缓存只是可重建副本；rotate 内部会吞掉并记录
        # CacheUnavailableError，绝不能为了 Redis 故障撤销权威记忆。 rotate换门牌号
        await self._search_cache.rotate(user_id=user_id, reason="added")
        return item

    async def delete(
        self,
        *,
        user_id: UUID,
        memory_id: UUID,
    ) -> None:
        """幂等删除 owner-scoped 记忆，并尽力使旧查询缓存失效.

        Args:
            user_id: 认证链提供的可信用户 UUID。
            memory_id: 待删除记忆 UUID。

        Raises:
            MemoryUnavailableError: PostgreSQL 无法完成删除；缓存故障不会导致抛出。
        """
        await self._store.delete(user_id=user_id, memory_id=memory_id)
        await self._search_cache.rotate(user_id=user_id, reason="deleted")

    @staticmethod
    def _degraded_search_result(
        error: MemoryUnavailableError,
    ) -> MemorySearchResult:
        """记录脱敏故障类型并返回不缓存的显式降级结果."""
        logger.warning(
            "memory_search_degraded",
            error_type=type(error).__name__,
        )
        return MemorySearchResult(
            items=(),
            status=MemorySearchStatus.DEGRADED,
            error_code=MEMORY_UNAVAILABLE_ERROR_CODE,
        )


__all__ = ["MEMORY_UNAVAILABLE_ERROR_CODE", "MemoryService"]
