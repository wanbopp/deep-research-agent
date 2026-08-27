"""长期记忆查询缓存的 generation、一致性和序列化规则."""

import json
import re
from dataclasses import dataclass
from typing import Literal
from uuid import UUID, uuid4

from pydantic import ValidationError

from app.core.logging import logger
from app.schemas.memory import (
    MemoryItem,
    MemoryQuery,
    MemorySearchResult,
    MemorySearchStatus,
)
from app.services.cache import Cache, CacheUnavailableError, build_cache_key

MEMORY_GENERATION_CACHE_NAMESPACE = "memory_generation"
MEMORY_SEARCH_CACHE_NAMESPACE = "memory_search"
MEMORY_CACHE_VERSION = "v1"

MemoryCacheRotationReason = Literal["added", "deleted"]

_GENERATION_PATTERN = re.compile(r"^[0-9a-f]{32}$")


@dataclass(frozen=True, slots=True)
class MemoryCacheLookup:
    """保存一次查询使用的 generation 快照和缓存结果.

    ``items is None`` 表示 miss，空元组则表示命中了“正常但没有记忆”的缓存值。
    generation/key 为 None 表示 Redis 不可用，本次搜索应直接回源且不写缓存。
    """

    items: tuple[MemoryItem, ...] | None
    generation: str | None
    key: str | None

    @property
    def is_hit(self) -> bool:
        """返回本次 lookup 是否命中了一个经过严格校验的缓存值."""
        return self.items is not None


def build_memory_generation_cache_key(user_id: UUID) -> str:
    """为可信用户构造不暴露原始 UUID 的 generation key."""
    return build_cache_key(
        namespace=MEMORY_GENERATION_CACHE_NAMESPACE,
        version=MEMORY_CACHE_VERSION,
        identity=str(user_id),
    )


def build_memory_search_cache_key(
    *,
    user_id: UUID,
    generation: str,
    query: MemoryQuery,
) -> str:
    """为一个用户 generation 下的确定查询构造哈希 key.

    Args:
        user_id: 认证链提供的可信用户 UUID。
        generation: 当前用户缓存 namespace 的随机版本。
        query: 已校验的查询正文、kind 集合和返回上限。

    Returns:
        不包含原始 user_id、query 或 kind 文本的 SHA-256 缓存 key。
    """
    identity = json.dumps(
        {
            "user_id": str(user_id),
            "generation": generation,
            "text": query.text,
            "kinds": (sorted(kind.value for kind in query.kinds) if query.kinds is not None else None),
            "limit": query.limit,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return build_cache_key(
        namespace=MEMORY_SEARCH_CACHE_NAMESPACE,
        version=MEMORY_CACHE_VERSION,
        identity=identity,
    )


class MemorySearchCache:
    """在应用层实现长期记忆查询缓存和 generation 失效.

    本类只依赖 ``Cache`` Protocol，不导入 redis-py。缓存是 PostgreSQL 的性能
    副本：任何 Redis 故障都允许搜索回源，不能改变 owner-scoped 数据库语义。
    """

    def __init__(
        self,
        cache: Cache,
        *,
        search_ttl_seconds: int,
        generation_ttl_seconds: int,
    ) -> None:
        """保存缓存协议和两个明确的 TTL.

        Args:
            cache: production 使用 RedisCache，确定性测试可使用 InMemoryCache。
            search_ttl_seconds: 单个查询结果的最大陈旧窗口。
            generation_ttl_seconds: 用户 namespace 版本的生存时间。

        Raises:
            ValueError: TTL 非正数，或 generation 比查询结果更早过期。
        """
        if search_ttl_seconds <= 0:
            raise ValueError("search_ttl_seconds must be greater than 0")
        if generation_ttl_seconds < search_ttl_seconds:
            raise ValueError("generation_ttl_seconds must not be shorter than search_ttl_seconds")
        self._cache = cache
        self._search_ttl_seconds = search_ttl_seconds
        self._generation_ttl_seconds = generation_ttl_seconds

    async def lookup(
        self,
        *,
        user_id: UUID,
        query: MemoryQuery,
    ) -> MemoryCacheLookup:
        """读取并严格验证一个用户查询缓存.

        调用链入口。整个 lookup 流程分三步：
          1. 获取/创建 generation（用户的缓存版本号）；
          2. 用 generation + 查询参数构造唯一 key，从 Redis 读取缓存；
          3. 对缓存值做严格校验（格式、owner、数量），任何一项不通过都视为 miss。

        Args:
            user_id: 当前认证用户的可信 UUID。
            query: 已校验的长期记忆查询。

        Returns:
            命中、未命中或缓存不可用的显式 lookup 快照。
        """
        # 第一步：获取当前用户的 generation 版本号。
        # generation 是所有缓存 key 的命名空间前缀——同一个查询在不同 generation
        # 下会产生不同的 key，因此切换 generation 后旧缓存自动失效。
        # 返回 None 表示 Redis 不可用，此时放弃缓存，调用方应直接回源数据库。
        generation = await self._get_or_create_generation(user_id)
        if generation is None:
            return MemoryCacheLookup(items=None, generation=None, key=None)

        # 第二步：用 generation + 用户 ID + 查询参数构造唯一缓存 key。
        # key 内部是 SHA-256 哈希，不暴露原始 user_id 或查询文本。
        # 注意：generation 参与了 key 的构造，所以 generation 一变，key 就完全不同。
        key = build_memory_search_cache_key(
            user_id=user_id,
            generation=generation,
            query=query,
        )
        try:
            payload = await self._cache.get(key)
        except CacheUnavailableError:
            # Redis 在读取瞬间不可用 → fail-open：返回 miss 但不放弃 generation，
            # 调用方回源后仍可尝试 store_if_current 写入。
            self._log_cache_skip(operation="search_get")
            return MemoryCacheLookup(
                items=None,
                generation=generation,
                key=key,
            )

        # 第三步：缓存值严格校验。即使 key 命中，也不能直接信任内容——
        # Redis 中的数据可能被篡改、序列化格式可能变更、owner 可能不匹配。
        if payload is None:
            # 正常 miss：key 存在但无值（首次查询或已过期）。
            return MemoryCacheLookup(
                items=None,
                generation=generation,
                key=key,
            )

        try:
            cached_result = MemorySearchResult.model_validate_json(payload)
        except ValidationError:
            # 缓存值格式损坏（可能是旧版本序列化格式或手动篡改），
            # 删除脏值并视为 miss，下次回源后会写入正确格式。
            await self._reject_cached_value(key=key, reason="invalid_payload")
            return MemoryCacheLookup(
                items=None,
                generation=generation,
                key=key,
            )

        # 安全检查：缓存中的每条记忆必须属于当前用户，且结果数量不超过请求上限。
        # 这是最后一道防线——即使 generation 机制正常，也要防止跨用户数据泄漏。
        owner_matches = all(item.user_id == user_id for item in cached_result.items)
        result_shape_matches = (
            cached_result.status is MemorySearchStatus.AVAILABLE
            and len(cached_result.items) <= query.limit
            and owner_matches
        )
        if not result_shape_matches:
            await self._reject_cached_value(key=key, reason="contract_mismatch")
            return MemoryCacheLookup(
                items=None,
                generation=generation,
                key=key,
            )

        # 全部校验通过：返回缓存命中结果。
        return MemoryCacheLookup(
            items=cached_result.items,
            generation=generation,
            key=key,
        )

    async def store_if_current(
        self,
        *,
        user_id: UUID,
        lookup: MemoryCacheLookup,
        items: tuple[MemoryItem, ...],
    ) -> None:
        """仅在 generation 未变化时缓存刚回源的结果.

        调用链：lookup miss → 回源数据库查询 → store_if_current 写回缓存。
        这一步是"条件写入"——只有 generation 没被并发操作改变时才写入。

        Args:
            user_id: 当前认证用户的可信 UUID。
            lookup: 回源前取得的 generation 快照和目标 key。
            items: MemoryStore 刚返回且已复核 owner 的权威结果。

        Notes:
            若并发 add/delete 已旋转 generation，旧查询结果会被直接放弃，不能写入
            新 namespace。这是避免并发陈旧回填的关键比较。
        """
        # 如果 lookup 阶段 Redis 就不可用（generation/key 为 None），放弃写入。
        if lookup.generation is None or lookup.key is None:
            return

        # 再次读取当前 generation，与 lookup 时保存的快照比较。
        # 如果两者不同，说明在 lookup → 回源 这段时间内，有其他请求执行了
        # add/delete 并触发了 rotate()，generation 已更新。
        # 此时写入旧 key 毫无意义（新查询会用新 generation 构造新 key），直接跳过。
        generation_key = build_memory_generation_cache_key(user_id)
        try:
            current_generation = await self._cache.get(generation_key)
        except CacheUnavailableError:
            self._log_cache_skip(operation="generation_recheck")
            return

        if current_generation != lookup.generation:
            logger.info("memory_cache_store_skipped", reason="generation_changed")
            return

        # generation 一致，安全写入。序列化后存入 Redis，TTL 为短查询缓存时间。
        payload = MemorySearchResult(items=items).model_dump_json()
        try:
            await self._cache.set(
                lookup.key,
                payload,
                ttl_seconds=self._search_ttl_seconds,
            )
        except CacheUnavailableError:
            # 写入失败不影响正确性——下次查询会再次 miss 并回源，只是少了一次缓存机会。
            self._log_cache_skip(operation="search_set")

    async def rotate(
        self,
        *,
        user_id: UUID,
        reason: MemoryCacheRotationReason,
    ) -> None:
        """在数据库 mutation 成功后切换用户缓存 namespace.

        调用链：add() 或 delete() 成功写入数据库 → rotate() 使旧缓存失效。
        这是 generation 机制的核心操作——通过更换版本号，让所有基于旧版本号构造的
        缓存 key 自动失效，无需逐个删除。

        Args:
            user_id: 已成功写入或删除记忆的可信用户 UUID。
            reason: 受控日志原因，不包含身份、正文或缓存 key。

        Notes:
            先 delete 再 set。若 set 失败，下一次搜索会在 miss 时创建新 generation；
            若 Redis 整体不可用，旧查询值最多只存活到短查询 TTL。
        """
        generation_key = build_memory_generation_cache_key(user_id)
        # 生成新的 32 位十六进制版本号（uuid4 的 hex 表示）。
        new_generation = uuid4().hex

        # 先删除旧 generation。这一步即使失败也不影响后续——set 会覆盖。
        # 但先 delete 可以确保在 set 失败时，旧 generation 至少被清除，
        # 下一次 lookup 会走到 _get_or_create_generation 的"不存在"分支创建新值。
        try:
            await self._cache.delete(generation_key)
        except CacheUnavailableError:
            self._log_cache_skip(operation="generation_delete")

        # 写入新 generation。如果这一步失败：
        # - 旧 generation 已被 delete，lookup 会走到 _get_or_create_generation 创建新值。
        # - 最坏情况：旧缓存值继续被命中，直到短查询 TTL 过期。
        try:
            await self._cache.set(
                generation_key,
                new_generation,
                ttl_seconds=self._generation_ttl_seconds,
            )
        except CacheUnavailableError:
            self._log_cache_skip(operation="generation_set")
            return

        logger.info("memory_cache_generation_rotated", reason=reason)

    async def _get_or_create_generation(self, user_id: UUID) -> str | None:
        """读取合法 generation；不存在或损坏时尽力创建新值.

        这是 generation 机制的底层方法，被 lookup() 在每次查询时调用。
        它保证返回一个合法的 32 位 hex 版本号，或者在 Redis 不可用时返回 None。

        三种返回路径：
          1. Redis 中存在合法 generation → 直接返回（最常见，缓存命中路径）。
          2. Redis 中不存在或值损坏 → 创建新 generation 并写入 → 返回新值。
          3. Redis 完全不可用 → 返回 None → 调用方放弃缓存，直接回源数据库。
        """
        key = build_memory_generation_cache_key(user_id)

        # 尝试从 Redis 读取当前 generation。
        try:
            generation = await self._cache.get(key)
        except CacheUnavailableError:
            # Redis 不可用 → 返回 None，调用方会跳过缓存直接查数据库。
            # 这是 fail-open 策略：缓存故障不应阻塞业务。
            self._log_cache_skip(operation="generation_get")
            return None

        # 存在且格式合法（32 位十六进制）→ 直接返回，无需创建。
        if generation is not None and _GENERATION_PATTERN.fullmatch(generation):
            return generation

        # 存在但格式损坏（可能被手动篡改或序列化错误）→ 删除脏值，继续创建新值。
        if generation is not None:
            await self._reject_cached_value(key=key, reason="invalid_generation")

        # generation 不存在或已损坏 → 生成新的 32 位 hex 版本号。
        # uuid4().hex 保证 32 位十六进制，符合 _GENERATION_PATTERN 校验。
        new_generation = uuid4().hex
        try:
            await self._cache.set(
                key,
                new_generation,
                ttl_seconds=self._generation_ttl_seconds,
            )
        except CacheUnavailableError:
            # 写入失败 → 返回 None，本次查询放弃缓存。
            # 下一次查询会再次尝试创建，不影响正确性。
            self._log_cache_skip(operation="generation_create")
            return None
        return new_generation

    async def _reject_cached_value(self, *, key: str, reason: str) -> None:
        """记录固定原因并尽力删除损坏值，不输出 key 或 payload.

        被 lookup() 和 _get_or_create_generation() 调用，用于清理格式错误、
        owner 不匹配或 generation 损坏的缓存值。删除失败不影响正确性——
        旧值最多存活到 TTL 过期。
        """
        logger.warning("memory_cache_value_rejected", reason=reason)
        try:
            await self._cache.delete(key)
        except CacheUnavailableError:
            self._log_cache_skip(operation="corrupt_delete")

    @staticmethod
    def _log_cache_skip(*, operation: str) -> None:
        """记录不含身份、key 和缓存值的 fail-open 事件.

        所有缓存跳过/失败路径都通过此方法记录日志，确保日志中不包含
        用户 ID、缓存 key 或缓存值等敏感信息，只记录操作名称。
        """
        logger.warning("memory_cache_operation_skipped", operation=operation)


__all__ = [
    "MEMORY_CACHE_VERSION",
    "MEMORY_GENERATION_CACHE_NAMESPACE",
    "MEMORY_SEARCH_CACHE_NAMESPACE",
    "MemoryCacheLookup",
    "MemorySearchCache",
    "build_memory_generation_cache_key",
    "build_memory_search_cache_key",
]
