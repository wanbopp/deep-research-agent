"""MemoryService 缓存、安全策略和降级语义的聚焦测试."""

from datetime import UTC, datetime
from uuid import UUID, uuid4

import pytest

from app.infrastructure.cache import InMemoryCache
from app.schemas.memory import (
    MemoryCreate,
    MemoryItem,
    MemoryKind,
    MemoryQuery,
    MemorySearchResult,
    MemorySearchStatus,
)
from app.services.memory import MemoryUnavailableError
from app.services.memory_cache import (
    MemorySearchCache,
    build_memory_search_cache_key,
)
from app.services.memory_policy import MemoryRejectedError
from app.services.memory_service import MemoryService


class _RecordingMemoryStore:
    """只记录应用协议调用的确定性 Store.

    它不模拟模型输出或向量相似度；测试只观察 MemoryService 是否回源、失效和
    拒绝调用。真实 Embedding/pgvector 行为由独立端到端 smoke 负责。
    """

    def __init__(self, items: tuple[MemoryItem, ...]) -> None:
        """保存初始权威条目和可观察调用计数."""
        self.items = list(items)
        self.search_calls = 0
        self.list_calls = 0
        self.add_calls = 0
        self.delete_calls = 0
        self.search_unavailable = False
        self.list_unavailable = False

    async def search(
        self,
        *,
        user_id: UUID,
        query: MemoryQuery,
    ) -> tuple[MemoryItem, ...]:
        """按 owner/kind/limit 返回确定性结果，或抛稳定故障."""
        self.search_calls += 1
        if self.search_unavailable:
            raise MemoryUnavailableError()
        matches = [
            item
            for item in self.items
            if item.user_id == user_id and (query.kinds is None or item.kind in query.kinds)
        ]
        return tuple(matches[: query.limit])

    async def list(self, *, user_id: UUID) -> tuple[MemoryItem, ...]:
        """返回该用户全部条目（创建时间倒序），或抛稳定故障."""
        self.list_calls += 1
        if self.list_unavailable:
            raise MemoryUnavailableError()
        owned = [item for item in self.items if item.user_id == user_id]
        owned.sort(key=lambda item: (item.created_at, item.id), reverse=True)
        return tuple(owned)

    async def add(
        self,
        *,
        user_id: UUID,
        memory: MemoryCreate,
    ) -> MemoryItem:
        """把候选转换为权威条目并记录调用."""
        self.add_calls += 1
        now = datetime.now(UTC)
        item = MemoryItem(
            id=uuid4(),
            user_id=user_id,
            content=memory.content,
            kind=memory.kind,
            source_thread_id=memory.source_thread_id,
            created_at=now,
            updated_at=now,
        )
        self.items.append(item)
        return item

    async def delete(self, *, user_id: UUID, memory_id: UUID) -> None:
        """执行 owner-scoped 幂等删除并记录调用."""
        self.delete_calls += 1
        self.items = [item for item in self.items if not (item.user_id == user_id and item.id == memory_id)]


def _memory_item(*, user_id: UUID, source_thread_id: UUID) -> MemoryItem:
    """构造一条带合法审计时间的测试记忆."""
    now = datetime.now(UTC)
    return MemoryItem(
        id=uuid4(),
        user_id=user_id,
        content="用户喜欢简短的中文回答",
        kind=MemoryKind.PREFERENCE,
        source_thread_id=source_thread_id,
        created_at=now,
        updated_at=now,
    )


@pytest.mark.anyio
async def test_memory_service_caches_rotates_and_rejects_credentials_before_store() -> None:
    """缓存命中、写后失效和敏感内容拒绝应形成一条完整应用链."""
    user_id = uuid4()
    source_thread_id = uuid4()
    original = _memory_item(
        user_id=user_id,
        source_thread_id=source_thread_id,
    )
    store = _RecordingMemoryStore((original,))
    cache = InMemoryCache()
    service = MemoryService(
        store,
        cache,
        search_cache_ttl_seconds=60,
        generation_ttl_seconds=3600,
    )
    query = MemoryQuery(text="回答风格")

    first = await service.search(user_id=user_id, query=query)
    second = await service.search(user_id=user_id, query=query)
    assert first.status is MemorySearchStatus.AVAILABLE
    assert second.items == first.items
    assert store.search_calls == 1

    # 高度疑似 credential 的候选必须在调用 Store 前失败。由于生产 Store 的 add
    # 才会请求真实 Embedding，该计数同时证明敏感正文没有跨越模型边界。
    with pytest.raises(MemoryRejectedError) as exc_info:
        await service.add(
            user_id=user_id,
            memory=MemoryCreate(
                content="api_key=sk-not-a-real-secret-1234567890",
                kind=MemoryKind.FACT,
                source_thread_id=source_thread_id,
            ),
        )
    assert exc_info.value.code.value == "sensitive_credential"
    assert store.add_calls == 0

    added = await service.add(
        user_id=user_id,
        memory=MemoryCreate(
            content="用户要求不要在回答中泄露 API Key",
            kind=MemoryKind.CONSTRAINT,
            source_thread_id=source_thread_id,
        ),
    )
    after_add = await service.search(user_id=user_id, query=query)
    assert store.add_calls == 1
    assert store.search_calls == 2
    assert added in after_add.items

    await service.delete(user_id=user_id, memory_id=original.id)
    after_delete = await service.search(user_id=user_id, query=query)
    assert store.delete_calls == 1
    assert store.search_calls == 3
    assert original not in after_delete.items


@pytest.mark.anyio
async def test_memory_service_distinguishes_empty_degraded_and_rejects_wrong_owner_cache() -> None:
    """正常空结果、Store 故障和缓存 owner 污染必须保持不同语义."""
    user_id = uuid4()
    query = MemoryQuery(text="不存在的长期记忆")
    cache = InMemoryCache()
    store = _RecordingMemoryStore(())
    service = MemoryService(
        store,
        cache,
        search_cache_ttl_seconds=60,
        generation_ttl_seconds=3600,
    )

    empty_result = await service.search(user_id=user_id, query=query)
    assert empty_result.status is MemorySearchStatus.AVAILABLE
    assert empty_result.items == ()
    assert empty_result.error_code is None

    store.search_unavailable = True
    # 换一个 query，避免命中上一步缓存的正常空结果。
    degraded = await service.search(
        user_id=user_id,
        query=MemoryQuery(text="触发真实存储故障语义"),
    )
    assert degraded.is_degraded
    assert degraded.items == ()
    assert degraded.error_code == "MEMORY_UNAVAILABLE"

    # 单独验证缓存层：即使 payload 是合法 MemorySearchResult，只要条目属于另一个
    # 用户也必须按损坏值处理，不能把 SHA-256 key 当作授权机制。
    store.search_unavailable = False
    memory_cache = MemorySearchCache(
        cache,
        search_ttl_seconds=60,
        generation_ttl_seconds=3600,
    )
    owner_query = MemoryQuery(text="owner check")
    miss = await memory_cache.lookup(user_id=user_id, query=owner_query)
    assert miss.generation is not None
    assert miss.key is not None
    expected_key = build_memory_search_cache_key(
        user_id=user_id,
        generation=miss.generation,
        query=owner_query,
    )
    assert expected_key == miss.key
    assert str(user_id) not in expected_key
    assert owner_query.text not in expected_key

    wrong_owner = _memory_item(
        user_id=uuid4(),
        source_thread_id=uuid4(),
    )
    await cache.set(
        miss.key,
        MemorySearchResult(items=(wrong_owner,)).model_dump_json(),
        ttl_seconds=60,
    )
    rejected = await memory_cache.lookup(user_id=user_id, query=owner_query)
    assert not rejected.is_hit


def _build_list_service(store: _RecordingMemoryStore) -> MemoryService:
    """构造只做列表验证的 MemoryService（列表不使用缓存）."""
    return MemoryService(
        store,
        InMemoryCache(),
        search_cache_ttl_seconds=60,
        generation_ttl_seconds=3600,
    )


@pytest.mark.anyio
async def test_memory_service_list_returns_only_owned_items_newest_first() -> None:
    """列表必须只包含当前用户条目，且最新创建的排在前面."""
    user_id = uuid4()
    source_thread_id = uuid4()
    older_time = datetime(2026, 8, 1, tzinfo=UTC)
    newer_time = datetime(2026, 8, 20, tzinfo=UTC)
    older = MemoryItem(
        id=uuid4(),
        user_id=user_id,
        content="较早的记忆",
        kind=MemoryKind.FACT,
        source_thread_id=source_thread_id,
        created_at=older_time,
        updated_at=older_time,
    )
    newer = MemoryItem(
        id=uuid4(),
        user_id=user_id,
        content="较晚的记忆",
        kind=MemoryKind.FACT,
        source_thread_id=source_thread_id,
        created_at=newer_time,
        updated_at=newer_time,
    )
    other_user_item = _memory_item(user_id=uuid4(), source_thread_id=uuid4())
    store = _RecordingMemoryStore((older, newer, other_user_item))
    service = _build_list_service(store)

    items = await service.list(user_id=user_id)

    assert store.list_calls == 1
    assert [item.id for item in items] == [newer.id, older.id]


@pytest.mark.anyio
async def test_memory_service_list_propagates_store_unavailable() -> None:
    """存储故障必须向上抛出，由 REST 层映射为 503，不能伪装成空列表."""
    store = _RecordingMemoryStore(())
    store.list_unavailable = True
    service = _build_list_service(store)

    with pytest.raises(MemoryUnavailableError):
        await service.list(user_id=uuid4())


@pytest.mark.anyio
async def test_memory_service_list_rejects_wrong_owner_items() -> None:
    """Store 返回其他用户条目属于契约违规，必须按不可用处理."""
    user_id = uuid4()
    foreign = _memory_item(user_id=uuid4(), source_thread_id=uuid4())
    store = _ForeignOwnerListStore(foreign)
    service = _build_list_service(store)

    with pytest.raises(MemoryUnavailableError):
        await service.list(user_id=user_id)


class _ForeignOwnerListStore(_RecordingMemoryStore):
    """故意返回其他用户条目的违规 Store，只用于契约复核测试."""

    def __init__(self, foreign_item: MemoryItem) -> None:
        """保存将由 list 返回的他人条目."""
        super().__init__(())
        self._foreign_item = foreign_item

    async def list(self, *, user_id: UUID) -> tuple[MemoryItem, ...]:
        """无视作用域返回他人条目，模拟 adapter 回归."""
        self.list_calls += 1
        return (self._foreign_item,)
