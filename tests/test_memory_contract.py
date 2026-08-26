"""长期记忆 schema 与应用协议的聚焦契约测试."""

from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest
from pydantic import ValidationError

from app.schemas.memory import (
    MAX_QUERY_LIMIT,
    MemoryCreate,
    MemoryItem,
    MemoryKind,
    MemoryQuery,
)
from app.services.memory import MemoryUnavailableError


def test_memory_models_keep_candidate_identity_and_query_boundaries_separate() -> None:
    """一个聚焦测试覆盖候选、权威归属、查询上限和严格模型配置."""
    source_thread_id = uuid4()
    user_id = uuid4()
    memory_id = uuid4()
    created_at = datetime.now(UTC)

    # MemoryCreate 可以来自未来提取器，但不允许携带 user_id。归属将在调用
    # MemoryStore.add(user_id=...) 时由可信认证上下文单独提供。
    candidate = MemoryCreate(
        content="  用户喜欢简短回答  ",
        kind=MemoryKind.PREFERENCE,
        source_thread_id=source_thread_id,
    )
    assert candidate.content == "用户喜欢简短回答"
    assert "user_id" not in type(candidate).model_fields

    item = MemoryItem(
        id=memory_id,
        user_id=user_id,
        content=candidate.content,
        kind=candidate.kind,
        source_thread_id=candidate.source_thread_id,
        created_at=created_at,
        updated_at=created_at,
    )
    assert item.user_id == user_id
    assert item.id == memory_id

    query = MemoryQuery(
        text="  如何组织回答  ",
        kinds=frozenset(
            {
                MemoryKind.PREFERENCE,
                MemoryKind.CONSTRAINT,
            }
        ),
    )
    assert query.text == "如何组织回答"
    assert query.limit == 5

    # 查询中禁止 user_id，阻止客户端或模型选择其他用户 namespace。
    with pytest.raises(ValidationError, match="extra_forbidden"):
        MemoryQuery.model_validate(
            {
                "text": "private memory",
                "user_id": str(uuid4()),
            }
        )

    for invalid_limit in (0, MAX_QUERY_LIMIT + 1):
        with pytest.raises(ValidationError):
            MemoryQuery(text="query", limit=invalid_limit)

    with pytest.raises(ValidationError):
        MemoryQuery(text="query", kinds=frozenset())

    # 记忆审计时间必须带时区且保持单调，避免跨 worker 排序出现歧义。
    with pytest.raises(ValidationError):
        MemoryItem(
            id=memory_id,
            user_id=user_id,
            content=candidate.content,
            kind=candidate.kind,
            source_thread_id=source_thread_id,
            created_at=created_at,
            updated_at=created_at - timedelta(seconds=1),
        )

    with pytest.raises(ValidationError, match="frozen"):
        item.content = "不能原地修改"

    assert str(MemoryUnavailableError()) == "Memory backend is unavailable"
