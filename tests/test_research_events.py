"""Research 事件 v1 判别联合与兼容 reader 测试."""

from datetime import UTC, datetime
from uuid import uuid4

import pytest
from pydantic import ValidationError

from app.api.sse import encode_research_sse_event
from app.schemas.research_events import (
    LegacyResearchEvent,
    NodeCompletedEvent,
    ResearchEventType,
    parse_research_event,
    validate_research_event_payload,
)


def test_node_event_is_discriminated_and_sse_keeps_cursor() -> None:
    """节点事件使用固定名称，节点名属于 payload，SSE id 仍可用于续传."""
    run_id = uuid4()
    event = parse_research_event(
        {
            "event_id": 17,
            "schema_version": 1,
            "event": "node_completed",
            "run_id": run_id,
            "payload": {"node": "retrieve", "status": "validating", "evidence_count": 3},
            "created_at": datetime.now(UTC),
        }
    )

    assert isinstance(event, NodeCompletedEvent)
    assert event.payload.node == "retrieve"
    frame = encode_research_sse_event(event)
    assert frame.startswith("id: 17\nevent: node_completed\n")
    assert '"schema_version":1' in frame
    assert str(run_id) in frame


def test_payload_is_validated_before_persistence() -> None:
    """未知字段和正文不能借宽松 dict 混入持久事件."""
    with pytest.raises(ValidationError):
        validate_research_event_payload(
            ResearchEventType.NODE_COMPLETED,
            {"node": "retrieve", "raw_document": "不允许进入事件表"},
        )


def test_unknown_legacy_event_remains_readable() -> None:
    """迁移前的未知事件以 schema_version=0 只读，不伪装成 v1."""
    event = parse_research_event(
        {
            "event_id": 3,
            "schema_version": 0,
            "event": "legacy_custom_event",
            "run_id": None,
            "payload": {"old": True},
            "created_at": datetime.now(UTC),
        }
    )

    assert isinstance(event, LegacyResearchEvent)
    assert event.payload == {"old": True}


def test_unknown_v1_event_is_rejected() -> None:
    """新版本出现未知事件时显式失败，避免页面静默解释错误协议."""
    with pytest.raises(ValidationError):
        parse_research_event(
            {
                "event_id": 4,
                "schema_version": 1,
                "event": "future_event",
                "run_id": None,
                "payload": {},
                "created_at": datetime.now(UTC),
            }
        )
