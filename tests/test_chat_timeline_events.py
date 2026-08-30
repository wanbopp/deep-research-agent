"""Chat Timeline v2 的公开协议边界测试."""

from uuid import UUID, uuid4

import pytest
from pydantic import TypeAdapter, ValidationError

from app.api.sse import encode_sse_event
from app.schemas.chat import ChatTimelineEvent, ItemDeltaEvent, TurnStartedEvent


_adapter = TypeAdapter(ChatTimelineEvent)


def _turn_started_payload() -> dict[str, object]:
    """构造一份完整 payload，便于分别验证版本和严格字段."""
    return {
        "schema_version": 2,
        "event": "turn.started",
        "thread_id": str(uuid4()),
        "turn_id": str(uuid4()),
        "client_message_id": str(uuid4()),
        "user_item_id": str(uuid4()),
        "agent_item_id": str(uuid4()),
    }


def test_timeline_event_preserves_all_correlation_ids() -> None:
    """首事件应完整回显 Thread、Turn 与客户端消息关联 ID."""
    payload = _turn_started_payload()

    event = _adapter.validate_python(payload)

    assert isinstance(event, TurnStartedEvent)
    assert event.schema_version == 2
    assert event.thread_id == UUID(str(payload["thread_id"]))
    assert event.turn_id == UUID(str(payload["turn_id"]))
    assert event.client_message_id == UUID(str(payload["client_message_id"]))


@pytest.mark.parametrize(
    ("field", "value"),
    (("schema_version", 1), ("event", "token"), ("unexpected", True)),
)
def test_timeline_event_rejects_wrong_version_unknown_event_and_extra_fields(
    field: str,
    value: object,
) -> None:
    """公开边界必须拒绝旧版本、旧事件名和额外字段."""
    payload = _turn_started_payload()
    payload[field] = value

    with pytest.raises(ValidationError):
        _adapter.validate_python(payload)


def test_sse_frame_uses_timeline_event_name_and_json_envelope() -> None:
    """SSE 帧头与 JSON 判别字段必须保持一致."""
    event = ItemDeltaEvent(
        thread_id=uuid4(),
        turn_id=uuid4(),
        item_id="agent-1",
        delta="你好",
    )

    frame = encode_sse_event(event)

    assert frame.startswith("event: item.delta\ndata: ")
    assert '"schema_version":2' in frame
    assert '"event":"item.delta"' in frame
    assert frame.endswith("\n\n")
