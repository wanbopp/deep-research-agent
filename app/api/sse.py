"""Server-Sent Events encoding helpers."""

from app.schemas.chat import ChatStreamEvent


def encode_sse_event(event: ChatStreamEvent) -> str:
    """把一个聊天流事件编码为独立的 SSE 文本帧."""
    event_name = event.event
    payload = event.model_dump_json()
    return f"event: {event_name}\ndata: {payload}\n\n"
