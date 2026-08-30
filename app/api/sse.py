"""Server-Sent Events encoding helpers."""

from app.schemas.chat import ChatStreamEvent
from app.schemas.research_events import ResearchEventResponse


def encode_sse_event(event: ChatStreamEvent) -> str:
    """把一个聊天流事件编码为独立的 SSE 文本帧."""
    event_name = event.event
    payload = event.model_dump_json()
    return f"event: {event_name}\ndata: {payload}\n\n"


def encode_research_sse_event(event: ResearchEventResponse) -> str:
    """把持久研究事件编码为带 id 的 SSE 帧.

    ``id`` 行会被浏览器记录。连接中断后客户端通过 Last-Event-ID 发回最后一个
    已见 ID，API 再从 PostgreSQL 查询更大的事件，因此断线不会丢关键进度。
    """
    return f"id: {event.event_id}\nevent: {event.event}\ndata: {event.model_dump_json()}\n\n"
