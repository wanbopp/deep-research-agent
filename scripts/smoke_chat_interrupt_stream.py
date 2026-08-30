"""使用真实 Provider 验证 Timeline v2 HITL 暂停与 checkpoint 快照恢复."""

import asyncio
import json
import os
from time import perf_counter
from uuid import UUID

os.environ["DEBUG"] = "false"
os.environ["LOG_LEVEL"] = "WARNING"
os.environ["MAX_LLM_CALL_RETRIES"] = "1"
os.environ["LLM_TOTAL_TIMEOUT"] = "45"
os.environ["MAX_TOKENS"] = "256"

from langchain_core.messages import AIMessage  # noqa: E402

from app.agents.chat.runtime import create_chat_runtime  # noqa: E402
from app.agents.chat.tools.ask_human import ask_human  # noqa: E402
from app.core.config import settings  # noqa: E402
from app.infrastructure.chat_guard import InProcessChatExecutionGuard  # noqa: E402
from app.schemas.chat import (  # noqa: E402
    ChatRequest,
    ChatTimelineEvent,
    ItemDeltaEvent,
    ItemStartedEvent,
    PendingActionCreatedEvent,
    TimelineErrorEvent,
    TurnCompletedEvent,
    TurnStartedEvent,
)
from app.services.chat import ChatService  # noqa: E402
from app.services.chat_session_ownership import InProcessChatSessionOwnershipVerifier  # noqa: E402

THREAD_ID = UUID("00000000-0000-4000-8000-000000000104")
SMOKE_USER_ID = UUID("00000000-0000-4000-8000-000000000001")
SMOKE_PROMPT = (
    "Call the ask_human tool exactly once to ask whether this bounded "
    "stream smoke action is approved. Do not answer directly."
)


async def run_interrupt_stream_smoke() -> int:
    """验证实时事件、LangGraph interrupt 与 canonical snapshot 使用同一身份."""
    started_at = perf_counter()
    if not settings.OPENAI_API_KEY:
        print(json.dumps({"ok": False, "error_type": "MissingApiKey"}))
        return 1

    graph = create_chat_runtime()
    service = ChatService(
        graph,
        execution_guard=InProcessChatExecutionGuard(),
        ownership_verifier=InProcessChatSessionOwnershipVerifier({(SMOKE_USER_ID, THREAD_ID)}),
        graph_timeout_seconds=60.0,
    )
    request = ChatRequest(thread_id=THREAD_ID, message=SMOKE_PROMPT)
    config = ChatService._build_config(user_id=SMOKE_USER_ID, public_thread_id=THREAD_ID)
    events: list[ChatTimelineEvent] = []
    try:
        async for event in service.stream_turn(request, user_id=SMOKE_USER_ID):
            events.append(event)
        graph_snapshot = await graph.aget_state(config)
        timeline_snapshot = await service.get_timeline_snapshot(
            session_id=THREAD_ID,
            user_id=SMOKE_USER_ID,
        )
    except Exception as error:
        print(
            json.dumps(
                {
                    "ok": False,
                    "error_type": type(error).__name__,
                    "elapsed_ms": round((perf_counter() - started_at) * 1000, 2),
                }
            )
        )
        return 1

    started = next((event for event in events if isinstance(event, TurnStartedEvent)), None)
    tool_started = next(
        (
            event
            for event in events
            if isinstance(event, ItemStartedEvent) and event.item_type == "toolCall" and event.name == ask_human.name
        ),
        None,
    )
    pending = next((event for event in events if isinstance(event, PendingActionCreatedEvent)), None)
    terminal = events[-1] if events and isinstance(events[-1], TurnCompletedEvent) else None
    errors = [event for event in events if isinstance(event, TimelineErrorEvent)]
    deltas = [event for event in events if isinstance(event, ItemDeltaEvent)]

    checkpoint_interrupts = tuple(interrupt for task in graph_snapshot.tasks for interrupt in task.interrupts)
    checkpoint_question = checkpoint_interrupts[0].value if len(checkpoint_interrupts) == 1 else None
    messages = graph_snapshot.values.get("messages", [])
    pending_ai = next(
        (
            message
            for message in reversed(messages if isinstance(messages, list) else [])
            if isinstance(message, AIMessage) and message.tool_calls
        ),
        None,
    )
    tool_calls = pending_ai.tool_calls if pending_ai is not None else []
    tool_call = tool_calls[0] if len(tool_calls) == 1 else None
    tool_args = tool_call.get("args") if tool_call is not None else None
    tool_question = tool_args.get("question") if isinstance(tool_args, dict) else None

    snapshot_pending = next(
        (event for event in timeline_snapshot.events if isinstance(event, PendingActionCreatedEvent)),
        None,
    )
    snapshot_terminal = (
        timeline_snapshot.events[-1]
        if timeline_snapshot.events and isinstance(timeline_snapshot.events[-1], TurnCompletedEvent)
        else None
    )

    identity_matches = (
        started is not None
        and pending is not None
        and snapshot_pending is not None
        and started.turn_id == pending.turn_id == snapshot_pending.turn_id
        and pending.request_id == snapshot_pending.request_id
        and pending.item_id == snapshot_pending.item_id
    )
    question_matches = (
        pending is not None
        and snapshot_pending is not None
        and isinstance(checkpoint_question, str)
        and isinstance(tool_question, str)
        and pending.question == snapshot_pending.question == checkpoint_question == tool_question
    )
    ok = (
        started is not None
        and tool_started is not None
        and pending is not None
        and terminal is not None
        and terminal.status == "waitingForUser"
        and snapshot_terminal is not None
        and snapshot_terminal.status == "waitingForUser"
        and graph_snapshot.next == ("tools",)
        and len(checkpoint_interrupts) == 1
        and len(tool_calls) == 1
        and identity_matches
        and question_matches
        and not errors
        and not deltas
    )
    print(
        json.dumps(
            {
                "ok": ok,
                "model": settings.DEFAULT_LLM_MODEL,
                "event_count": len(events),
                "snapshot_event_count": len(timeline_snapshot.events),
                "tool_started": tool_started is not None,
                "waiting_terminal": terminal is not None and terminal.status == "waitingForUser",
                "snapshot_waiting_terminal": (
                    snapshot_terminal is not None and snapshot_terminal.status == "waitingForUser"
                ),
                "checkpoint_interrupt_count": len(checkpoint_interrupts),
                "identity_matches": identity_matches,
                "question_matches": question_matches,
                "error_event_count": len(errors),
                "unexpected_delta_count": len(deltas),
                "elapsed_ms": round((perf_counter() - started_at) * 1000, 2),
            }
        )
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(run_interrupt_stream_smoke()))
