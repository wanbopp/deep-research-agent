"""Use a real provider to verify tool lifecycle stream events.

本脚本验证一次完整真实工具循环在 ChatService 边界产生的事件协议：

1. 第一次真实模型调用选择 ``get_current_utc_time``；
2. chat update 中的完整 ToolCall 被转换为 tool(started)；
3. tools 节点执行本地 Python 工具并产生 ToolMessage；
4. tools update 被转换为与 started 使用相同调用 ID 的 tool(success)；
5. 第二次真实模型调用读取工具结果并产生最终文本 token；
6. LangGraph 到达 END 后产生唯一 done(completed)。

建议依次在事件循环、tool_events 分类、调用 ID 比较和最终 ok 汇总处设置断点。
脚本只输出事件数量、状态、布尔检查与耗时，不输出 key、prompt、工具结果、
调用 ID、token 内容或完整模型回答。
"""

import asyncio
import json
import os
from time import perf_counter
from uuid import UUID

# 必须在首次导入 app.core.config 前设置。一次工具循环包含两个 chat 节点，
# 每个节点分别获得 45 秒 LLMService 总预算；外层 graph 使用 100 秒总预算。
os.environ["DEBUG"] = "false"
os.environ["LOG_LEVEL"] = "WARNING"
os.environ["MAX_LLM_CALL_RETRIES"] = "1"
os.environ["LLM_TOTAL_TIMEOUT"] = "45000"
os.environ["MAX_TOKENS"] = "256"

# 这些导入有意位于环境变量设置之后，E402 noqa 用于保护 Settings 初始化顺序。
from app.agents.chat.runtime import create_chat_runtime  # noqa: E402
from app.agents.chat.tools.current_time import get_current_utc_time  # noqa: E402
from app.core.config import settings  # noqa: E402
from app.infrastructure.chat_guard import InProcessChatExecutionGuard  # noqa: E402
from app.schemas.chat import (  # noqa: E402
    ChatRequest,
    ChatStreamEvent,
    DoneStreamEvent,
    ErrorStreamEvent,
    InterruptStreamEvent,
    TokenStreamEvent,
    ToolStreamEvent,
)
from app.services.chat import ChatService  # noqa: E402

EXPECTED_REPLY = "REAL_TOOL_STREAM_OK"
GRAPH_TIMEOUT_SECONDS = 1000.0
SMOKE_USER_ID = UUID("00000000-0000-4000-8000-000000000001")

# 模型只能提出 ToolCall，真正执行工具的是 LangGraph 的 tools 节点。固定回复使
# smoke 可以验证第二次模型调用确实发生，同时避免依赖时间字符串的具体内容。
SMOKE_PROMPT = (
    "Call the get_current_utc_time tool exactly once. "
    "After receiving its result, do not call any tool again and "
    "reply with exactly REAL_TOOL_STREAM_OK."
)


async def run_tool_stream_smoke() -> int:
    """Consume one real tool Agent stream and return a process exit code."""
    started_at = perf_counter()

    if not settings.OPENAI_API_KEY:
        print(json.dumps({"ok": False, "error_type": "MissingApiKey"}))
        return 1

    # 使用生产 runtime，确保 smoke 与 FastAPI dependency 走相同的模型、工具、
    # 节点和 graph 装配路径。当前脚本进程拥有独立 InMemorySaver。
    service = ChatService(
        create_chat_runtime(),
        # 仅适用于当前单进程脚本；生产应用由 lifespan 注入 Redis 分布式 guard。
        execution_guard=InProcessChatExecutionGuard(),
        graph_timeout_seconds=GRAPH_TIMEOUT_SECONDS,
    )

    events: list[ChatStreamEvent] = []
    token_parts: list[str] = []

    try:
        # 断点 1：观察每个 event 的具体类型。预期主要顺序为：
        # ToolStreamEvent(started) -> ToolStreamEvent(success) ->
        # TokenStreamEvent... -> DoneStreamEvent(completed)。
        async for event in service.stream_turn(
            ChatRequest(
                thread_id="smoke-chat-tool-stream",
                message=SMOKE_PROMPT,
            ),
            user_id=SMOKE_USER_ID,
        ):
            events.append(event)

            # token 只在内存中用于恢复固定最终回答，不打印任何模型正文。
            if isinstance(event, TokenStreamEvent):
                token_parts.append(event.text)
    except Exception as error:
        # stream_turn 内部错误通常会成为 ErrorStreamEvent；这里保护 runtime
        # 创建等更外层异常。只输出异常类型和可选状态码，不使用 str(error)。
        print(
            json.dumps(
                {
                    "ok": False,
                    "error_type": type(error).__name__,
                    "status_code": getattr(error, "status_code", None),
                    "elapsed_ms": round(
                        (perf_counter() - started_at) * 1000,
                        2,
                    ),
                }
            )
        )
        return 1

    # 保留事件在原始序列中的位置。只有分类结果而没有位置，就无法证明
    # started 确实发生在 success 之前，也无法验证 done 是否位于最后。
    indexed_tool_events = [(index, event) for index, event in enumerate(events) if isinstance(event, ToolStreamEvent)]
    indexed_visible_tokens = [
        (index, event)
        for index, event in enumerate(events)
        if isinstance(event, TokenStreamEvent) and event.text.strip()
    ]

    done_events = [event for event in events if isinstance(event, DoneStreamEvent)]
    error_events = [event for event in events if isinstance(event, ErrorStreamEvent)]
    interrupt_events = [event for event in events if isinstance(event, InterruptStreamEvent)]

    # 断点 2：正常路径必须恰好包含 started 和 success 两个工具事件。
    # 如果模型重复调用工具或工具执行失败，数量或状态列表会直接暴露差异。
    tool_statuses = [event.status for _, event in indexed_tool_events]
    tool_event_count_matches = len(indexed_tool_events) == 2

    started_event = indexed_tool_events[0][1] if tool_event_count_matches else None
    success_event = indexed_tool_events[1][1] if tool_event_count_matches else None

    tool_statuses_match = tool_statuses == ["started", "success"]
    tool_names_match = (
        started_event is not None
        and success_event is not None
        and started_event.name == get_current_utc_time.name
        and success_event.name == get_current_utc_time.name
    )

    # 断点 3：name 只能说明调用了哪个工具；tool_call_id 才能证明 success
    # 回答的是同一次 started。只输出比较结果，不能输出 ID 本身。
    tool_call_id_matches = (
        started_event is not None
        and success_event is not None
        and bool(started_event.tool_call_id.strip())
        and started_event.tool_call_id == success_event.tool_call_id
    )

    tool_order_matches = tool_event_count_matches and indexed_tool_events[0][0] < indexed_tool_events[1][0]

    # 忽略纯空白 token 的位置，避免 provider 在 ToolCall 前发送空白 chunk
    # 导致误报；第一段可见最终文本仍应发生在工具成功之后。
    first_visible_token_after_success = (
        bool(indexed_visible_tokens)
        and tool_event_count_matches
        and indexed_tool_events[1][0] < indexed_visible_tokens[0][0]
    )

    final_text = "".join(token_parts)
    content_matches = final_text.strip() == EXPECTED_REPLY

    done_status = done_events[0].status if len(done_events) == 1 else None
    last_event_is_done = bool(events) and isinstance(events[-1], DoneStreamEvent)

    # 断点 4：ok 同时覆盖工具语义、关联关系、事件顺序、最终文本和终止状态。
    # 任一条件失败都返回非零退出码，避免只凭控制台“看起来正常”判断成功。
    ok = (
        tool_event_count_matches
        and tool_statuses_match
        and tool_names_match
        and tool_call_id_matches
        and tool_order_matches
        and first_visible_token_after_success
        and len(token_parts) >= 1
        and content_matches
        and len(done_events) == 1
        and done_status == "completed"
        and last_event_is_done
        and not error_events
        and not interrupt_events
    )

    # 输出只包含结构摘要。工具调用 ID、工具返回的时间、模型文本和 prompt
    # 始终留在进程内存，不进入终端或日志文件。
    print(
        json.dumps(
            {
                "ok": ok,
                "model": settings.DEFAULT_LLM_MODEL,
                "event_count": len(events),
                "tool_event_count": len(indexed_tool_events),
                "tool_statuses": tool_statuses,
                "tool_event_count_matches": tool_event_count_matches,
                "tool_names_match": tool_names_match,
                "tool_call_id_matches": tool_call_id_matches,
                "tool_order_matches": tool_order_matches,
                "first_visible_token_after_success": (first_visible_token_after_success),
                "token_event_count": len(token_parts),
                "content_matches": content_matches,
                "done_event_count": len(done_events),
                "done_status": done_status,
                "last_event_is_done": last_event_is_done,
                "error_event_count": len(error_events),
                "error_codes": [event.code for event in error_events],
                "unexpected_interrupt_event_count": len(interrupt_events),
                "elapsed_ms": round(
                    (perf_counter() - started_at) * 1000,
                    2,
                ),
            }
        )
    )

    return 0 if ok else 1


if __name__ == "__main__":
    # asyncio.run 管理顶层事件循环，SystemExit 把验收结果变成 shell/CI 退出码。
    raise SystemExit(asyncio.run(run_tool_stream_smoke()))
