"""Verify the Chat SSE HTTP boundary with a real provider call.

本脚本覆盖两个互补场景：

1. 通过 httpx ASGITransport 调用真实 ``POST /api/v1/chat/stream``，验证
   FastAPI、dependency、ChatService、LangGraph、真实模型、SSE encoder 和 HTTP
   headers 的完整路径；
2. 使用 Starlette StreamingResponse 与 ASGI 2.3 消息做确定性断开探针，验证收到
   ``http.disconnect`` 后，取消会进入内容异步生成器。

第二部分不替代真实模型请求，也不伪造模型行为；它只隔离验证网络传输层的取消
机制。控制台只输出状态、数量和布尔检查，不输出 API key、prompt、thread_id、
token 正文、完整响应、异常文本或 SSE data。
"""

import asyncio
import json
import os
from collections.abc import AsyncIterator
from time import perf_counter
from typing import cast
from uuid import uuid4

# Settings 会在 app.core.config 首次导入时创建。提前写入这些环境变量，可以继续
# 使用本地忽略文件中的 key/base_url/model，同时把真实请求限制为一次尝试、
# 45 秒 LLM 总预算和 256 个输出 token。
os.environ["DEBUG"] = "false"
os.environ["LOG_LEVEL"] = "WARNING"
os.environ["MAX_LLM_CALL_RETRIES"] = "1"
os.environ["LLM_TOTAL_TIMEOUT"] = "45"
os.environ["MAX_TOKENS"] = "256"

from fastapi.responses import StreamingResponse  # noqa: E402
from httpx import ASGITransport, AsyncClient, Response  # noqa: E402
from pydantic import TypeAdapter  # noqa: E402
from starlette.types import Message, Receive, Scope, Send  # noqa: E402

from app.core.config import settings  # noqa: E402
from app.main import app  # noqa: E402
from app.schemas.chat import (  # noqa: E402
    ChatStreamEvent,
    DoneStreamEvent,
    ErrorStreamEvent,
    InterruptStreamEvent,
    TokenStreamEvent,
    ToolStreamEvent,
)

EXPECTED_REPLY = "REAL_HTTP_SSE_OK"
HTTP_TIMEOUT_SECONDS = 120.0

# 该提示明确禁止工具，确保本 smoke 只验证最短的真实文本路径：
# HumanMessage -> chat -> token... -> done(completed)。工具和 HITL 的内部 stream
# 已在 6D-3B/6D-3C 分别通过真实 provider 验证。
SMOKE_PROMPT = "Do not call any tool. Reply with exactly REAL_HTTP_SSE_OK and nothing else."

# ChatStreamEvent 是 Annotated 判别联合。TypeAdapter 会根据 JSON 中的 event 字段
# 选择 Token/Tool/Interrupt/Error/Done 模型，并继续执行各模型的严格字段校验。
STREAM_EVENT_ADAPTER = TypeAdapter(ChatStreamEvent)


def _elapsed_ms(started_at: float) -> float:
    """Return elapsed milliseconds without exposing request data."""
    return round((perf_counter() - started_at) * 1000, 2)


def _print_failure(
    *,
    stage: str,
    started_at: float,
    status_code: int | None = None,
    error_type: str | None = None,
) -> None:
    """Print a safe failure summary without exception or response text."""
    print(
        json.dumps(
            {
                "ok": False,
                "stage": stage,
                "status_code": status_code,
                "error_type": error_type,
                "elapsed_ms": _elapsed_ms(started_at),
            }
        )
    )


def _sse_field_value(line: str) -> str:
    """Return one SSE field value, removing only its optional first space."""
    _, separator, value = line.partition(":")
    if not separator:
        raise ValueError("SSE field must contain a colon")

    # SSE 语法允许冒号后有一个可选空格。只能删除这一个协议空格，不能 strip，
    # 否则 data 中有意义的前导或尾随空白可能被破坏。
    return value[1:] if value.startswith(" ") else value


async def _read_sse_events(response: Response) -> tuple[ChatStreamEvent, ...]:
    """Incrementally parse and validate independent SSE frames."""
    events: list[ChatStreamEvent] = []
    event_name: str | None = None
    data_lines: list[str] = []

    # 断点 1：aiter_lines() 表达的是 HTTP body 文本行，不是模型 token。
    # 一个 token 事件会依次出现 event 行、data 行和空行；空行到达之前，客户端
    # 还不能确定当前帧已经完整。
    async for line in response.aiter_lines():
        if not line:
            # 连续空行在没有待处理字段时可以忽略；有字段时则分发一帧。
            if event_name is None and not data_lines:
                continue

            if event_name is None or not data_lines:
                raise ValueError("SSE frame must contain event and data fields")

            # SSE 允许一帧包含多条 data 行，语义是用换行连接。当前 encoder 只
            # 产生一行，但 parser 保留标准行为，避免客户端实现绑定内部细节。
            raw_data = "\n".join(data_lines)
            payload: object = json.loads(raw_data)
            event = STREAM_EVENT_ADAPTER.validate_python(payload)

            # event 行用于 SSE 客户端分发，JSON event 字段用于 Pydantic 判别；
            # 两者必须一致，否则同一帧会在传输层和应用层拥有两种含义。
            if event.event != event_name:
                raise ValueError("SSE event name must match JSON event field")

            events.append(event)
            event_name = None
            data_lines = []
            continue

        # 以冒号开头的是 SSE comment/heartbeat，不属于业务事件。
        if line.startswith(":"):
            continue

        if line.startswith("event:"):
            if event_name is not None:
                raise ValueError("SSE frame must not repeat the event field")
            event_name = _sse_field_value(line)
        elif line.startswith("data:"):
            data_lines.append(_sse_field_value(line))
        else:
            # 当前公开协议只允许 event/data。主动拒绝未知字段，能及时暴露 encoder
            # 漂移，而不是在 smoke 中静默忽略错误格式。
            raise ValueError("SSE frame contains an unsupported field")

    # 正常 encoder 总以空行结束。如果流结束后仍有字段，说明最后一帧被截断。
    if event_name is not None or data_lines:
        raise ValueError("SSE stream ended with an incomplete frame")

    return tuple(events)


async def _probe_disconnect_cancels_body_iterator() -> tuple[bool, bool]:
    """Verify Starlette's ASGI 2.3 disconnect cancellation deterministically."""
    first_body_sent = asyncio.Event()
    never_complete = asyncio.Event()
    cancelled_seen = False
    generator_finalized = False

    async def probe_body() -> AsyncIterator[str]:
        """Yield once, then wait until StreamingResponse cancels this task."""
        nonlocal cancelled_seen, generator_finalized

        try:
            # 内容只需是合法 SSE 文本；这里测试的是 StreamingResponse/ASGI，
            # 不涉及 LLM、ChatService 或业务事件判断。
            yield 'event: done\ndata: {"event":"done","status":"completed"}\n\n'
            await never_complete.wait()
        except asyncio.CancelledError:
            # ASGI 2.3 分支监听到 http.disconnect 后会取消 stream_response task，
            # CancelledError 应在生成器当前 await 位置出现，并继续向上传播。
            cancelled_seen = True
            raise
        finally:
            # finally 表示即使取消发生，生成器仍有机会释放本地资源。
            generator_finalized = True

    response = StreamingResponse(
        probe_body(),
        media_type="text/event-stream",
    )

    # Starlette 1.6.0 在 ASGI HTTP spec < 2.4 时，会并发运行 stream_response
    # 和 listen_for_disconnect。当前 Uvicorn 0.52.1 声明 HTTP spec 2.3，正走此分支。
    scope = cast(
        Scope,
        {
            "type": "http",
            "asgi": {"version": "3.0", "spec_version": "2.3"},
            "http_version": "1.1",
            "method": "GET",
            "scheme": "http",
            "path": "/disconnect-probe",
            "raw_path": b"/disconnect-probe",
            "query_string": b"",
            "root_path": "",
            "headers": [],
            "client": ("127.0.0.1", 1),
            "server": ("testserver", 80),
            "state": {},
        },
    )

    request_message_sent = False

    async def receive() -> Message:
        """Send one empty request, then disconnect after the first body chunk."""
        nonlocal request_message_sent

        if not request_message_sent:
            request_message_sent = True
            return {
                "type": "http.request",
                "body": b"",
                "more_body": False,
            }

        await first_body_sent.wait()
        return {"type": "http.disconnect"}

    async def send(message: Message) -> None:
        """Observe when StreamingResponse has emitted its first body chunk."""
        if message["type"] == "http.response.body" and message.get("more_body"):
            first_body_sent.set()

    # 类型别名让断点中更容易分辨三个 ASGI 角色：scope 是连接元数据，receive
    # 从客户端收消息，send 向客户端发消息。
    typed_receive: Receive = receive
    typed_send: Send = send

    await asyncio.wait_for(
        response(scope, typed_receive, typed_send),
        timeout=2.0,
    )

    return cancelled_seen, generator_finalized


async def run_chat_sse_api_smoke() -> int:
    """Run one real SSE request and one deterministic disconnect probe."""
    started_at = perf_counter()

    if not settings.OPENAI_API_KEY:
        _print_failure(
            stage="configuration",
            started_at=started_at,
            error_type="MissingApiKey",
        )
        return 1

    try:
        # ASGITransport 仍会经过 FastAPI middleware、Pydantic、dependency、route
        # 与 StreamingResponse，但不需要额外启动端口。provider 调用依然是真实网络
        # 请求；transport 只替代“本机客户端到 ASGI app”这一小段连接。
        transport = ASGITransport(app=app)
        async with AsyncClient(
            transport=transport,
            base_url="http://testserver",
            timeout=HTTP_TIMEOUT_SECONDS,
        ) as client:
            async with client.stream(
                "POST",
                "/api/v1/chat/stream",
                json={
                    "thread_id": f"smoke-sse-{uuid4().hex}",
                    "message": SMOKE_PROMPT,
                },
                headers={"Accept": "text/event-stream"},
            ) as response:
                status_code = response.status_code
                content_type = response.headers.get("content-type", "")
                cache_control = response.headers.get("cache-control")
                accel_buffering = response.headers.get("x-accel-buffering")

                if status_code != 200:
                    # 不读取或打印错误 body，避免 provider/内部错误信息泄漏。
                    _print_failure(
                        stage="http_status",
                        started_at=started_at,
                        status_code=status_code,
                    )
                    return 1

                # 断点 2：这里消费的是 route 输出的 SSE，不是直接调用 ChatService。
                # 每一帧都经过 JSON 解析和 Pydantic 判别联合校验。
                events = await _read_sse_events(response)

        # 断开探针在真实请求完成后单独运行，失败不会再次调用 provider。
        (
            disconnect_cancelled_generator,
            disconnect_finalized_generator,
        ) = await _probe_disconnect_cancels_body_iterator()
    except Exception as error:
        _print_failure(
            stage="execution",
            started_at=started_at,
            error_type=type(error).__name__,
        )
        return 1

    token_events = [event for event in events if isinstance(event, TokenStreamEvent)]
    tool_events = [event for event in events if isinstance(event, ToolStreamEvent)]
    interrupt_events = [event for event in events if isinstance(event, InterruptStreamEvent)]
    error_events = [event for event in events if isinstance(event, ErrorStreamEvent)]
    done_events = [event for event in events if isinstance(event, DoneStreamEvent)]

    # token 文本只在进程内存中用于和固定预期值比较，摘要不输出正文。
    final_text = "".join(event.text for event in token_events)
    content_matches = final_text.strip() == EXPECTED_REPLY

    content_type_matches = content_type.partition(";")[0].strip().lower() == "text/event-stream"
    cache_control_matches = cache_control == "no-cache"
    proxy_buffering_disabled = accel_buffering == "no"
    done_status = done_events[0].status if len(done_events) == 1 else None
    last_event_is_done = bool(events) and isinstance(events[-1], DoneStreamEvent)

    # 断点 3：ok 同时约束 HTTP、SSE、Agent 结果和断开取消四层行为。
    # 只检查固定回答是不够的，因为 route 可能返回了错误媒体类型、缺少 done，
    # 或在客户端断开后仍让后台生成器继续运行。
    ok = (
        status_code == 200
        and content_type_matches
        and cache_control_matches
        and proxy_buffering_disabled
        and len(token_events) >= 1
        and content_matches
        and len(done_events) == 1
        and done_status == "completed"
        and last_event_is_done
        and not tool_events
        and not interrupt_events
        and not error_events
        and disconnect_cancelled_generator
        and disconnect_finalized_generator
    )

    print(
        json.dumps(
            {
                "ok": ok,
                "model": settings.DEFAULT_LLM_MODEL,
                "status_code": status_code,
                "content_type_matches": content_type_matches,
                "cache_control_matches": cache_control_matches,
                "proxy_buffering_disabled": proxy_buffering_disabled,
                "event_count": len(events),
                "event_types": [event.event for event in events],
                "token_event_count": len(token_events),
                "content_matches": content_matches,
                "done_event_count": len(done_events),
                "done_status": done_status,
                "last_event_is_done": last_event_is_done,
                "unexpected_tool_event_count": len(tool_events),
                "unexpected_interrupt_event_count": len(interrupt_events),
                "error_event_count": len(error_events),
                "error_codes": [event.code for event in error_events],
                "disconnect_cancelled_generator": disconnect_cancelled_generator,
                "disconnect_finalized_generator": disconnect_finalized_generator,
                "elapsed_ms": _elapsed_ms(started_at),
            }
        )
    )

    return 0 if ok else 1


if __name__ == "__main__":
    # asyncio.run 管理顶层事件循环；退出码 0/1 让 PowerShell 和 CI 都能可靠
    # 判断验收是否通过，而不依赖人工阅读大量日志。
    raise SystemExit(asyncio.run(run_chat_sse_api_smoke()))
