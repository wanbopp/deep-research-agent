"""Use a real provider to verify completed text stream events.

本脚本验证 ChatService 的纯文本流式边界，而不是评价模型回答质量：

1. 真实 provider 生成的 AIMessageChunk 被转换为 TokenStreamEvent；
2. 所有 token 按原顺序拼接后得到固定预期文本；
3. LangGraph 到达 END 后产生唯一 DoneStreamEvent(completed)；
4. 纯文本场景不应产生 tool、interrupt 或 error 事件。

脚本不会输出 API key、完整 prompt、token 内容或完整模型回答，只输出结构化、
可复现的脱敏验收摘要。

建议按以下顺序设置断点：

1. settings key 检查：观察环境配置是否在首次导入时正确冻结；
2. ChatService 创建：观察生产 runtime 如何组装 graph、LLM 和工具；
3. async for 循环体：观察事件如何按 token...done 的顺序到达；
4. final_text 聚合：观察多个 provider chunk 如何恢复为完整语义；
5. ok 汇总：理解一次验收为什么要同时覆盖内容、顺序和排除项。
"""

import asyncio
import json
import os
from time import perf_counter
from uuid import UUID

# Settings 在首次导入 app.core.config 时创建，因此必须先设置受控请求参数。
# dotenv 默认不会覆盖已经存在的环境变量，这些值可以限制本次真实请求成本。
# 断点提示：app 模块导入完成后再修改这些变量已经太晚，因为全局 settings
# 对象已经读取并保存了旧值。
os.environ["DEBUG"] = "false"
os.environ["LOG_LEVEL"] = "WARNING"
os.environ["MAX_LLM_CALL_RETRIES"] = "1"
os.environ["LLM_TOTAL_TIMEOUT"] = "45"
os.environ["MAX_TOKENS"] = "256"

# 这些导入故意位于环境变量设置之后。E402 通常要求 import 位于文件顶部，
# 这里使用 noqa 是有明确生命周期原因的例外，不是随意关闭代码规范。
from app.agents.chat.runtime import create_chat_runtime  # noqa: E402
from app.core.config import settings  # noqa: E402
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

EXPECTED_REPLY = "REAL_STREAM_TEXT_OK"
GRAPH_TIMEOUT_SECONDS = 60.0
SMOKE_USER_ID = UUID("00000000-0000-4000-8000-000000000001")

# 明确禁止工具调用，让本次实验只验证最短路径 chat -> END。
SMOKE_PROMPT = "Do not call any tool. Reply with exactly REAL_STREAM_TEXT_OK and nothing else."


async def run_text_stream_smoke() -> int:
    """Consume one real text stream and return a process exit code."""
    # perf_counter 使用单调时钟，不受系统时间被手工调整或网络校时影响，
    # 适合衡量一次真实请求的耗时。
    started_at = perf_counter()

    # 尽早失败可以避免在 runtime 深处才得到难以理解的认证异常。
    if not settings.OPENAI_API_KEY:
        print(json.dumps({"ok": False, "error_type": "MissingApiKey"}))
        return 1

    # LLMService、ToolRegistry、chat/tools 节点、InMemorySaver 和 compiled graph。
    # 本脚本复用生产装配入口，避免 smoke 与实际应用走两套不同代码路径。
    # 每次脚本执行只创建一个 runtime；60 秒限制整轮 graph，而不是单个 chunk。
    service = ChatService(
        create_chat_runtime(),
        graph_timeout_seconds=GRAPH_TIMEOUT_SECONDS,
    )

    # events 保存完整应用事件序列，用于验证 done 是否唯一且位于最后；
    # token_parts 只保存文本分片，用于恢复完整回答。二者承担不同验收职责。
    events: list[ChatStreamEvent] = []
    token_parts: list[str] = []

    try:
        # stream_turn 是异步生成器。调用它会创建异步迭代器；真正的图执行和
        # 首次网络请求发生在 async for 请求下一项时，而不是构造迭代器时。
        # 每次迭代拿到应用层事件，不是 LangGraph 原始 part；模型 chunk、
        # ToolCall 和 snapshot 的框架细节已经被 ChatService 隔离。
        async for event in service.stream_turn(
            ChatRequest(
                thread_id="smoke-chat-stream-text",
                message=SMOKE_PROMPT,
            ),
            user_id=SMOKE_USER_ID,
        ):
            # 断点 3：观察 type(event).__name__ 和 event.event。正常纯文本路径
            # 会多次进入 TokenStreamEvent，最后一次进入 DoneStreamEvent；
            # chunk 次数由 provider 决定，不能写死。
            events.append(event)

            # 只在内存中收集文本用于固定答案比较。不能给每个 chunk 自行添加
            # 空格，因为 provider 返回的 chunk 已经包含原始空格和换行。
            if isinstance(event, TokenStreamEvent):
                token_parts.append(event.text)
    except Exception as error:
        # stream_turn 通常会把运行异常转换成 ErrorStreamEvent；这里仍保护
        # runtime 创建或调用边界上的意外异常。禁止打印 str(error)，避免泄露
        # provider 响应、prompt 或其他内部信息。
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

    # 断点 4：provider 可以把同一句文本切成 1 个或多个 chunk，因此不检查
    # 精确 token 数量。按到达顺序直接 join，才能恢复原始语义。
    final_text = "".join(token_parts)
    content_matches = final_text.strip() == EXPECTED_REPLY

    # 同一份 events 按具体 Pydantic 事件类型分类。这里不是重新解析 LangGraph，
    # 而是在应用协议层验证“应该出现什么”和“不应该出现什么”。
    done_events = [event for event in events if isinstance(event, DoneStreamEvent)]
    error_events = [event for event in events if isinstance(event, ErrorStreamEvent)]
    tool_events = [event for event in events if isinstance(event, ToolStreamEvent)]
    interrupt_events = [event for event in events if isinstance(event, InterruptStreamEvent)]

    # done 必须是唯一且最后一个事件。只检查存在 done 不够，否则协议错误地
    # 在 done 后继续发送 token 时，客户端仍可能被误判为验收成功。
    last_event_is_done = bool(events) and isinstance(events[-1], DoneStreamEvent)
    done_status = done_events[0].status if len(done_events) == 1 else None

    # 断点 5：ok 是本 checkpoint 的行为保证集合：
    # - 内容层：有 token，且聚合文本匹配；
    # - 终止层：唯一 completed done，并且它位于最后；
    # - 排除层：纯文本场景不能意外调用工具、暂停或报错。
    # 任一条件失败都返回非零退出码，避免依赖人眼阅读日志判断成功。
    ok = (
        len(token_parts) >= 1
        and content_matches
        and len(done_events) == 1
        and done_status == "completed"
        and last_event_is_done
        and not error_events
        and not tool_events
        and not interrupt_events
    )

    # 只输出数量、状态和布尔检查。不输出 token_parts、final_text、prompt、
    # thread_id、调用 ID、checkpoint、凭据或 provider 完整响应。
    print(
        json.dumps(
            {
                "ok": ok,
                "model": settings.DEFAULT_LLM_MODEL,
                "event_count": len(events),
                "token_event_count": len(token_parts),
                "done_event_count": len(done_events),
                "done_status": done_status,
                "last_event_is_done": last_event_is_done,
                "content_matches": content_matches,
                "error_event_count": len(error_events),
                "error_codes": [event.code for event in error_events],
                "unexpected_tool_event_count": len(tool_events),
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
    # asyncio.run 创建事件循环、执行顶层协程并在结束后关闭事件循环。
    # SystemExit 把 smoke 的业务结果转换为 shell/CI 可判断的进程退出码。
    raise SystemExit(asyncio.run(run_text_stream_smoke()))
