"""Use a real provider to verify the Human-in-the-loop interrupt stream.

本脚本只验证 HITL 的“暂停阶段”，不发送 ``Command(resume=...)``：

1. 真实模型根据工具 schema 选择 ``ask_human``；
2. chat 节点的完整 ToolCall 被转换为 ``tool(started)``；
3. tools 节点执行 ``ask_human``，并在 ``interrupt(question)`` 处暂停；
4. LangGraph checkpointer 保存未完成的 tools 节点及 interrupt 元数据；
5. ChatService 读取 snapshot，依次产生 ``interrupt`` 与
   ``done(interrupted)``。

应用层事件证明“客户端看见了什么”，checkpoint 证明“图为什么暂停”。脚本会
交叉检查这两层证据，但只输出数量、状态和布尔结果，不输出 API key、prompt、
人工问题正文、ToolCall 参数、调用 ID 或 checkpoint 内容。
"""

import asyncio
import json
import os
from time import perf_counter

# 这些限制必须在首次导入 app.core.config 前写入环境变量。python-dotenv 默认
# 不覆盖已经存在的变量，因此本 smoke 可以在继续读取本地 key/base_url/model 的
# 同时，将真实请求限制为一次尝试、45 秒 LLM 总预算和 256 个输出 token。
os.environ["DEBUG"] = "false"
os.environ["LOG_LEVEL"] = "WARNING"
os.environ["MAX_LLM_CALL_RETRIES"] = "1"
os.environ["LLM_TOTAL_TIMEOUT"] = "45"
os.environ["MAX_TOKENS"] = "256"

# 这些导入有意放在环境变量之后。E402 noqa 不是忽略设计问题，而是保护
# Settings 的初始化顺序，避免模块导入时先读取到未收紧的运行参数。
from langchain_core.messages import AIMessage  # noqa: E402
from langchain_core.runnables import RunnableConfig  # noqa: E402

from app.agents.chat.runtime import create_chat_runtime  # noqa: E402
from app.agents.chat.tools.ask_human import ask_human  # noqa: E402
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

THREAD_ID = "smoke-chat-interrupt-stream"
GRAPH_TIMEOUT_SECONDS = 60.0

# 提示词只要求模型作出一次工具决策。模型不能自己伪造“已批准”，因为本节的
# 目标就是观察它在得到人工回答之前停下来。恢复阶段会在后续独立验证。
SMOKE_PROMPT = (
    "Call the ask_human tool exactly once to ask whether this bounded "
    "stream smoke action is approved. Do not answer directly."
)


async def run_interrupt_stream_smoke() -> int:
    """Consume one real HITL stream and return a shell-friendly exit code."""
    started_at = perf_counter()

    # 这里只能判断 key 是否存在，不能打印 SecretStr、环境变量或请求 headers。
    if not settings.OPENAI_API_KEY:
        print(json.dumps({"ok": False, "error_type": "MissingApiKey"}))
        return 1

    # 保留 graph 引用有两个目的：
    #   1. ChatService 使用它执行生产流式路径；
    #   2. 流结束后，smoke 使用同一个 graph/checkpointer 读取暂停快照。
    # 如果再次 create_chat_runtime()，会得到另一个 InMemorySaver，自然找不到
    # 当前 thread 的 checkpoint。
    graph = create_chat_runtime()
    service = ChatService(
        graph,
        graph_timeout_seconds=GRAPH_TIMEOUT_SECONDS,
    )

    request = ChatRequest(
        thread_id=THREAD_ID,
        message=SMOKE_PROMPT,
    )

    # 这份 config 与 ChatService._build_config() 使用相同 thread_id 和递归上限。
    # thread_id 是 checkpoint 的“存档键”；只要它不同，读取到的就是另一条会话。
    config: RunnableConfig = {
        "configurable": {
            "thread_id": THREAD_ID,
        },
        "recursion_limit": 8,
    }

    events: list[ChatStreamEvent] = []

    try:
        # 断点 1：观察 event 的实际类型。
        #
        # 第一个关键事件来自 chat update：模型已经完整生成 ask_human ToolCall，
        # 但 Python 工具还没返回，所以状态只能是 started。
        #
        # tools 节点随后进入 interrupt()。它没有返回 ToolMessage，因此这里不应
        # 出现 success/error；astream 结束后，ChatService 会根据 snapshot 补出
        # InterruptStreamEvent 和 DoneStreamEvent(interrupted)。
        async for event in service.stream_turn(request):
            events.append(event)

        # 断点 2：astream 停止并不代表图到达 END。读取 snapshot 后，next
        # 应只包含 tools 节点。这表示 tools 尚未完成，未来 resume 时需要从
        # 这个节点重新执行。
        snapshot = await graph.aget_state(config)
    except Exception as error:
        # ChatService 内部执行错误通常会成为 ErrorStreamEvent；这里主要保护
        # runtime 创建和额外 checkpoint 检查。禁止使用 str(error)，避免上游
        # 异常文本意外携带 URL、请求数据或 provider 响应内容。
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

    # 保留原始索引，验证对外事件的先后顺序，而不只是验证三类事件都存在。
    indexed_tool_events = [(index, event) for index, event in enumerate(events) if isinstance(event, ToolStreamEvent)]
    indexed_interrupt_events = [
        (index, event) for index, event in enumerate(events) if isinstance(event, InterruptStreamEvent)
    ]
    indexed_done_events = [(index, event) for index, event in enumerate(events) if isinstance(event, DoneStreamEvent)]
    token_events = [event for event in events if isinstance(event, TokenStreamEvent)]
    error_events = [event for event in events if isinstance(event, ErrorStreamEvent)]

    # 当前协议一次只允许一个人工问题，因此正常路径应恰好得到三个业务事件：
    # tool(started)、interrupt、done(interrupted)。如果模型重复调用工具，或在
    # ToolCall 前输出了可见解释文字，下面的数量与顺序检查都会暴露差异。
    one_tool_event = len(indexed_tool_events) == 1
    one_interrupt_event = len(indexed_interrupt_events) == 1
    one_done_event = len(indexed_done_events) == 1

    started_event = indexed_tool_events[0][1] if one_tool_event else None
    interrupt_event = indexed_interrupt_events[0][1] if one_interrupt_event else None
    done_event = indexed_done_events[0][1] if one_done_event else None

    tool_statuses = [event.status for _, event in indexed_tool_events]
    tool_started_matches = (
        started_event is not None
        and started_event.name == ask_human.name
        and started_event.status == "started"
        and bool(started_event.tool_call_id.strip())
    )

    # InterruptStreamEvent 没有 tool_call_id，这是有意的公开协议设计：客户端只
    # 需要显示问题，不应理解 LangGraph 内部消息。要证明它属于刚才那次 ToolCall，
    # smoke 必须继续查看 checkpoint 中待处理的 AIMessage。
    values = snapshot.values
    raw_messages = values.get("messages") if isinstance(values, dict) else None
    messages = raw_messages if isinstance(raw_messages, list) else []

    # 从后向前寻找最近一条带 ToolCall 的 AIMessage。暂停发生时还没有对应的
    # ToolMessage，因此这条 AIMessage 正是 tools 节点正在处理的输入。
    pending_ai = next(
        (message for message in reversed(messages) if isinstance(message, AIMessage) and message.tool_calls),
        None,
    )
    pending_tool_calls = pending_ai.tool_calls if pending_ai is not None else []
    pending_tool_call = pending_tool_calls[0] if len(pending_tool_calls) == 1 else None

    pending_name = pending_tool_call.get("name") if pending_tool_call is not None else None
    pending_call_id = pending_tool_call.get("id") if pending_tool_call is not None else None
    pending_args = pending_tool_call.get("args") if pending_tool_call is not None else None
    pending_question = pending_args.get("question") if isinstance(pending_args, dict) else None

    # 一个 snapshot 可能包含多个 task，interrupt 属于具体 task。这里与生产
    # ChatService 一样展开全部 task，避免把 tasks[0] 当作永久成立的框架保证。
    checkpoint_interrupts = tuple(item for task in snapshot.tasks for item in task.interrupts)
    checkpoint_question = checkpoint_interrupts[0].value if len(checkpoint_interrupts) == 1 else None

    # 断点 3：下面三条关联共同回答“为什么这个 interrupt 属于这个 started”：
    #   1. pending ToolCall 的名称是 ask_human；
    #   2. started ID 与 pending ToolCall ID 相同；
    #   3. 对外问题、checkpoint interrupt 值和 ToolCall question 参数相同。
    pending_tool_name_matches = pending_name == ask_human.name
    tool_call_id_matches_checkpoint = (
        started_event is not None
        and isinstance(pending_call_id, str)
        and bool(pending_call_id.strip())
        and started_event.tool_call_id == pending_call_id
    )
    question_is_non_empty = interrupt_event is not None and bool(interrupt_event.question.strip())
    question_matches_checkpoint = (
        interrupt_event is not None
        and isinstance(checkpoint_question, str)
        and interrupt_event.question == checkpoint_question
    )
    question_matches_tool_args = (
        interrupt_event is not None
        and isinstance(pending_question, str)
        and interrupt_event.question == pending_question
    )

    paused_next_matches = snapshot.next == ("tools",)
    done_status = done_event.status if done_event is not None else None
    last_event_is_done = bool(events) and isinstance(events[-1], DoneStreamEvent)

    # started -> interrupt -> done 的位置必须严格递增。done 位于最后，表示本次
    # HTTP/SSE 流可以正常关闭；interrupted 则告诉调用方应展示问题并等待 resume。
    event_order_matches = (
        one_tool_event
        and one_interrupt_event
        and one_done_event
        and indexed_tool_events[0][0] < indexed_interrupt_events[0][0]
        and indexed_interrupt_events[0][0] < indexed_done_events[0][0]
    )

    # 断点 4：这是最终行为保证。这里明确拒绝 token、success/error 工具事件和
    # ErrorStreamEvent，因为人工尚未回答，ask_human 不可能已经成功返回。
    ok = (
        one_tool_event
        and tool_statuses == ["started"]
        and tool_started_matches
        and one_interrupt_event
        and question_is_non_empty
        and one_done_event
        and done_status == "interrupted"
        and event_order_matches
        and last_event_is_done
        and paused_next_matches
        and len(checkpoint_interrupts) == 1
        and len(pending_tool_calls) == 1
        and pending_tool_name_matches
        and tool_call_id_matches_checkpoint
        and question_matches_checkpoint
        and question_matches_tool_args
        and not token_events
        and not error_events
    )

    # 所有敏感或可识别内容只参与内存比较。摘要不打印 question、prompt、args、
    # ToolCall ID、checkpoint 数据或完整模型输出，可以安全用于手工验收和 CI 日志。
    print(
        json.dumps(
            {
                "ok": ok,
                "model": settings.DEFAULT_LLM_MODEL,
                "event_count": len(events),
                "tool_event_count": len(indexed_tool_events),
                "tool_statuses": tool_statuses,
                "tool_started_matches": tool_started_matches,
                "interrupt_event_count": len(indexed_interrupt_events),
                "question_is_non_empty": question_is_non_empty,
                "done_event_count": len(indexed_done_events),
                "done_status": done_status,
                "event_order_matches": event_order_matches,
                "last_event_is_done": last_event_is_done,
                "paused_next_matches": paused_next_matches,
                "checkpoint_interrupt_count": len(checkpoint_interrupts),
                "pending_tool_call_count": len(pending_tool_calls),
                "pending_tool_name_matches": pending_tool_name_matches,
                "tool_call_id_matches_checkpoint": (tool_call_id_matches_checkpoint),
                "question_matches_checkpoint": question_matches_checkpoint,
                "question_matches_tool_args": question_matches_tool_args,
                "unexpected_token_event_count": len(token_events),
                "error_event_count": len(error_events),
                "error_codes": [event.code for event in error_events],
                "elapsed_ms": round(
                    (perf_counter() - started_at) * 1000,
                    2,
                ),
            }
        )
    )

    return 0 if ok else 1


if __name__ == "__main__":
    # asyncio.run 创建顶层事件循环；SystemExit 把行为验收结果映射为 0/1，
    # PowerShell 和 CI 都能据此判断脚本是否真正通过。
    raise SystemExit(asyncio.run(run_interrupt_stream_smoke()))
