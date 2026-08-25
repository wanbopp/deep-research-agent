"""Use a real provider to verify the ChatService application boundary.

这个 smoke 关注的不是单次模型文本质量，而是 ChatService 是否正确完成：

1. 把 ChatRequest.message 转换为 HumanMessage。
2. 把 ChatRequest.thread_id 放入 LangGraph config。
3. 复用同一个 compiled graph 和 checkpointer 保存多轮历史。
4. 把最终 AIMessage 转换为稳定的 ChatResponse。

脚本会发送两次受控真实请求。第二轮必须从相同 thread_id 的 checkpoint 中
读取第一轮代号，才能证明 Service 没有为每次调用重建 runtime。
"""

import asyncio
import json
import os
from time import perf_counter
from uuid import UUID

# Settings 在 app.core.config 首次导入时创建，因此受控请求参数必须先写入
# os.environ。load_dotenv 默认不会覆盖已经存在的环境变量。
os.environ["DEBUG"] = "false"
os.environ["LOG_LEVEL"] = "WARNING"
os.environ["MAX_LLM_CALL_RETRIES"] = "1"
os.environ["LLM_TOTAL_TIMEOUT"] = "30"
os.environ["MAX_TOKENS"] = "512"

from app.agents.chat.runtime import create_chat_runtime  # noqa: E402
from app.core.config import settings  # noqa: E402
from app.infrastructure.chat_guard import InProcessChatExecutionGuard  # noqa: E402
from app.schemas.chat import ChatRequest, ChatResponse  # noqa: E402
from app.services.chat import ChatService  # noqa: E402

THREAD_ID = "smoke-chat-service"
CODE_WORD = "BLUEBERRY"
FIRST_EXPECTED = "TURN_ONE_OK"
SECOND_EXPECTED = "MEMORY_OK: BLUEBERRY"
GRAPH_TIMEOUT_SECONDS = 60.0
SMOKE_USER_ID = UUID("00000000-0000-4000-8000-000000000001")

# Prompt 明确禁止工具调用，使本实验只验证 completed 分支。
# ask_human/interrupt 已由前一个 checkpoint 的真实 smoke 独立验证。
FIRST_PROMPT = (
    "Do not call any tool. Remember the code word BLUEBERRY for this conversation and reply with exactly TURN_ONE_OK."
)
SECOND_PROMPT = (
    "Do not call any tool. What code word did I ask you to remember? Reply with exactly MEMORY_OK: BLUEBERRY."
)


async def run_chat_service_smoke() -> int:
    """Run two real turns through one ChatService and return an exit code."""
    started_at = perf_counter()

    try:
        # runtime 和 service 只能创建一次。两轮复用同一个 InMemorySaver，
        # 相同 thread_id 才能定位并继续第一轮写入的 checkpoint。
        runtime = create_chat_runtime()
        service = ChatService(
            runtime,
            # 独立 smoke 没有 FastAPI lifespan；只在本进程内验证执行权生命周期。
            execution_guard=InProcessChatExecutionGuard(),
            graph_timeout_seconds=GRAPH_TIMEOUT_SECONDS,
        )

        first_result = await service.run_turn(
            ChatRequest(
                thread_id=THREAD_ID,
                message=FIRST_PROMPT,
            ),
            user_id=SMOKE_USER_ID,
        )

        # 第二轮只提交当前的新用户消息，不重复传第一轮历史。
        # add_messages 与 checkpointer 会把它追加到同一 thread 的状态中。
        second_result = await service.run_turn(
            ChatRequest(
                thread_id=THREAD_ID,
                message=SECOND_PROMPT,
            ),
            user_id=SMOKE_USER_ID,
        )

        # Service 已隐藏 LangGraph 内部状态；smoke 为验证历史确实累计，
        # 才直接从同一个 runtime 读取 checkpoint。生产 route 不应这样做。
        # Service 写入的是“可信 user_id + 公开 thread_id”组成的内部 key，白盒读取
        # 必须复用同一构造方法。直接使用 THREAD_ID 会查看另一个空的身份空间。
        config = ChatService._build_config(
            user_id=SMOKE_USER_ID,
            public_thread_id=THREAD_ID,
        )
        snapshot = await runtime.aget_state(config)
        raw_messages = snapshot.values.get("messages", [])
        messages = list(raw_messages) if isinstance(raw_messages, list) else []
    except Exception as error:
        # 异常摘要只暴露类型和可选状态码，不打印 str(error)，避免 provider
        # 响应、prompt 或其他可能包含敏感信息的内容进入控制台。
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

    # run_turn 的返回类型是 ChatResponse | ChatInterrupt。不能直接读取
    # .message；先检查类型，也能发现模型是否意外选择了 ask_human。
    first_is_response = isinstance(first_result, ChatResponse)
    second_is_response = isinstance(second_result, ChatResponse)

    thread_matches = (
        first_is_response
        and second_is_response
        and first_result.thread_id == THREAD_ID
        and second_result.thread_id == THREAD_ID
    )
    roles_match = (
        first_is_response
        and second_is_response
        and first_result.message.role == "assistant"
        and second_result.message.role == "assistant"
    )
    first_content_matches = first_is_response and first_result.message.content.strip() == FIRST_EXPECTED
    second_content_matches = second_is_response and second_result.message.content.strip() == SECOND_EXPECTED

    # 没有工具调用时，两轮状态应恰好是 Human、AI、Human、AI。
    # 这比只检查第二轮文本更强：它还证明消息由 reducer 追加，而非覆盖。
    message_types = [type(message).__name__ for message in messages]
    message_history_matches = message_types == [
        "HumanMessage",
        "AIMessage",
        "HumanMessage",
        "AIMessage",
    ]

    ok = (
        first_is_response
        and second_is_response
        and thread_matches
        and roles_match
        and first_content_matches
        and second_content_matches
        and message_history_matches
    )

    # 只输出结构、布尔结果和耗时；不输出 prompt、代号、模型完整回答、
    # key、消息 ID 或 checkpoint 对象。
    print(
        json.dumps(
            {
                "ok": ok,
                "model": settings.DEFAULT_LLM_MODEL,
                "first_response_type": type(first_result).__name__,
                "second_response_type": type(second_result).__name__,
                "thread_matches": thread_matches,
                "roles_match": roles_match,
                "first_content_matches": first_content_matches,
                "second_content_matches": second_content_matches,
                "message_count": len(messages),
                "message_types": message_types,
                "elapsed_ms": round(
                    (perf_counter() - started_at) * 1000,
                    2,
                ),
            }
        )
    )

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(run_chat_service_smoke()))
