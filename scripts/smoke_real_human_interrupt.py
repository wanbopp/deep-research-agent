"""使用真实 provider 验证 Human-in-the-loop Agent 的完整循环.

本脚本把前两个 checkpoint 已分别验证的能力接到一起：

1. 真实模型根据工具 schema 生成 ``ask_human`` ToolCall。
2. 工具内部的 ``interrupt(question)`` 暂停 LangGraph。
3. 外部使用相同 ``thread_id`` 和 ``Command(resume=...)`` 恢复执行。
4. 工具结果被包装成与原 ToolCall 配对的 ToolMessage。
5. 真实模型读取 ToolMessage，再生成不包含工具调用的最终 AIMessage。

脚本只输出结构和布尔检查结果，不输出问题、人工回答、调用 ID、完整
prompt、模型正文或 checkpoint 内容。
"""

import asyncio
import json
import os
from time import perf_counter

os.environ["DEBUG"] = "false"
os.environ["LOG_LEVEL"] = "WARNING"

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage  # noqa: E402
from langchain_core.runnables import RunnableConfig  # noqa: E402
from langgraph.checkpoint.memory import InMemorySaver  # noqa: E402
from langgraph.types import Command  # noqa: E402
from pydantic import SecretStr  # noqa: E402

from app.agents.chat.graph import build_chat_graph  # noqa: E402
from app.agents.chat.nodes import create_chat_node, create_tool_node  # noqa: E402
from app.agents.chat.tools.ask_human import ask_human  # noqa: E402
from app.agents.chat.tools.registry import ToolRegistry  # noqa: E402
from app.core.config import settings  # noqa: E402
from app.schemas.llm import ModelSpec  # noqa: E402
from app.services.llm.factory import create_openai_chat_model  # noqa: E402
from app.services.llm.registry import LLMRegistry  # noqa: E402
from app.services.llm.service import LLMService  # noqa: E402

EXPECTED_REPLY = "REAL_HITL_OK"
HUMAN_REPLY = "approved"
GRAPH_TIMEOUT_SECONDS = 60.0

SMOKE_PROMPT = (
    "Call the ask_human tool exactly once to ask whether this bounded "
    "smoke action is approved. Do not answer directly. "
    "After receiving the human response, do not call any tool again and "
    "reply with exactly REAL_HITL_OK."
)


async def run_real_human_interrupt_smoke() -> int:
    """使用真实 provider 验证 Human-in-the-loop Agent 完整循环."""
    started_at = perf_counter()

    if not settings.OPENAI_API_KEY:
        print(json.dumps({"ok": False, "error_type": "MissingApiKey"}))
        return 1

    spec = ModelSpec(
        alias="primary",
        provider_model=settings.DEFAULT_LLM_MODEL,
        api_key=SecretStr(settings.OPENAI_API_KEY),
        base_url=settings.OPENAI_BASE_URL,
        temperature=settings.DEFAULT_LLM_TEMPERATURE,
        max_tokens=min(settings.MAX_TOKENS, 1024),
        request_timeout_seconds=25.0,
    )

    model_registry = LLMRegistry(
        [spec],
        create_openai_chat_model,
    )

    service = LLMService(
        model_registry,
        max_attempts=1,
        retry_wait_multiplier=0,
        total_timeout_seconds=30.0,
    )

    # ModelSpec 保存 provider 配置；LLMRegistry 把 primary 映射到
    # ModelSpec；LLMService 负责调用、超时和失败处理。

    tool_registry = ToolRegistry((ask_human,))

    # chat_node 使用 Registry
    #     -> 把工具名称、说明和参数结构告诉模型
    chat_node = create_chat_node(
        service,
        aliases=("primary",),
        tool_registry=tool_registry,
    )

    # tool_node 使用 Registry
    # -> 根据模型返回的工具名称找到真实 Python 工具
    tool_node = create_tool_node(
        tool_registry,
        tool_timeout_seconds=1.0,
    )

    graph = build_chat_graph(
        chat_node,
        tool_node=tool_node,
        checkpointer=InMemorySaver(),
    )

    config: RunnableConfig = {
        "configurable": {
            "thread_id": "smoke-real-human-interrupt",
        },
        # 防止路由错误时在 chat 与 tools 之间无限循环。
        "recursion_limit": 6,
    }

    try:
        # 第一次真实模型请求负责做决策，而不是执行工具。正常情况下，
        # 模型返回带 ask_human ToolCall 的 AIMessage；图随后进入 tools，
        # ask_human 调用 interrupt，使 ainvoke 以暂停结果正常返回。
        first_result = await asyncio.wait_for(
            graph.ainvoke(
                {
                    "messages": [
                        HumanMessage(content=SMOKE_PROMPT),
                    ]
                },
                config=config,
            ),
            timeout=GRAPH_TIMEOUT_SECONDS,
        )

        # aget_state 使用 thread_id 读取刚写入的 checkpoint。
        # 暂停时 next 应为 ("tools",)，表示 tools 节点尚未完成，恢复后
        # LangGraph 会从该节点开头重放，而不是从 interrupt 的下一行继续。
        paused_state = await graph.aget_state(config)

        first_result_dict = dict(first_result)
        first_has_interrupt = "__interrupt__" in first_result_dict
        first_messages = first_result["messages"]
        first_ai = first_messages[1] if len(first_messages) > 1 and isinstance(first_messages[1], AIMessage) else None
        tool_calls = first_ai.tool_calls if first_ai is not None else []

        # interrupt 元数据来自 checkpoint，而不是普通 messages 状态。
        # 保存它以后才能证明模型问题确实传入了 ask_human/interrupt。
        interrupts = paused_state.tasks[0].interrupts if paused_state.tasks else ()

        tool_call = tool_calls[0] if len(tool_calls) == 1 else None

        tool_call_id = tool_call["id"] if tool_call is not None else None
        tool_call_id_is_valid = isinstance(tool_call_id, str) and bool(tool_call_id)

        question = tool_call["args"].get("question") if tool_call is not None else None
        question_is_non_empty = isinstance(question, str) and bool(question.strip())

        interrupt_value = interrupts[0].value if len(interrupts) == 1 else None
        interrupt_matches_question = interrupt_value == question

        first_tool_call_count = len(tool_calls)
        tool_name_matches = len(tool_calls) == 1 and tool_calls[0]["name"] == ask_human.name

        # 只有首次真实模型确实选择 ask_human、参数有效并且图真正暂停，
        # 才允许发送人工回答。否则 resume 找不到正确的暂停点，继续执行
        # 只会掩盖第一阶段已经失败的事实。
        can_resume = (
            first_has_interrupt
            and paused_state.next == ("tools",)
            and len(interrupts) == 1
            and first_tool_call_count == 1
            and tool_name_matches
            and question_is_non_empty
            and interrupt_matches_question
            and tool_call_id_is_valid
        )

        if not can_resume:
            print(
                json.dumps(
                    {
                        "ok": False,
                        "stage": "before_resume",
                        "first_has_interrupt": first_has_interrupt,
                        "paused_next": list(paused_state.next),
                        "interrupt_count": len(interrupts),
                        "first_tool_call_count": first_tool_call_count,
                        "tool_name_matches": tool_name_matches,
                        "question_is_non_empty": question_is_non_empty,
                        "interrupt_matches_question": interrupt_matches_question,
                        "tool_call_id_is_valid": tool_call_id_is_valid,
                        "elapsed_ms": round(
                            (perf_counter() - started_at) * 1000,
                            2,
                        ),
                    }
                )
            )
            return 1

        # Command(resume=...) 是 LangGraph 控制输入，不会追加 HumanMessage。
        # 相同 thread_id 让 checkpointer 找到暂停任务；tools 节点重放到
        # 同一个 interrupt 位置时，interrupt 返回 HUMAN_REPLY。
        resumed_result = await asyncio.wait_for(
            graph.ainvoke(
                Command(resume=HUMAN_REPLY),
                config=config,
            ),
            timeout=GRAPH_TIMEOUT_SECONDS,
        )

        completed_state = await graph.aget_state(config)

        # resumed_result["messages"] 是该 thread 的完整累计状态，并非只包含
        # 恢复阶段新增的消息。成功路径应为 Human、带 ToolCall 的 AI、
        # ToolMessage、最终 AIMessage 四条消息。
        messages = resumed_result["messages"]
        tool_message = messages[2] if len(messages) > 2 and isinstance(messages[2], ToolMessage) else None
        final_ai = messages[3] if len(messages) > 3 and isinstance(messages[3], AIMessage) else None

        # ToolCall ID 是模型请求与工具回执之间的关联键。只有 ID 相同，
        # 后续模型才能知道这条 ToolMessage 回答的是哪一个工具请求。
        tool_call_id_matches = (
            tool_message is not None and tool_call_id_is_valid and tool_message.tool_call_id == tool_call_id
        )
        tool_status = tool_message.status if tool_message is not None else None
        tool_content_matches = tool_message is not None and tool_message.content == HUMAN_REPLY

        # 最后一条 AIMessage 没有 tool_calls，才表示 Agent 已决定结束工具循环。
        # 仅检查固定文本不够，否则模型可能绕过 ask_human 直接回答。
        final_has_no_tool_calls = final_ai is not None and not final_ai.tool_calls
        final_content_matches = (
            final_ai is not None and isinstance(final_ai.content, str) and final_ai.content.strip() == EXPECTED_REPLY
        )

        message_types = [type(message).__name__ for message in messages]
        message_types_match = message_types == [
            "HumanMessage",
            "AIMessage",
            "ToolMessage",
            "AIMessage",
        ]

        # ok 汇总本 checkpoint 的全部行为保证。任一条件失败都会返回
        # 非零退出码，使本地命令或 CI 不依赖人眼判断控制台内容。
        ok = (
            len(messages) == 4
            and message_types_match
            and tool_call_id_matches
            and tool_status == "success"
            and tool_content_matches
            and final_has_no_tool_calls
            and final_content_matches
            and completed_state.next == ()
        )

        # 输出可观测但脱敏的验收摘要。布尔值说明每一层是否满足要求，
        # 同时避免泄露问题正文、人工回答、调用 ID 和模型完整内容。
        print(
            json.dumps(
                {
                    "ok": ok,
                    "model": settings.DEFAULT_LLM_MODEL,
                    "first_has_interrupt": first_has_interrupt,
                    "paused_next": list(paused_state.next),
                    "interrupt_count": len(interrupts),
                    "first_tool_call_count": first_tool_call_count,
                    "tool_name_matches": tool_name_matches,
                    "question_is_non_empty": question_is_non_empty,
                    "interrupt_matches_question": interrupt_matches_question,
                    "message_count": len(messages),
                    "message_types": message_types,
                    "tool_call_id_matches": tool_call_id_matches,
                    "tool_status": tool_status,
                    "tool_content_matches": tool_content_matches,
                    "final_has_no_tool_calls": final_has_no_tool_calls,
                    "final_content_matches": final_content_matches,
                    "completed_next": list(completed_state.next),
                    "elapsed_ms": round(
                        (perf_counter() - started_at) * 1000,
                        2,
                    ),
                }
            )
        )

        return 0 if ok else 1
    except Exception as error:
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


if __name__ == "__main__":
    raise SystemExit(asyncio.run(run_real_human_interrupt_smoke()))
