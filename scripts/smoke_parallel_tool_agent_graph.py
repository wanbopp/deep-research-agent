"""使用真实 provider 验证同轮双 ToolCall 的完整 Agent 循环."""

import asyncio
import json
import os
from time import perf_counter
from typing import Literal

os.environ["DEBUG"] = "false"
os.environ["LOG_LEVEL"] = "WARNING"

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage  # noqa: E402
from langchain_core.tools import tool  # noqa: E402
from pydantic import SecretStr  # noqa: E402

from app.agents.chat.graph import build_chat_graph  # noqa: E402
from app.agents.chat.nodes import create_chat_node, create_tool_node  # noqa: E402
from app.agents.chat.tools.registry import ToolRegistry  # noqa: E402
from app.core.config import settings  # noqa: E402
from app.schemas.llm import ModelSpec  # noqa: E402
from app.services.llm.factory import create_openai_chat_model  # noqa: E402
from app.services.llm.registry import LLMRegistry  # noqa: E402
from app.services.llm.service import LLMService  # noqa: E402

EXPECTED_REPLY = "REAL_PARALLEL_TOOL_AGENT_OK"
GRAPH_TIMEOUT_SECONDS = 60.0

SMOKE_PROMPT = (
    "Call fetch_research_signal exactly twice in the same assistant turn: "
    "once with source alpha and once with source beta. "
    "After receiving both tool results, do not call any tool again and reply "
    "with exactly REAL_PARALLEL_TOOL_AGENT_OK."
)


@tool
async def fetch_research_signal(
    source: Literal["alpha", "beta"],
) -> str:
    """Fetch one bounded research signal from the requested source."""
    delays = {
        "alpha": 0.6,
        "beta": 0.4,
    }
    await asyncio.sleep(delays[source])
    return f"{source}-signal"


async def run_parallel_tool_agent_graph_smoke() -> int:
    """执行一次真实双工具调用 Agent smoke，并返回进程退出码."""
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

    tool_registry = ToolRegistry((fetch_research_signal,))

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
    )

    try:
        async with asyncio.timeout(GRAPH_TIMEOUT_SECONDS):
            final_state = await graph.ainvoke(
                {
                    "messages": [
                        HumanMessage(content=SMOKE_PROMPT),
                    ]
                },
                config={"recursion_limit": 6},
            )
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
    messages = final_state["messages"]

    first_ai = messages[1] if len(messages) > 1 and isinstance(messages[1], AIMessage) else None
    first_tool_message = messages[2] if len(messages) > 2 and isinstance(messages[2], ToolMessage) else None
    second_tool_message = messages[3] if len(messages) > 3 and isinstance(messages[3], ToolMessage) else None
    final_ai = messages[4] if len(messages) > 4 and isinstance(messages[4], AIMessage) else None

    tool_calls = first_ai.tool_calls if first_ai is not None else []

    tool_statuses = [message.status for message in (first_tool_message, second_tool_message) if message is not None]

    final_has_no_tool_calls = final_ai is not None and not final_ai.tool_calls

    final_content_matches = (
        final_ai is not None and isinstance(final_ai.content, str) and final_ai.content.strip() == EXPECTED_REPLY
    )

    # 检查模型是否在第一轮准确生成了两条调用。
    first_tool_call_count = len(tool_calls)

    # 两条调用都必须指向 fetch_research_signal。
    tool_names_match = len(tool_calls) == 2 and all(call["name"] == fetch_research_signal.name for call in tool_calls)

    # 不要求模型先返回 alpha 还是 beta，只要求二者各有一次。
    tool_args = [call["args"] for call in tool_calls]
    tool_args_match = len(tool_args) == 2 and {"source": "alpha"} in tool_args and {"source": "beta"} in tool_args

    # 提取模型生成的两个调用编号。
    tool_call_ids = [call["id"] for call in tool_calls]

    # 提取两个工具执行结果所引用的调用编号。
    tool_message_ids = [
        message.tool_call_id
        for message in (
            first_tool_message,
            second_tool_message,
        )
        if message is not None
    ]

    # 两个调用 ID 必须都是非空字符串，并且不能重复。
    tool_call_ids_are_valid = (
        len(tool_call_ids) == 2
        and all(isinstance(call_id, str) and bool(call_id) for call_id in tool_call_ids)
        and len(set(tool_call_ids)) == 2
    )

    # ToolMessage 的 ID 顺序必须与 ToolCall 完全相同。
    tool_call_ids_match = tool_call_ids_are_valid and tool_message_ids == tool_call_ids

    # 两个本地工具都应该成功执行。
    tool_statuses_match = tool_statuses == [
        "success",
        "success",
    ]

    # 完整 Agent 循环应恰好产生这五条消息。
    expected_message_types = [
        "HumanMessage",
        "AIMessage",
        "ToolMessage",
        "ToolMessage",
        "AIMessage",
    ]
    message_types = [type(message).__name__ for message in messages]
    message_types_match = message_types == expected_message_types

    # 汇总所有必须同时成立的执行保证。
    ok = (
        len(messages) == 5
        and first_ai is not None
        and first_tool_message is not None
        and second_tool_message is not None
        and final_ai is not None
        and first_tool_call_count == 2
        and tool_names_match
        and tool_args_match
        and tool_call_ids_match
        and tool_statuses_match
        and message_types_match
        and final_has_no_tool_calls
        and final_content_matches
    )

    # 只输出安全摘要，不输出 prompt、模型正文、工具结果或调用 ID。
    print(
        json.dumps(
            {
                "ok": ok,
                "model": settings.DEFAULT_LLM_MODEL,
                "message_count": len(messages),
                "message_types": message_types,
                "first_tool_call_count": first_tool_call_count,
                "tool_names_match": tool_names_match,
                "tool_args_match": tool_args_match,
                "tool_call_ids_match": tool_call_ids_match,
                "tool_statuses": tool_statuses,
                "final_has_no_tool_calls": final_has_no_tool_calls,
                "final_content_matches": final_content_matches,
                "elapsed_ms": round(
                    (perf_counter() - started_at) * 1000,
                    2,
                ),
            }
        )
    )

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(run_parallel_tool_agent_graph_smoke()))
