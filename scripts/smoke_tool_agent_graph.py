"""使用真实 provider 验证完整 chat/tool Agent 循环."""

import asyncio
import json
import os
from time import perf_counter
from uuid import UUID

# 必须在导入任何 app 模块前设置，避免 provider 调试日志输出正文。
os.environ["DEBUG"] = "false"
os.environ["LOG_LEVEL"] = "WARNING"

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage  # noqa: E402
from langgraph.errors import GraphBubbleUp  # noqa: E402
from pydantic import SecretStr  # noqa: E402

from app.agents.chat.graph import build_chat_graph  # noqa: E402
from app.agents.chat.context import ChatRuntimeContext  # noqa: E402
from app.agents.chat.nodes import create_chat_node, create_tool_node  # noqa: E402
from app.agents.chat.tools.current_time import get_current_utc_time  # noqa: E402
from app.agents.chat.tools.registry import ToolRegistry  # noqa: E402
from app.core.config import settings  # noqa: E402
from app.schemas.llm import ModelSpec  # noqa: E402
from app.services.llm.factory import create_openai_chat_model  # noqa: E402
from app.services.llm.registry import LLMRegistry  # noqa: E402
from app.services.llm.service import LLMService  # noqa: E402
from app.tools import (  # noqa: E402
    RuntimeToolRegistry,
    ToolDescriptor,
    ToolExecutor,
    ToolExposure,
    ToolRisk,
)
from app.tools.adapters import LangChainToolAdapter  # noqa: E402

EXPECTED_REPLY = "REAL_TOOL_AGENT_OK"
GRAPH_TIMEOUT_SECONDS = 60.0
SMOKE_CONTEXT = ChatRuntimeContext(user_id=UUID("00000000-0000-4000-8000-000000000001"))

SMOKE_PROMPT = (
    "Call the get_current_utc_time tool exactly once. "
    "After receiving its result, do not call any tool again and reply "
    "with exactly REAL_TOOL_AGENT_OK."
)


async def run_tool_agent_graph_smoke() -> int:
    """执行一次有界真实工具 Agent 调用，并返回进程退出码."""
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
        # 整图包含两次模型调用，为外层 60 秒预算预留空间。
        request_timeout_seconds=25.0,
    )

    model_registry = LLMRegistry(
        [spec],
        create_openai_chat_model,
    )
    service = LLMService(
        model_registry,
        # 配置或 provider 错误时不重复付费请求。
        max_attempts=1,
        retry_wait_multiplier=0,
        total_timeout_seconds=30.0,
    )

    # 同一个 Registry 同时决定模型可见 schema 和 Python 执行白名单。
    tool_registry = ToolRegistry((get_current_utc_time,))
    runtime_registry = RuntimeToolRegistry()
    runtime_registry.register(
        ToolDescriptor(
            name=get_current_utc_time.name,
            namespace="local",
            exposure=ToolExposure.MODEL,
            risk=ToolRisk.READ_ONLY,
            timeout_seconds=5.0,
            output_token_limit=128,
            supports_parallel=True,
            requires_approval=False,
        ),
        LangChainToolAdapter(get_current_utc_time),
    )
    executor = ToolExecutor(runtime_registry, passthrough_exception_types=(GraphBubbleUp,))
    chat_node = create_chat_node(
        service,
        aliases=("primary",),
        tool_registry=tool_registry,
    )
    tool_node = create_tool_node(tool_registry, executor=executor)
    graph = build_chat_graph(
        chat_node,
        tool_node=tool_node,
    )

    try:
        # LLMService 限制每次模型调用；外层 timeout 限制整张图。
        async with asyncio.timeout(GRAPH_TIMEOUT_SECONDS):
            final_state = await graph.ainvoke(
                {
                    "messages": [
                        HumanMessage(content=SMOKE_PROMPT),
                    ]
                },
                # 限制图节点步数，防止模型持续重复调用工具。
                config={"recursion_limit": 6},
                context=SMOKE_CONTEXT,
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
    tool_message = messages[2] if len(messages) > 2 and isinstance(messages[2], ToolMessage) else None
    final_ai = messages[3] if len(messages) > 3 and isinstance(messages[3], AIMessage) else None

    first_tool_call_count = len(first_ai.tool_calls) if first_ai is not None else 0
    first_tool_call_id = (
        first_ai.tool_calls[0]["id"] if first_ai is not None and len(first_ai.tool_calls) == 1 else None
    )

    tool_call_id_matches = (
        tool_message is not None
        and isinstance(first_tool_call_id, str)
        and tool_message.tool_call_id == first_tool_call_id
    )
    tool_status = tool_message.status if tool_message is not None else None
    final_has_no_tool_calls = final_ai is not None and not final_ai.tool_calls
    final_content_matches = (
        final_ai is not None and isinstance(final_ai.content, str) and final_ai.content.strip() == EXPECTED_REPLY
    )

    expected_types = [
        "HumanMessage",
        "AIMessage",
        "ToolMessage",
        "AIMessage",
    ]
    message_types = [type(message).__name__ for message in messages]

    ok = (
        len(messages) == 4
        and message_types == expected_types
        and first_tool_call_count == 1
        and tool_call_id_matches
        and tool_status == "success"
        and final_has_no_tool_calls
        and final_content_matches
    )

    # 不输出 prompt、工具结果、模型正文、调用 ID 或 provider 响应。
    print(
        json.dumps(
            {
                "ok": ok,
                "model": settings.DEFAULT_LLM_MODEL,
                "message_count": len(messages),
                "message_types": message_types,
                "first_tool_call_count": first_tool_call_count,
                "tool_call_id_matches": tool_call_id_matches,
                "tool_status": tool_status,
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
    """ChatOpenAI request_timeout_seconds=25.
            -> 单次 provider 网络请求上限
        LLMService total_timeout_seconds=30
          -> 单次 chat node 内 retry/fallback 总预算
        asyncio.timeout(60)
          -> 整张图上限，包括两次模型调用和一次工具执行
    """
    raise SystemExit(asyncio.run(run_tool_agent_graph_smoke()))
