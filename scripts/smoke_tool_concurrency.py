"""Validate concurrent tool execution without calling a model.

AIMessage(tool_calls=[slow, fast, fail])
                    |
                    v
             create_tool_node
                    |
        asyncio.gather 并发调度三条调用
                    |
                    v
ToolMessage[success, success, error]
"""

import asyncio
import inspect
import json
from time import perf_counter
from uuid import UUID

from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.messages.tool import tool_call
from langchain_core.tools import tool
from langgraph.runtime import Runtime

from app.agents.chat.context import ChatRuntimeContext
from app.agents.chat.nodes import create_tool_node
from app.agents.chat.tools.registry import ToolRegistry


@tool
async def slow_value() -> str:
    """Return the slow local smoke value."""
    await asyncio.sleep(0.6)
    return "slow"


@tool
async def fast_value() -> str:
    """Return the fast local smoke value."""
    await asyncio.sleep(0.4)
    return "fast"


@tool
async def controlled_failure() -> str:
    """Raise one controlled local error after a short delay."""
    await asyncio.sleep(0.2)
    raise RuntimeError("controlled local smoke failure")


async def run_tool_concurrency_smoke() -> int:
    """执行本地并发工具 smoke，并返回进程退出码."""
    registry = ToolRegistry(
        (
            slow_value,
            fast_value,
            controlled_failure,
        )
    )

    # 工厂此时只返回内部的 tool_node 函数，不会执行任何工具。
    # 因此变量 node 实际指向已经绑定 registry 和超时配置的 tool_node。
    node = create_tool_node(
        registry,
        tool_timeout_seconds=1.0,
    )

    model_message = AIMessage(
        content="",
        tool_calls=[
            tool_call(name=slow_value.name, args={}, id="call-slow"),
            tool_call(name=fast_value.name, args={}, id="call-fast"),
            tool_call(
                name=controlled_failure.name,
                args={},
                id="call-fail",
            ),
        ],
    )

    started_at = perf_counter()

    # 调用 async 函数会创建协程对象，但函数体要到 await 时才开始推进。
    node_result = node(
        {"messages": [model_message]},
        runtime=Runtime(context=ChatRuntimeContext(user_id=UUID("00000000-0000-4000-8000-000000000001"))),
    )

    # StateNode 为了兼容 LangGraph，同时允许同步和异步节点，
    # 因此静态返回类型是 ChatState | Awaitable[ChatState]。
    # 当前 tool node 应当是异步节点；先做运行时检查，也帮助
    # Pyright 把联合类型收窄为 Awaitable。
    if not inspect.isawaitable(node_result):
        raise TypeError("tool node must return an awaitable")

    state_update = await node_result
    elapsed_seconds = perf_counter() - started_at

    messages = state_update["messages"]
    tool_messages = [message for message in messages if isinstance(message, ToolMessage)]

    ids = [message.tool_call_id for message in tool_messages]
    statuses = [message.status for message in tool_messages]
    contents = [message.content for message in tool_messages]

    order_matches = ids == ["call-slow", "call-fast", "call-fail"]
    statuses_match = statuses == ["success", "success", "error"]
    success_content_matches = contents[:2] == ["slow", "fast"]

    # 1.2 秒是顺序执行总耗时；0.9 秒给本机调度保留余量，
    # 同时仍能区分当前调用是否真正并发。
    within_concurrent_budget = elapsed_seconds < 0.9

    ok = (
        len(tool_messages) == 3
        and order_matches
        and statuses_match
        and success_content_matches
        and within_concurrent_budget
    )

    # 只输出结构、布尔结果和耗时，不输出异常正文。
    print(
        json.dumps(
            {
                "ok": ok,
                "tool_message_count": len(tool_messages),
                "order_matches": order_matches,
                "statuses": statuses,
                "success_content_matches": success_content_matches,
                "within_concurrent_budget": within_concurrent_budget,
                "elapsed_ms": round(elapsed_seconds * 1000, 2),
            }
        )
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(run_tool_concurrency_smoke()))
