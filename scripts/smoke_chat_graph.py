"""对最小聊天图执行一次受控真实 provider 冒烟测试."""

import asyncio
import json
import os
from time import perf_counter
from uuid import UUID

# 必须在导入 app 模块之前设置。
# development 环境默认会开启 DEBUG 和 INFO 日志，可能输出 SDK traceback。
# 冒烟测试只需要安全 JSON 摘要，因此主动降低日志级别。
os.environ["DEBUG"] = "false"
os.environ["LOG_LEVEL"] = "WARNING"

# 这些导入必须位于环境变量设置之后。
# noqa 用来告诉 Ruff：这里的延迟导入是有意为之。
from langchain_core.messages import AIMessage, HumanMessage  # noqa: E402
from pydantic import SecretStr  # noqa: E402

from app.agents.chat.graph import build_chat_graph  # noqa: E402
from app.agents.chat.context import ChatRuntimeContext  # noqa: E402
from app.agents.chat.nodes import create_chat_node  # noqa: E402
from app.core.config import settings  # noqa: E402
from app.schemas.llm import ModelSpec  # noqa: E402
from app.services.llm.factory import create_openai_chat_model  # noqa: E402
from app.services.llm.registry import LLMRegistry  # noqa: E402
from app.services.llm.service import LLMService  # noqa: E402

# 使用固定短文本验证模型是否遵循了输入要求。
# 输出时只记录是否匹配，不打印完整模型响应。
EXPECTED_REPLY = "REAL_GRAPH_OK"

# 这是发送给真实模型的最小测试消息。
SMOKE_PROMPT = "Reply with exactly REAL_GRAPH_OK and nothing else."
SMOKE_CONTEXT = ChatRuntimeContext(user_id=UUID("00000000-0000-4000-8000-000000000001"))


async def run_chat_graph_smoke() -> int:
    """执行一次真实聊天图调用，并返回适合进程使用的退出码."""
    started_at = perf_counter()

    # 在创建模型前检查密钥，避免把配置错误误判为 Agent 错误。
    # 这里只判断密钥是否存在，绝不打印密钥内容。
    if not settings.OPENAI_API_KEY:
        print(
            json.dumps(
                {
                    "ok": False,
                    "error_type": "MissingApiKey",
                }
            )
        )
        return 1

    # ModelSpec 描述一个可由稳定 alias 查找的真实模型。
    # Agent node 只认识 primary，不需要知道 provider 的真实模型名称。
    spec = ModelSpec(
        alias="primary",
        provider_model=settings.DEFAULT_LLM_MODEL,
        api_key=SecretStr(settings.OPENAI_API_KEY),
        base_url=settings.OPENAI_BASE_URL,
        temperature=settings.DEFAULT_LLM_TEMPERATURE,
        # 这是简单回复，但推理模型可能消耗额外 completion token。
        # 1024 保持有界，同时为真实 provider 留出合理空间。
        max_tokens=min(settings.MAX_TOKENS, 1024),
    )

    # Registry 负责：
    # primary alias -> ModelSpec -> 通过 factory 创建 ChatOpenAI。
    registry = LLMRegistry(
        [spec],
        create_openai_chat_model,
    )

    # LLMService 负责真实调用过程中的超时、重试和 fallback 边界。
    # 冒烟测试限制为一次 attempt，配置或权限错误时不重复付费请求。
    service = LLMService(
        registry,
        max_attempts=1,
        retry_wait_multiplier=0,
        total_timeout_seconds=min(
            settings.LLM_TOTAL_TIMEOUT,
            60,
        ),
    )

    # chat node 只持有 LLMService 和 alias。
    # 它不直接创建 ChatOpenAI，也不读取环境变量。
    chat_node = create_chat_node(
        service,
        aliases=("primary",),
    )

    # 当前最小图的执行顺序：
    # START -> chat -> END
    graph = build_chat_graph(chat_node)

    try:
        # 初始状态只包含一条 HumanMessage。
        # chat node 调用真实模型后返回一条 AIMessage 增量；
        # add_messages reducer 会把它追加到初始消息之后。
        final_state = await graph.ainvoke(
            {
                "messages": [
                    HumanMessage(content=SMOKE_PROMPT),
                ]
            },
            context=SMOKE_CONTEXT,
        )
    except Exception as error:
        # smoke 脚本允许在最外层捕获异常，以便输出安全诊断摘要。
        # 不使用 str(error)，因为 provider 异常可能包含请求或响应内容。
        print(
            json.dumps(
                {
                    "ok": False,
                    "error_type": type(error).__name__,
                    "status_code": getattr(
                        error,
                        "status_code",
                        None,
                    ),
                    "elapsed_ms": round(
                        (perf_counter() - started_at) * 1000,
                        2,
                    ),
                }
            )
        )
        return 1
        # graph.ainvoke() 返回最终状态。
        # 成功时应包含原始 HumanMessage 和新增的 AIMessage。
    messages = final_state["messages"]
    last_message = messages[-1] if messages else None

    # AIMessage.content 的静态类型不只包含 str，
    # 因此必须先确认它是字符串，才能调用 strip()。
    content_matches = (
        isinstance(last_message, AIMessage)
        and isinstance(last_message.content, str)
        and last_message.content.strip() == EXPECTED_REPLY
    )

    # 这里验证的不只是“模型返回了内容”，而是四层行为保证：
    # 1. reducer 保留了输入消息；
    # 2. node 增加了一条模型消息；
    # 3. 消息类型顺序正确；
    # 4. 模型回复符合本次测试要求。
    ok = (
        len(messages) == 2
        and isinstance(messages[0], HumanMessage)
        and isinstance(last_message, AIMessage)
        and content_matches
    )

    # 只输出类型、数量、布尔结果和耗时。
    # 不输出 API key、完整 prompt 或完整模型正文。
    print(
        json.dumps(
            {
                "ok": ok,
                "model": settings.DEFAULT_LLM_MODEL,
                "message_count": len(messages),
                "message_types": [type(message).__name__ for message in messages],
                "content_matches": content_matches,
                "elapsed_ms": round(
                    (perf_counter() - started_at) * 1000,
                    2,
                ),
            }
        )
    )

    # 只有所有检查都通过时返回 0。
    # 如果 provider 返回 200，但状态或内容不符合要求，仍然返回 1。
    return 0 if ok else 1


if __name__ == "__main__":
    # asyncio.run() 的返回值就是 run_chat_graph_smoke() 的整数结果。
    # SystemExit 把这个整数转换成操作系统可观察的进程退出码。
    raise SystemExit(asyncio.run(run_chat_graph_smoke()))
