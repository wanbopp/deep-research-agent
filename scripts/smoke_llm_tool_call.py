"""使用真实 provider 验证模型工具调用请求."""

import asyncio
import json
import os
from time import perf_counter

# 必须在导入任何 app 模块之前设置。
# development 环境默认会开启 DEBUG 和 INFO 日志，可能输出 SDK traceback。
# 冒烟测试只需要安全 JSON 摘要，因此主动降低日志级别。
os.environ["DEBUG"] = "false"
os.environ["LOG_LEVEL"] = "WARNING"

# 这些导入必须位于环境变量设置之后。
# noqa 用来告诉 Ruff：这里的延迟导入是有意为之。
from langchain_core.messages import AIMessage, HumanMessage  # noqa: E402
from pydantic import SecretStr  # noqa: E402

from app.agents.chat.tools.current_time import get_current_utc_time  # noqa: E402
from app.core.config import settings  # noqa: E402
from app.schemas.llm import ModelSpec  # noqa: E402
from app.services.llm.factory import create_openai_chat_model  # noqa: E402
from app.services.llm.registry import LLMRegistry  # noqa: E402
from app.services.llm.service import LLMService  # noqa: E402


async def run_llm_tool_call_smoke() -> int:
    """执行一次真实工具调用决策，并返回适合进程使用的退出码."""
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
    # LLMService 只认识 primary，不需要知道 provider 的真实模型名称。
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

    try:
        response = await service.call(
            [
                HumanMessage(
                    content=("Call the get_current_utc_time tool exactly once. Do not answer with a time directly.")
                )
            ],
            aliases=("primary",),
            tools=(get_current_utc_time,),
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

    is_ai_message = isinstance(response, AIMessage)
    tool_calls = response.tool_calls if is_ai_message else []
    first_call = tool_calls[0] if tool_calls else None

    name_matches = first_call is not None and first_call["name"] == get_current_utc_time.name
    args_match = first_call is not None and first_call["args"] == {}
    has_call_id = first_call is not None and isinstance(first_call["id"], str) and bool(first_call["id"])

    ok = is_ai_message and len(tool_calls) == 1 and name_matches and args_match and has_call_id

    elapsed_ms = round(
        (perf_counter() - started_at) * 1000,
        2,
    )

    print(
        json.dumps(
            {
                "ok": ok,
                "model": settings.DEFAULT_LLM_MODEL,
                "response_type": type(response).__name__,
                "tool_call_count": len(tool_calls),
                "tool_name_matches": name_matches,
                "args_match": args_match,
                "has_call_id": has_call_id,
                "elapsed_ms": elapsed_ms,
            }
        )
    )

    return 0 if ok else 1


if __name__ == "__main__":
    # asyncio.run() 的返回值就是 smoke 协程的整数结果。
    # SystemExit 把这个整数转换成操作系统可观察的进程退出码。
    raise SystemExit(asyncio.run(run_llm_tool_call_smoke()))
