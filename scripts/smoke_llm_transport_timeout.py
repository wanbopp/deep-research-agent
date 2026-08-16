"""使用真实 provider 验证模型 transport timeout."""

import asyncio
import json
import os
from time import perf_counter

# 必须在导入 app 模块前设置。
# 避免开发环境默认开启 DEBUG，从而打印 provider 请求 traceback。
os.environ["DEBUG"] = "false"
os.environ["LOG_LEVEL"] = "WARNING"

# 这些模块会间接导入应用日志和配置，因此需要延迟到
# DEBUG、LOG_LEVEL 设置完成之后。
from langchain_core.messages import HumanMessage  # noqa: E402
from pydantic import SecretStr  # noqa: E402

from app.core.config import settings  # noqa: E402
from app.schemas.llm import ModelSpec  # noqa: E402
from app.services.llm.errors import AllModelsFailedError  # noqa: E402
from app.services.llm.factory import create_openai_chat_model  # noqa: E402
from app.services.llm.registry import LLMRegistry  # noqa: E402
from app.services.llm.service import LLMService  # noqa: E402

# 单次网络请求只允许等待 1 毫秒。
# 真实互联网请求基本不可能在该时间内完成，
# 因此预期 OpenAI SDK 抛出 APITimeoutError。
TRANSPORT_TIMEOUT_SECONDS = 0.001

# LLMService 的整体预算必须大于 transport timeout。
# 这样 transport 异常有时间进入 LLMService 的异常处理逻辑。
TOTAL_TIMEOUT_SECONDS = 5.0

# LLMService 不会直接向上暴露 provider 异常对象，
# 而是把安全的类型名称保存到 ModelFailure 中。
EXPECTED_ERROR_TYPE = "APITimeoutError"


async def run_transport_timeout_smoke() -> int:
    """执行一次真实网络超时实验，并返回进程退出码."""
    started_at = perf_counter()

    # 只判断 key 是否存在，绝不输出 key 的内容。
    if not settings.OPENAI_API_KEY:
        print(
            json.dumps(
                {
                    "ok": False,
                    "expected_timeout": False,
                    "error_type": "MissingApiKey",
                }
            )
        )
        return 1

    # 为本次 smoke 单独创建模型配置。
    # 这里故意把单次 provider 网络超时设置为 1 毫秒。
    spec = ModelSpec(
        alias="primary",
        provider_model=settings.DEFAULT_LLM_MODEL,
        api_key=SecretStr(settings.OPENAI_API_KEY),
        base_url=settings.OPENAI_BASE_URL,
        temperature=settings.DEFAULT_LLM_TEMPERATURE,
        max_tokens=16,
        request_timeout_seconds=(TRANSPORT_TIMEOUT_SECONDS),
    )

    # Registry 负责根据 primary alias 创建 ChatOpenAI。
    # factory 会把 request_timeout_seconds 翻译为
    # ChatOpenAI(timeout=0.001)。
    registry = LLMRegistry(
        [spec],
        create_openai_chat_model,
    )

    # 只允许一次 attempt。
    # transport timeout 后不重复发送真实请求。
    service = LLMService(
        registry,
        max_attempts=1,
        retry_wait_multiplier=0,
        total_timeout_seconds=TOTAL_TIMEOUT_SECONDS,
    )

    try:
        # 这是一次真实 provider 网络调用。
        # 正常情况下，请求会在建立连接或等待响应时超时。
        await service.call(
            [
                HumanMessage(
                    content="Reply with OK.",
                )
            ],
            aliases=("primary",),
        )
    except AllModelsFailedError as error:
        # APITimeoutError 属于可重试 provider 异常。
        # max_attempts=1，因此 primary 立即耗尽，
        # LLMService 最终抛出 AllModelsFailedError。
        #
        # failures 只保存 alias 和错误类型名称，
        # 不包含 prompt、模型正文或 provider 异常正文。
        provider_error_types = [failure.error_type for failure in error.failures]

        expected_timeout = provider_error_types == [EXPECTED_ERROR_TYPE]

        elapsed_ms = round(
            (perf_counter() - started_at) * 1000,
            2,
        )

        # transport timeout 应明显早于 LLMService 的整体预算。
        # 如果超过整体预算，说明底层网络取消仍可能不及时。
        within_total_budget = elapsed_ms < TOTAL_TIMEOUT_SECONDS * 1000

        ok = expected_timeout and within_total_budget

        print(
            json.dumps(
                {
                    "ok": ok,
                    "expected_timeout": (expected_timeout),
                    "within_total_budget": (within_total_budget),
                    "provider_error_types": (provider_error_types),
                    "elapsed_ms": elapsed_ms,
                }
            )
        )

        return 0 if ok else 1

    except Exception as error:
        # 其他异常表示实验没有沿着预期的 transport
        # timeout 路径结束。
        #
        # 不使用 str(error)，避免泄露 provider 返回内容。
        print(
            json.dumps(
                {
                    "ok": False,
                    "expected_timeout": False,
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

    else:
        # 如果请求在 1 毫秒内成功，说明本次实验没有触发
        # 预期 timeout，因此不能把它判定为通过。
        print(
            json.dumps(
                {
                    "ok": False,
                    "expected_timeout": False,
                    "error_type": "UnexpectedSuccess",
                    "elapsed_ms": round(
                        (perf_counter() - started_at) * 1000,
                        2,
                    ),
                }
            )
        )
        return 1


if __name__ == "__main__":
    # 把协程返回的 0/1 转换成操作系统可观察的退出码。
    raise SystemExit(asyncio.run(run_transport_timeout_smoke()))
