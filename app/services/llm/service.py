"""Resilient LLM invocation service."""

import asyncio
from collections.abc import Awaitable, Callable, Mapping, Sequence
from time import perf_counter
from typing import TypeVar

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.exceptions import OutputParserException
from langchain_core.tools import BaseTool
from pydantic import BaseModel, ValidationError
from openai import APIConnectionError, InternalServerError, RateLimitError
from tenacity import (
    AsyncRetrying,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from app.core.logging import logger
from app.observability import MetricsRuntime, metrics
from app.services.llm.errors import (
    AllModelsFailedError,
    LLMTimeoutError,
    ModelFailure,
    StructuredOutputError,
)
from app.services.llm.registry import LLMRegistry

# 接收异常并回答 "是否允许重试"
RetryPredicate = Callable[[BaseException], bool]

# 只列出明确具有临时性的 provider 异常
# 是一个 tuple，里面可以放任意多个 Exception 的异常类
_RETRYABLE_PROVIDER_ERRORS: tuple[type[Exception], ...] = (
    APIConnectionError,
    RateLimitError,
    InternalServerError,
)

# 表示最终必须是某种 Pydantic BaseModel 子类,只能返回 Pydantic BaseModel。
StructuredT = TypeVar("StructuredT", bound=BaseModel)

# 表示内部执行链可以返回任意类型
OutputT = TypeVar("OutputT")

# 接收当前 attempt 创建的模型，并异步返回调用结果。
# OutputT 让整条 retry/fallback/timeout 链保留 invoker 的返回类型。
# 一个函数，接收一个 BaseChatModel，调用后返回一个可以 await 的对象；await 完以后，得到 OutputT
ModelInvoker = Callable[[BaseChatModel], Awaitable[OutputT]]


def _is_retryable_provider_error(error: BaseException) -> bool:
    """判断 provider 异常是否值得重试."""
    # isinstance 的第二个参数可以是类型元组；匹配其中任意类型就返回 True。
    return isinstance(error, _RETRYABLE_PROVIDER_ERRORS)


class LLMService:
    """通过 Registry 调用聊天模型，并逐步增加弹性策略."""

    def __init__(
        self,
        registry: LLMRegistry,
        *,
        max_attempts: int = 3,
        retry_wait_multiplier: float = 0.25,
        total_timeout_seconds: float = 60.0,
        retry_predicate: RetryPredicate = _is_retryable_provider_error,
        metrics_runtime: MetricsRuntime = metrics,
    ) -> None:
        """保存模型 Registry、单模型重试配置和整次调用总预算."""
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        if retry_wait_multiplier < 0:
            raise ValueError("retry_wait_multiplier cannot be negative")
        if total_timeout_seconds <= 0:
            raise ValueError("total_timeout_seconds must be greater than 0")

        self._registry = registry
        self._max_attempts = max_attempts
        self._retry_wait_multiplier = retry_wait_multiplier
        self._total_timeout_seconds = total_timeout_seconds
        self._retry_predicate = retry_predicate
        self._metrics = metrics_runtime

    async def _invoke_once(
        self,
        alias: str,
        invoker: ModelInvoker[OutputT],
        overrides: Mapping[str, object] | None = None,
    ) -> OutputT:
        """使用指定 alias 完成一次异步模型调用.

        - _invoke_once() 不再关心输入是文本消息还是结构化请求。
        - 返回值从 BaseMessage 改为 OutputT；invoker 返回什么类型，本方法就返回什么类型。
        - Registry 职责不变；它仍然只负责创建当前 alias 对应的模型。
        """
        # 通过 Registry 创建 model。
        model = self._registry.get(alias=alias, overrides=overrides)

        # 具体调用方式由公开入口注入：
        # text 路径会执行 model.ainvoke(messages),
        # structured 路径以后会执行 wrapped_model.ainvoke(messages)
        return await invoker(model)

    async def _invoke_with_retry(
        self,
        alias: str,
        invoker: ModelInvoker[OutputT],
        overrides: Mapping[str, object] | None = None,
    ) -> OutputT:
        """对指定模型执行有限次数的异步重试，并原样返回 invoker 的结果类型."""
        retrying = AsyncRetrying(
            # max_attempts 包含第一次调用。
            stop=stop_after_attempt(self._max_attempts),
            # 第一次失败后的等待约为 multiplier，随后指数增长。
            # 最大等待限制为 2 秒，避免单模型占用过多总预算。
            wait=wait_exponential(
                multiplier=self._retry_wait_multiplier,
                max=2.0,
            ),
            # 只有 predicate 返回 True 的异常才允许重试。
            retry=retry_if_exception(self._retry_predicate),
            # attempts 耗尽后重新抛出最后一个原始异常，
            # 而不是向上层暴露 tenacity.RetryError。
            reraise=True,
        )

        async for attempt in retrying:
            # with attempt 会把代码块中的异常交给 tenacity 判断。
            # 没有这一层，tenacity 无法捕获和调度下一次 attempt。
            with attempt:
                attempt_started_at = perf_counter()
                try:
                    result = await self._invoke_once(
                        alias=alias,
                        invoker=invoker,
                        overrides=overrides,
                    )
                    self._metrics.observe_llm_attempt(model_alias=alias, outcome="success")
                    self._record_usage(alias=alias, result=result)
                    return result
                except Exception as error:
                    self._metrics.observe_llm_attempt(model_alias=alias, outcome="error")
                    logger.warning(
                        "llm_attempt_failed",
                        alias=alias,
                        attempt=attempt.retry_state.attempt_number,
                        elapsed_ms=round(
                            (perf_counter() - attempt_started_at) * 1000,
                            2,
                        ),
                        error_type=type(error).__name__,
                    )
                    raise

        # AsyncRetrying 正常情况下只会返回结果或抛出异常。
        # 保留这一行是为了让类型检查器确认所有路径都有返回值。
        raise RuntimeError("Retry loop exited without a result")

    async def _call_with_fallback(
        self,
        invoker: ModelInvoker[OutputT],
        *,  # 策略参数必须通过关键字传递，避免 aliases 与 overrides 混淆。
        aliases: Sequence[str],
        overrides: Mapping[str, object] | None = None,
    ) -> OutputT:
        """按照 alias 顺序执行 retry 和 fallback，并返回 invoker 的结果."""
        # str 本身也是 Sequence[str]，但传入 "primary" 会被逐字符遍历，
        # 因此需要显式拒绝字符串和空序列。
        if isinstance(aliases, str) or not aliases:
            raise ValueError("aliases must contain at least one model alias")

        # 每次 call() 都创建自己的失败记录，不能保存在 self 上。
        # 这样并发请求之间不会共享当前模型或失败状态。
        failures: list[ModelFailure] = []

        for alias in aliases:
            try:
                return await self._invoke_with_retry(
                    alias=alias,
                    invoker=invoker,
                    overrides=overrides,
                )
            except Exception as error:
                # non-retryable error 不应该 fallback。
                # 使用裸 raise 保留原异常与 traceback。
                if not self._retry_predicate(error):
                    raise

                error_type = type(error).__name__

                logger.warning(
                    "llm_model_exhausted",
                    alias=alias,
                    error_type=error_type,
                )
                # 不保存 error 对象，也不调用 str(error)。
                failures.append(
                    ModelFailure(
                        alias=alias,
                        error_type=error_type,
                    )
                )

        # 只有所有 alias 都因 retryable error 耗尽时才会到这里。
        raise AllModelsFailedError(tuple(failures))

    async def _execute(
        self,
        invoker: ModelInvoker[OutputT],
        *,
        aliases: Sequence[str],
        overrides: Mapping[str, object] | None = None,
        operation: str,
    ) -> OutputT:
        """在总时间预算内执行完整 retry 和 fallback 链."""
        # 保存 timeout 对象，异常发生后可以检查究竟是不是总预算到期。
        timeout_context = asyncio.timeout(
            self._total_timeout_seconds,
        )

        started_at = perf_counter()
        outcome = "error"
        try:
            async with timeout_context:
                result = await self._call_with_fallback(
                    invoker=invoker,
                    aliases=aliases,
                    overrides=overrides,
                )
                outcome = "success"
                return result
        except TimeoutError:
            # 内部代码也可能主动抛 TimeoutError。
            # context 没有过期时，应保留原始异常。
            if not timeout_context.expired():
                raise

            logger.error(
                "llm_call_timed_out",
                timeout_seconds=self._total_timeout_seconds,
            )

            # 只有整个调用预算真正到期，才转换为领域错误。
            outcome = "timeout"
            raise LLMTimeoutError(
                self._total_timeout_seconds,
            ) from None
        finally:
            self._metrics.observe_llm_call(
                operation=operation,
                outcome=outcome,
                duration_seconds=perf_counter() - started_at,
            )

    async def call(
        self,
        messages: Sequence[BaseMessage],
        *,
        aliases: Sequence[str],
        tools: Sequence[BaseTool] | None = None,
        overrides: Mapping[str, object] | None = None,
    ) -> BaseMessage:
        """使用共享弹性链路执行普通文本模型调用."""
        # 固定本次调用的工具快照，不能保存到 self。
        call_tools = tuple(tools or ())

        async def invoke_text(model: BaseChatModel) -> BaseMessage:
            """使用当前 alias 对应的模型处理本次消息.

            messages 只存在于公开的 call() 闭包中；
            内部弹性链路不再关心输入和输出的具体类型。
            """
            # bind_tools 返回新的 runnable，需要保留它。
            runnable = model.bind_tools(call_tools) if call_tools else model

            # 必须调用绑定后的 runnable，而不是原始 model。
            return await runnable.ainvoke(messages)

        return await self._execute(
            invoke_text,
            aliases=aliases,
            overrides=overrides,
            operation="text",
        )

    async def call_structured(
        self,
        messages: Sequence[BaseMessage],
        *,
        response_model: type[StructuredT],
        aliases: Sequence[str],
        overrides: Mapping[str, object] | None = None,
    ) -> StructuredT:
        """使用共享弹性链路调用模型，并返回校验后的 Pydantic 对象."""

        async def invoke_structured(
            model: BaseChatModel,
        ) -> StructuredT:
            """为当前模型绑定响应 schema，并校验最终结果."""
            # 调用 model.with_structured_output(response_model)
            # 得到只服务于当前 attempt 的 structured runnable。
            structured_model = model.with_structured_output(
                response_model,
                method="function_calling",  # 利用 tool-calling 协议，让模型按照固定参数结构返回数据
            )
            try:
                # 异步调用 structured runnable，传入 messages。
                result = await structured_model.ainvoke(messages)
                # 使用 response_model.model_validate(result)
                # 对 LangChain 的结果再守一次最终类型。
                return response_model.model_validate(result)

            except (OutputParserException, ValidationError) as error:
                # 转换成 StructuredOutputError。
                # 使用 from None 隐藏可能包含原始输出的异常链。
                raise StructuredOutputError(
                    schema_name=response_model.__name__,
                    error_type=type(error).__name__,
                ) from None

        # 调用 self._execute()。
        # 传入 invoke_structured、aliases 和 overrides。
        return await self._execute(
            invoker=invoke_structured,
            aliases=aliases,
            overrides=overrides,
            operation="structured",
        )

    def _record_usage(self, *, alias: str, result: object) -> None:
        """从 LangChain 标准 usage_metadata 读取明确 token 数，不检查响应正文."""
        usage = getattr(result, "usage_metadata", None)
        if not isinstance(usage, Mapping):
            return
        input_tokens = usage.get("input_tokens")
        output_tokens = usage.get("output_tokens")
        self._metrics.observe_llm_tokens(
            model_alias=alias,
            input_tokens=input_tokens if isinstance(input_tokens, int) else None,
            output_tokens=output_tokens if isinstance(output_tokens, int) else None,
        )
