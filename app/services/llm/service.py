"""Resilient LLM invocation service."""

import asyncio
from collections.abc import Callable, Mapping, Sequence
from time import perf_counter

from langchain_core.messages import BaseMessage
from openai import APIConnectionError, InternalServerError, RateLimitError
from tenacity import (
    AsyncRetrying,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from app.core.logging import logger
from app.services.llm.errors import (
    AllModelsFailedError,
    LLMTimeoutError,
    ModelFailure,
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

    async def _invoke_once(
        self,
        alias: str,
        messages: Sequence[BaseMessage],
        overrides: Mapping[str, object] | None = None,
    ) -> BaseMessage:
        """使用指定 alias 完成一次异步模型调用."""
        # 通过 Registry 创建 model。
        model = self._registry.get(alias=alias, overrides=overrides)

        # 异步调用模型，并直接返回 BaseMessage。
        return await model.ainvoke(messages)

    async def _invoke_with_retry(
        self,
        alias: str,
        messages: Sequence[BaseMessage],
        overrides: Mapping[str, object] | None = None,
    ) -> BaseMessage:
        """对指定模型执行有限次数的异步重试."""
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
                    return await self._invoke_once(
                        alias,
                        messages,
                        overrides,
                    )
                except Exception as error:
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
        messages: Sequence[BaseMessage],
        *,  # 后续参数必须通过关键字传递，避免 messages 与 aliases 位置混淆。
        aliases: Sequence[str],
        overrides: Mapping[str, object] | None = None,
    ) -> BaseMessage:
        """按照 alias 顺序执行 retry 和 fallback."""
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
                    alias,
                    messages,
                    overrides,
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

    async def call(
        self,
        messages: Sequence[BaseMessage],
        *,
        aliases: Sequence[str],
        overrides: Mapping[str, object] | None = None,
    ) -> BaseMessage:
        """在总时间预算内执行完整 retry 和 fallback 链."""
        # 保存 timeout 对象，异常发生后可以检查究竟是不是总预算到期。
        timeout_context = asyncio.timeout(
            self._total_timeout_seconds,
        )

        try:
            async with timeout_context:
                return await self._call_with_fallback(
                    messages,
                    aliases=aliases,
                    overrides=overrides,
                )
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
            raise LLMTimeoutError(
                self._total_timeout_seconds,
            ) from None
