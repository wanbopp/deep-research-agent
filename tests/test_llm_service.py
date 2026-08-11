"""LLM retry, fallback, timeout, and concurrency contracts."""

from collections.abc import Mapping, Sequence
import asyncio
import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import HumanMessage
from pydantic import SecretStr

from app.schemas.llm import ModelSpec
from app.services.llm.errors import AllModelsFailedError, LLMTimeoutError
from app.services.llm.registry import LLMRegistry
from app.services.llm.service import LLMService

type PlannedOutcome = str | Exception


class _TransientError(Exception):
    """模拟值得重试的临时 provider 异常."""


class _SequenceFactory:
    """按照 alias 的预设序列返回模型或抛出异常."""

    def __init__(
        self,
        plans: Mapping[str, Sequence[PlannedOutcome]],
    ) -> None:
        """复制预设序列，并初始化调用记录."""
        self._plans = {alias: list(outcomes) for alias, outcomes in plans.items()}
        self.calls: list[str] = []

    def __call__(
        self,
        spec: ModelSpec,
        overrides: Mapping[str, object],
    ) -> BaseChatModel:
        """消费当前 alias 的下一个预设结果."""
        self.calls.append(spec.alias)
        outcome = self._plans[spec.alias].pop(0)

        if isinstance(outcome, Exception):
            raise outcome

        return FakeListChatModel(responses=[outcome])


def _model_spec(alias: str) -> ModelSpec:
    """创建不包含真实凭据的测试模型配置."""
    return ModelSpec(
        alias=alias,
        provider_model="fake-model",
        api_key=SecretStr("unit-test-secret"),
    )


def _is_transient(error: BaseException) -> bool:
    """只有测试用临时异常允许重试和 fallback."""
    return isinstance(error, _TransientError)


@pytest.mark.anyio
async def test_retry_succeeds_and_non_retryable_error_stops_immediately() -> None:
    """临时错误应重试，非临时错误应立即传播."""
    messages = [HumanMessage(content="test prompt")]

    retry_factory = _SequenceFactory(
        {
            "primary": [
                _TransientError("temporary failure"),
                "retry-ok",
            ]
        }
    )
    retry_service = LLMService(
        LLMRegistry(
            [_model_spec("primary")],
            retry_factory,
        ),
        max_attempts=2,
        retry_wait_multiplier=0,
        retry_predicate=_is_transient,
    )

    retry_result = await retry_service.call(
        messages,
        aliases=("primary",),
    )

    assert retry_result.content == "retry-ok"
    assert retry_factory.calls == ["primary", "primary"]

    non_retryable_factory = _SequenceFactory(
        {
            "primary": [
                ValueError("invalid input"),
                "must-not-be-used",
            ]
        }
    )
    non_retryable_service = LLMService(
        LLMRegistry(
            [_model_spec("primary")],
            non_retryable_factory,
        ),
        max_attempts=3,
        retry_wait_multiplier=0,
        retry_predicate=_is_transient,
    )

    with pytest.raises(ValueError, match="invalid input"):
        await non_retryable_service.call(
            messages,
            aliases=("primary",),
        )

    assert non_retryable_factory.calls == ["primary"]


@pytest.mark.anyio
async def test_fallback_succeeds_and_all_failures_are_safely_summarized() -> None:
    """首模型耗尽后应 fallback，全失败时只返回安全摘要."""
    prompt_text = "PRIVATE_TEST_PROMPT"
    messages = [HumanMessage(content=prompt_text)]

    fallback_factory = _SequenceFactory(
        {
            "primary": [
                _TransientError("primary temporary failure"),
            ],
            "fast": [
                "fallback-ok",
            ],
        }
    )
    fallback_service = LLMService(
        LLMRegistry(
            [
                _model_spec("primary"),
                _model_spec("fast"),
            ],
            fallback_factory,
        ),
        max_attempts=1,
        retry_wait_multiplier=0,
        retry_predicate=_is_transient,
    )

    fallback_result = await fallback_service.call(
        messages,
        aliases=("primary", "fast"),
    )

    assert fallback_result.content == "fallback-ok"
    assert fallback_factory.calls == ["primary", "fast"]

    sensitive_error_text = "SENSITIVE_PROVIDER_DETAIL"
    all_failed_factory = _SequenceFactory(
        {
            "primary": [
                _TransientError(sensitive_error_text),
            ],
            "fast": [
                _TransientError(sensitive_error_text),
            ],
        }
    )
    all_failed_service = LLMService(
        LLMRegistry(
            [
                _model_spec("primary"),
                _model_spec("fast"),
            ],
            all_failed_factory,
        ),
        max_attempts=1,
        retry_wait_multiplier=0,
        retry_predicate=_is_transient,
    )

    with pytest.raises(AllModelsFailedError) as exc_info:
        await all_failed_service.call(
            messages,
            aliases=("primary", "fast"),
        )

    assert all_failed_factory.calls == ["primary", "fast"]

    failures = exc_info.value.failures
    assert [(failure.alias, failure.error_type) for failure in failures] == [
        ("primary", "_TransientError"),
        ("fast", "_TransientError"),
    ]

    error_summary = str(exc_info.value)
    assert "primary" in error_summary
    assert "fast" in error_summary
    assert "_TransientError" in error_summary
    assert sensitive_error_text not in error_summary
    assert prompt_text not in error_summary


@pytest.mark.anyio
async def test_total_timeout_and_concurrent_calls_keep_state_isolated() -> None:
    """总预算应终止慢调用，并发请求的 fallback 状态应相互隔离."""

    def slow_factory(
        spec: ModelSpec,
        overrides: Mapping[str, object],
    ) -> BaseChatModel:
        """返回一个超过测试总预算的异步 fake."""
        return FakeListChatModel(
            responses=["too-late"],
            sleep=0.05,
        )

    timeout_service = LLMService(
        LLMRegistry(
            [_model_spec("slow")],
            slow_factory,
        ),
        max_attempts=1,
        retry_wait_multiplier=0,
        total_timeout_seconds=0.01,
    )

    with pytest.raises(LLMTimeoutError) as exc_info:
        await timeout_service.call(
            [HumanMessage(content="timeout prompt")],
            aliases=("slow",),
        )

    assert exc_info.value.timeout_seconds == 0.01

    concurrent_calls: list[tuple[str, str]] = []

    def concurrent_factory(
        spec: ModelSpec,
        overrides: Mapping[str, object],
    ) -> BaseChatModel:
        """根据调用级 request_name 产生独立结果."""
        request_name = overrides.get("request_name")
        assert isinstance(request_name, str)

        concurrent_calls.append((request_name, spec.alias))

        if spec.alias == "primary":
            raise _TransientError("temporary failure")

        return FakeListChatModel(
            responses=[f"{request_name}-ok"],
            # 产生真实的协程交错机会，但不会访问网络。
            sleep=0.001,
        )

    concurrent_service = LLMService(
        LLMRegistry(
            [
                _model_spec("primary"),
                _model_spec("fast"),
            ],
            concurrent_factory,
        ),
        max_attempts=1,
        retry_wait_multiplier=0,
        retry_predicate=_is_transient,
        total_timeout_seconds=1,
    )

    first_result, second_result = await asyncio.gather(
        concurrent_service.call(
            [HumanMessage(content="request a")],
            aliases=("primary", "fast"),
            overrides={"request_name": "a"},
        ),
        concurrent_service.call(
            [HumanMessage(content="request b")],
            aliases=("primary", "fast"),
            overrides={"request_name": "b"},
        ),
    )

    assert first_result.content == "a-ok"
    assert second_result.content == "b-ok"

    assert concurrent_calls.count(("a", "primary")) == 1
    assert concurrent_calls.count(("a", "fast")) == 1
    assert concurrent_calls.count(("b", "primary")) == 1
    assert concurrent_calls.count(("b", "fast")) == 1
