"""LLM 模型配置与 Registry 契约测试."""

from collections.abc import Mapping

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_openai import ChatOpenAI
from pydantic import SecretStr, ValidationError

from app.schemas.llm import ModelSpec
from app.services.llm.errors import DuplicateModelAliasError, UnknownModelError
from app.services.llm.factory import create_openai_chat_model
from app.services.llm.registry import LLMRegistry


class _RecordingFactory:
    """记录 Registry 传入的配置和本次调用参数."""

    def __init__(self) -> None:
        self.calls: list[tuple[ModelSpec, Mapping[str, object]]] = []

    def __call__(
        self,
        spec: ModelSpec,
        overrides: Mapping[str, object],
    ) -> BaseChatModel:
        """记录调用并返回不会访问网络的 fake model."""
        self.calls.append((spec, overrides))
        return FakeListChatModel(responses=["ok"])


def test_model_spec_separates_alias_from_provider_and_hides_secret() -> None:
    """模型配置应区分稳定 alias、provider 名称，并隐藏密钥."""
    spec = ModelSpec(
        alias="primary",
        provider_model="gpt-4o-mini",
        api_key=SecretStr("unit-test-secret"),
    )

    assert spec.alias == "primary"
    assert spec.provider_model == "gpt-4o-mini"
    assert spec.temperature == 0.2
    assert spec.max_tokens == 2000
    assert spec.capabilities == frozenset({"text"})

    assert spec.api_key.get_secret_value() == "unit-test-secret"
    assert "unit-test-secret" not in repr(spec)


def test_model_spec_rejects_invalid_default_parameters() -> None:
    """模型默认参数越界时应在配置阶段失败."""
    base_payload: dict[str, object] = {
        "alias": "primary",
        "provider_model": "gpt-4o-mini",
        "api_key": SecretStr("unit-test-secret"),
    }
    invalid_overrides: tuple[dict[str, object], ...] = (
        {"temperature": -0.01},
        {"temperature": 2.01},
        {"max_tokens": 0},
    )

    for overrides in invalid_overrides:
        with pytest.raises(ValidationError):
            ModelSpec.model_validate({**base_payload, **overrides})


def test_model_spec_is_frozen_and_rejects_unknown_fields() -> None:
    """模型配置不可修改，也不能静默忽略拼错的字段."""
    spec = ModelSpec(
        alias="primary",
        provider_model="gpt-4o-mini",
        api_key=SecretStr("unit-test-secret"),
    )

    with pytest.raises(ValidationError, match="frozen"):
        spec.temperature = 0.5

    payload = spec.model_dump()
    payload["unexpected_option"] = True

    with pytest.raises(ValidationError, match="extra_forbidden"):
        ModelSpec.model_validate(payload)


def test_registry_resolves_alias_and_rejects_invalid_registry_state() -> None:
    """Registry 应解析已知 alias，并拒绝未知或重复 alias."""
    primary = ModelSpec(
        alias="primary",
        provider_model="gpt-4o-mini",
        api_key=SecretStr("unit-test-secret"),
    )
    fast = ModelSpec(
        alias="fast",
        provider_model="gpt-4o-mini",
        api_key=SecretStr("unit-test-secret"),
    )
    factory = _RecordingFactory()
    registry = LLMRegistry([primary, fast], factory)

    # 正常查找应保持注册顺序，并返回原来的 frozen 配置对象。
    assert registry.names() == ("primary", "fast")
    assert registry.resolve("primary") is primary

    # 未知 alias 应转换为领域异常，并提供可用名称。
    with pytest.raises(UnknownModelError) as exc_info:
        registry.resolve("missing")

    assert exc_info.value.alias == "missing"
    assert exc_info.value.available_aliases == ("primary", "fast")
    assert "unit-test-secret" not in str(exc_info.value)

    # 重复 alias 会造成查找结果不确定，因此构造阶段立即失败。
    with pytest.raises(DuplicateModelAliasError):
        LLMRegistry([primary, primary], factory)


def test_registry_and_openai_factory_keep_call_overrides_isolated() -> None:
    """Registry 调用参数应隔离，OpenAI factory 应正确翻译配置."""
    secret = "unit-test-secret"
    spec = ModelSpec(
        alias="primary",
        provider_model="gpt-4o-mini",
        api_key=SecretStr(secret),
        base_url="https://example.invalid/v1",
        temperature=0.2,
        max_tokens=2000,
    )
    recording_factory = _RecordingFactory()
    registry = LLMRegistry([spec], recording_factory)

    first_overrides: dict[str, object] = {"temperature": 0.1}
    second_overrides: dict[str, object] = {"temperature": 0.9}

    first_model = registry.get("primary", first_overrides)
    second_model = registry.get("primary", second_overrides)

    assert isinstance(first_model, FakeListChatModel)
    assert isinstance(second_model, FakeListChatModel)

    # factory 应分别收到两次调用的参数。
    assert recording_factory.calls == [
        (spec, {"temperature": 0.1}),
        (spec, {"temperature": 0.9}),
    ]

    # Registry 传给 factory 的应是新字典，而不是调用者原来的字典。
    first_call_overrides = recording_factory.calls[0][1]
    second_call_overrides = recording_factory.calls[1][1]

    assert first_call_overrides is not first_overrides
    assert second_call_overrides is not second_overrides
    assert first_call_overrides is not second_call_overrides

    # 本次调用参数不能修改共享的 frozen ModelSpec。
    assert spec.temperature == 0.2

    # 真实 factory 只构造对象，不调用 invoke，因此不会访问网络。
    client = create_openai_chat_model(
        spec,
        {
            "temperature": 0.7,
            "max_completion_tokens": 123,
        },
    )

    # isinstance 同时帮助 Pyright 把 BaseChatModel 缩窄成 ChatOpenAI。
    assert isinstance(client, ChatOpenAI)
    assert client.model_name == "gpt-4o-mini"
    assert client.temperature == 0.7
    assert client.max_tokens == 123
    assert client.max_retries == 0
    assert client.openai_api_base == "https://example.invalid/v1"
    assert secret not in repr(client)
