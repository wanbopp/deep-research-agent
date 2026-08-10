"""Registry for resolving stable model aliases."""

from collections.abc import Mapping, Sequence
from typing import Protocol

from langchain_core.language_models.chat_models import BaseChatModel

from app.schemas.llm import ModelSpec
from app.services.llm.errors import DuplicateModelAliasError, UnknownModelError


class ModelFactory(Protocol):
    """定义 Registry 可以调用的模型工厂形状."""

    def __call__(
        self,
        spec: ModelSpec,
        overrides: Mapping[str, object],
    ) -> BaseChatModel:
        """根据模型配置和本次调用参数创建聊天模型,不同的厂商模型提供不同的实现."""
        ...


class LLMRegistry:
    """保存模型配置，并通过稳定 alias 解析配置."""

    def __init__(
        self,
        specs: Sequence[ModelSpec],  # Sequence 序列
        factory: ModelFactory,
    ) -> None:
        """根据模型配置序列创建 Registry."""
        self._specs: dict[str, ModelSpec] = {}

        # Registry 只保存 factory，不关心它最终创建的是 OpenAI 还是什么模型。
        self._factory = factory

        for spec in specs:
            # 如果 alias 已存在，抛出 DuplicateModelAliasError。
            if spec.alias in self._specs:
                raise DuplicateModelAliasError(spec.alias)

            # 以 alias 为 key 保存整个 ModelSpec。
            self._specs[spec.alias] = spec

    def names(self) -> tuple[str, ...]:
        """按照注册顺序返回所有 alias."""
        return tuple(self._specs)

    def resolve(self, alias: str) -> ModelSpec:
        """根据 alias 返回对应配置."""
        try:
            # 从 self._specs 中读取配置。
            return self._specs[alias]
        except KeyError:
            raise UnknownModelError(alias, self.names()) from None

    def get(
        self,
        alias: str,
        overrides: Mapping[str, object] | None = None,
    ) -> BaseChatModel:
        """根据 alias 和本次调用参数创建聊天模型."""
        # 第一步：复用 resolve() 查找配置。
        # 未知 alias 会统一抛出 UnknownModelError。
        spec = self.resolve(alias=alias)

        # 第二步：为本次调用创建新的字典。
        # 即使调用者传入的是原始 dict，也不会把同一个 dict 直接交给 factory。
        call_overrides = dict(overrides or {})

        # 第三步：把固定配置和本次临时参数交给注入的 factory。
        return self._factory(spec, call_overrides)
