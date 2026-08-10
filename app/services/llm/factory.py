"""OpenAI-compatible chat model factories."""

from collections.abc import Mapping
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI

from app.schemas.llm import ModelSpec


def create_openai_chat_model(
    spec: ModelSpec,
    overrides: Mapping[str, object],
) -> BaseChatModel:
    """根据模型配置创建 OpenAI-compatible 聊天模型."""
    # 固定配置来自 ModelSpec。
    # 按照字段对应关系填写右侧值。
    params: dict[str, Any] = {
        "model": spec.provider_model,
        "api_key": spec.api_key,
        "temperature": spec.temperature,
        "max_completion_tokens": spec.max_tokens,
        # Lab 06 会统一实现 retry，关闭 ChatOpenAI 内部重试。
        "max_retries": 0,
    }

    # 自定义代理地址是可选配置。
    # base_url 不为 None 时才加入 params。
    if spec.base_url is not None:
        params["base_url"] = spec.base_url

    # 本次调用参数最后合并，因此它的优先级高于 ModelSpec 默认值。
    # spec.temperature == 0.2 overrides == {"temperature": 0.8}
    # overrides 优先级高于 ModelSpec
    params.update(overrides)

    # 创建对象不会发送网络请求；只有 invoke/ainvoke 才会调用 provider。
    return ChatOpenAI(**params)
