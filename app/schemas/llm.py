"""LLM model configuration schemas."""

from pydantic import BaseModel, ConfigDict, Field, SecretStr


class ModelSpec(BaseModel):
    """描述一个可通过稳定 alias 查找的模型配置."""

    # frozen 禁止修改模型字段，为后续 overrides 隔离提供基础
    # str_strip_whitespace 自动去除字符串首尾的空白
    model_config = ConfigDict(
        frozen=True,
        str_strip_whitespace=True,
        # 配置字段拼错时立即失败，不能静默丢弃。
        extra="forbid",
    )

    # alias 是业务代码使用的稳定名称，例如 primary
    alias: str = Field(min_length=1)

    # provider_model 是实际发送给 OpenAI-compatible provider 的模型名。
    # 它可以变化，但业务代码使用的 alias 应保持稳定。
    provider_model: str = Field(min_length=1)

    # SecretStr 会在 repr 和普通打印中隐藏真实值。
    # 当前测试直接传入 SecretStr，因此字段类型也必须是 SecretStr。
    api_key: SecretStr

    # OpenAI-compatible 服务可以使用自定义代理地址。
    # None 表示使用 client 的默认地址。
    base_url: str | None = None

    # OpenAI-compatible 模型常用范围为 0 到 2，包含边界值。
    temperature: float = Field(
        default=0.2,
        ge=0,
        le=2,
    )

    # 输出 token 上限必须是正整数。
    max_tokens: int = Field(
        default=2000,
        gt=0,
    )

    # 使用不可变集合，避免 frozen model 内部仍包含可变 set。
    capabilities: frozenset[str] = Field(
        default_factory=lambda: frozenset({"text"}),
    )

    # 限制单次 provider 网络请求的等待时间。
    # 它必须短于 LLMService 的整体预算，
    # 为异常记录、重试判断和 fallback 留出时间。
    request_timeout_seconds: float = Field(
        default=45.0,
        gt=0,
    )
