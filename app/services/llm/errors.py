"""Domain errors raised by the LLM service layer."""

from dataclasses import dataclass


class UnknownModelError(LookupError):
    """请求模型 alias 没有注册时抛出."""

    def __init__(
        self,
        alias: str,
        available_aliases: tuple[str, ...],
    ) -> None:
        """保存请求的 alias，并生成不包含敏感配置的错误信息."""
        # 保存成属性，方便调用者通过 exc.alias 获取出错的 alias。
        self.alias = alias
        self.available_aliases = available_aliases

        # join 会把 ("primary", "fast") 转换为 "primary, fast"。
        # 如果元组为空，join 的结果是空字符串，此时使用 "<none>"。
        available_text = ", ".join(available_aliases) or "<none>"

        # LookupError 本身已经负责保存异常消息。
        # {alias!r} 会给字符串加上引号，例如 'missing'。
        super().__init__(f"Unknown model alias {alias!r}. Available aliases: {available_text}")


class DuplicateModelAliasError(ValueError):
    """模型配置中出现重复 alias 时抛出."""

    def __init__(self, alias: str) -> None:
        """保存重复的 alias，并生成对应错误信息."""
        self.alias = alias

        # 这里只显示重复的 alias，不输出完整 ModelSpec。
        super().__init__(f"Duplicate model alias: {alias!r}")


# 装饰器生成初始化方法
@dataclass(frozen=True, slots=True)
class ModelFailure:
    """描述一个模型 alias 的最终失败结果."""

    alias: str
    error_type: str


class AllModelsFailedError(RuntimeError):
    """所有候选模型都失败时抛出."""

    def __init__(
        self,
        failures: tuple[ModelFailure, ...],
    ) -> None:
        """保存安全失败摘要，不保存 provider 原始异常."""
        self.failures = failures

        # 只拼接 alias 和异常类型，禁止使用 str(error)。
        summary = ", ".join(f"{failure.alias}: {failure.error_type}" for failure in failures) or "<none>"

        super().__init__(f"All configured models failed: {summary}")


class LLMTimeoutError(TimeoutError):
    """整条 LLM 调用链超过总时间预算时抛出."""

    def __init__(self, timeout_seconds: float) -> None:
        """保存总时间预算，不保存内部异常或请求内容."""
        self.timeout_seconds = timeout_seconds

        super().__init__(f"LLM call exceeded total timeout of {timeout_seconds:g} seconds")
