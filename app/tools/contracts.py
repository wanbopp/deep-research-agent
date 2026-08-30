"""工具描述、风险和执行结果契约."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from uuid import UUID


class ToolExposure(StrEnum):
    """工具可被哪一层调用."""

    MODEL = "model"
    INTERNAL = "internal"


class ToolRisk(StrEnum):
    """有限风险级别，策略不得依赖自由格式描述."""

    READ_ONLY = "read_only"
    INTERACTIVE = "interactive"
    WRITE = "write"
    DESTRUCTIVE = "destructive"


@dataclass(frozen=True, slots=True)
class ToolDescriptor:
    """注册时冻结的工具能力与执行上限."""

    name: str
    namespace: str
    exposure: ToolExposure
    risk: ToolRisk
    timeout_seconds: float
    output_token_limit: int
    supports_parallel: bool
    requires_approval: bool

    def __post_init__(self) -> None:
        """拒绝不可路由名称和无界预算."""
        if not self.name or not self.namespace:
            raise ValueError("tool name and namespace must not be empty")
        if self.timeout_seconds <= 0 or self.output_token_limit <= 0:
            raise ValueError("tool budgets must be greater than zero")
        if self.risk in {ToolRisk.WRITE, ToolRisk.DESTRUCTIVE} and not self.requires_approval:
            raise ValueError("write and destructive tools must require approval")

    @property
    def qualified_name(self) -> str:
        """返回 namespace:name 稳定身份."""
        return f"{self.namespace}:{self.name}"


@dataclass(frozen=True, slots=True)
class ToolExecutionContext:
    """服务端注入的可信执行上下文，模型不能提供 user_id 或批准集合."""

    user_id: UUID
    approved_tool_names: frozenset[str] = frozenset()


@dataclass(frozen=True, slots=True)
class ToolExecutionResult:
    """返回模型的有界结果或稳定错误类型."""

    tool_name: str
    status: str
    content: str
    truncated: bool
    output_tokens: int
    error_type: str | None = None


__all__ = [
    "ToolDescriptor",
    "ToolExecutionContext",
    "ToolExecutionResult",
    "ToolExposure",
    "ToolRisk",
]
