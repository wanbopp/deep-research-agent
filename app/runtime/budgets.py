"""只能收紧、不能被请求放大的统一预算策略."""

from __future__ import annotations

from dataclasses import dataclass, fields


@dataclass(frozen=True, slots=True)
class BudgetPolicy:
    """一次运行在时间、输入、工具和并发维度上的硬上限."""

    total_timeout_seconds: float = 300.0
    max_input_tokens: int = 12000
    max_tool_output_tokens: int = 2000
    max_evidence: int = 30
    max_retrieval_candidates: int = 80
    max_parallel_operations: int = 12

    def __post_init__(self) -> None:
        """所有预算必须为正，0 不能隐式表示无限制."""
        if any(getattr(self, field.name) <= 0 for field in fields(self)):
            raise ValueError("budget values must be greater than zero")

    def tighten(self, *requested: BudgetPolicy) -> BudgetPolicy:
        """逐字段取最小值，确保租户/请求/剩余预算只能收紧硬上限."""
        policies = (self, *requested)
        return BudgetPolicy(
            **{field.name: min(getattr(policy, field.name) for policy in policies) for field in fields(self)}
        )


__all__ = ["BudgetPolicy"]
