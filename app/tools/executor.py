"""统一工具执行链：授权、预算、门控、超时、截断与生命周期事件."""

from __future__ import annotations

import asyncio

from app.runtime import (
    BudgetPolicy,
    ContextAllocator,
    ContextFragment,
    ContextKind,
    ContextSource,
    Sensitivity,
    TrustLevel,
)
from app.tools.contracts import ToolExecutionContext, ToolExecutionResult
from app.tools.hooks import NoopToolHook, ToolHook, emit_safely, lifecycle_event
from app.tools.policy import ToolPolicy
from app.tools.registry import RuntimeToolRegistry


class ToolExecutor:
    """无请求状态的共享执行器；每次调用显式接收可信上下文."""

    def __init__(
        self,
        registry: RuntimeToolRegistry,
        *,
        policy: ToolPolicy | None = None,
        budget_policy: BudgetPolicy | None = None,
        hook: ToolHook | None = None,
        passthrough_exception_types: tuple[type[BaseException], ...] = (),
    ) -> None:
        """保存共享边界，并创建有界并发门和串行门."""
        self._registry = registry
        self._policy = policy or ToolPolicy()
        self._budget = budget_policy or BudgetPolicy()
        self._hook = hook or NoopToolHook()
        self._parallel_gate = asyncio.Semaphore(self._budget.max_parallel_operations)
        self._serial_gate = asyncio.Lock()
        self._passthrough = passthrough_exception_types

    async def execute(
        self,
        name: str,
        arguments: dict[str, object],
        *,
        context: ToolExecutionContext,
    ) -> ToolExecutionResult:
        """执行完整策略链；普通异常只返回类型，不返回敏感异常正文."""
        descriptor, tool = self._registry.resolve(name)
        self._policy.authorize(descriptor, context)
        await emit_safely(self._hook, lifecycle_event(descriptor, phase="started"))
        gate = self._parallel_gate if descriptor.supports_parallel else self._serial_gate
        timeout_seconds = min(descriptor.timeout_seconds, self._budget.total_timeout_seconds)
        output_limit = min(descriptor.output_token_limit, self._budget.max_tool_output_tokens)
        try:
            async with gate, asyncio.timeout(timeout_seconds):
                raw = await tool.invoke(arguments, context)
        except self._passthrough:
            raise
        except TimeoutError:
            result = ToolExecutionResult(
                tool_name=descriptor.name,
                status="error",
                content=f"Tool {descriptor.name!r} timed out.",
                truncated=False,
                output_tokens=0,
                error_type="TimeoutError",
            )
        except Exception as error:
            result = ToolExecutionResult(
                tool_name=descriptor.name,
                status="error",
                content=f"Tool {descriptor.name!r} failed with {type(error).__name__}.",
                truncated=False,
                output_tokens=0,
                error_type=type(error).__name__,
            )
        else:
            output_text = str(raw) or "(empty tool result)"
            allocated = ContextAllocator(max_tokens=output_limit).allocate(
                (
                    ContextFragment(
                        kind=ContextKind.TOOL_RESULT,
                        source=ContextSource.TOOL_OUTPUT,
                        trust_level=TrustLevel.UNTRUSTED,
                        sensitivity=Sensitivity.INTERNAL,
                        content=output_text,
                    ),
                )
            )
            fragment = allocated.fragments[0]
            result = ToolExecutionResult(
                tool_name=descriptor.name,
                status="success",
                content=fragment.content,
                truncated=fragment.truncated,
                output_tokens=fragment.estimated_tokens,
            )
        await emit_safely(
            self._hook,
            lifecycle_event(
                descriptor,
                phase="completed",
                status=result.status,
                truncated=result.truncated,
                error_type=result.error_type,
            ),
        )
        return result


__all__ = ["ToolExecutor"]
