"""统一 Tool Runtime 的策略、预算和失败隔离测试."""

import asyncio
from uuid import uuid4

import pytest

from app.runtime import BudgetPolicy
from app.tools import (
    RuntimeToolRegistry,
    ToolApprovalRequired,
    ToolDescriptor,
    ToolExecutionContext,
    ToolExecutor,
    ToolExposure,
    ToolRisk,
)
from app.tools.hooks import ToolLifecycleEvent


class FunctionTool:
    """测试用异步工具."""

    def __init__(self, result: object, *, delay: float = 0.0) -> None:
        """保存结果和可选延迟."""
        self._result = result
        self._delay = delay

    async def invoke(self, arguments: dict[str, object], context: ToolExecutionContext) -> object:
        """返回固定结果，同时证明身份不来自 arguments."""
        del arguments
        if self._delay:
            await asyncio.sleep(self._delay)
        assert context.user_id
        return self._result


class FailingHook:
    """模拟审计后端故障."""

    async def emit(self, event: ToolLifecycleEvent) -> None:
        """始终失败，工具结果仍应成功."""
        del event
        raise RuntimeError("hook offline")


def descriptor(
    *,
    name: str = "demo",
    risk: ToolRisk = ToolRisk.READ_ONLY,
    requires_approval: bool = False,
    timeout_seconds: float = 1.0,
    output_token_limit: int = 8,
) -> ToolDescriptor:
    """构造模型可见的本地 descriptor."""
    return ToolDescriptor(
        name=name,
        namespace="local",
        exposure=ToolExposure.MODEL,
        risk=risk,
        timeout_seconds=timeout_seconds,
        output_token_limit=output_token_limit,
        supports_parallel=True,
        requires_approval=requires_approval,
    )


def executor_for(tool: FunctionTool, item: ToolDescriptor, *, hook: FailingHook | None = None) -> ToolExecutor:
    """创建带较宽全局预算的独立执行器."""
    registry = RuntimeToolRegistry()
    registry.register(item, tool)
    return ToolExecutor(
        registry,
        budget_policy=BudgetPolicy(max_tool_output_tokens=100),
        hook=hook,
    )


@pytest.mark.anyio
async def test_tool_output_is_truncated_by_strictest_limit_and_hook_is_fail_open() -> None:
    """Descriptor 的较小输出预算生效，hook 失败不改变成功结果."""
    executor = executor_for(
        FunctionTool("one two three four five six seven eight nine ten"),
        descriptor(output_token_limit=5),
        hook=FailingHook(),
    )

    result = await executor.execute("demo", {}, context=ToolExecutionContext(user_id=uuid4()))

    assert result.status == "success"
    assert result.truncated is True
    assert result.output_tokens <= 5
    assert "ten" not in result.content


@pytest.mark.anyio
async def test_write_tool_requires_trusted_approval_before_execution() -> None:
    """模型参数不能伪造批准；只有可信上下文集合可以放行."""
    item = descriptor(risk=ToolRisk.WRITE, requires_approval=True)
    executor = executor_for(FunctionTool("written"), item)

    with pytest.raises(ToolApprovalRequired):
        await executor.execute(
            "demo",
            {"approved": True},
            context=ToolExecutionContext(user_id=uuid4()),
        )

    result = await executor.execute(
        "demo",
        {},
        context=ToolExecutionContext(user_id=uuid4(), approved_tool_names=frozenset({"demo"})),
    )
    assert result.status == "success"


@pytest.mark.anyio
async def test_timeout_and_exception_return_stable_safe_errors() -> None:
    """超时和普通异常不向模型暴露异常正文."""
    timeout_executor = executor_for(
        FunctionTool("late", delay=0.05),
        descriptor(timeout_seconds=0.001),
    )
    timed_out = await timeout_executor.execute(
        "demo",
        {},
        context=ToolExecutionContext(user_id=uuid4()),
    )
    assert timed_out.error_type == "TimeoutError"
    assert timed_out.status == "error"

    class SecretFailureTool(FunctionTool):
        async def invoke(self, arguments: dict[str, object], context: ToolExecutionContext) -> object:
            del arguments, context
            raise RuntimeError("credential=must-not-leak")

    failed = await executor_for(SecretFailureTool(None), descriptor()).execute(
        "demo",
        {},
        context=ToolExecutionContext(user_id=uuid4()),
    )
    assert failed.error_type == "RuntimeError"
    assert "credential" not in failed.content
