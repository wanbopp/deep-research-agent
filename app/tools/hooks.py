"""工具生命周期 hook；观测失败不能改变工具结果."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from app.core.logging import logger
from app.tools.contracts import ToolDescriptor


@dataclass(frozen=True, slots=True)
class ToolLifecycleEvent:
    """不包含参数和输出正文的安全生命周期事件."""

    tool_name: str
    namespace: str
    phase: str
    status: str | None = None
    truncated: bool | None = None
    error_type: str | None = None


class ToolHook(Protocol):
    """可选审计/事件投影 hook."""

    async def emit(self, event: ToolLifecycleEvent) -> None:
        """接收安全事件."""
        ...


class NoopToolHook:
    """默认零副作用 hook."""

    async def emit(self, event: ToolLifecycleEvent) -> None:
        """忽略事件."""
        del event


async def emit_safely(hook: ToolHook, event: ToolLifecycleEvent) -> None:
    """隔离 hook 故障，只记录异常类型."""
    try:
        await hook.emit(event)
    except Exception as error:
        logger.warning("tool_hook_failed", error_type=type(error).__name__)


def lifecycle_event(
    descriptor: ToolDescriptor,
    *,
    phase: str,
    status: str | None = None,
    truncated: bool | None = None,
    error_type: str | None = None,
) -> ToolLifecycleEvent:
    """从 descriptor 构造不含正文的事件."""
    return ToolLifecycleEvent(
        tool_name=descriptor.name,
        namespace=descriptor.namespace,
        phase=phase,
        status=status,
        truncated=truncated,
        error_type=error_type,
    )


__all__ = ["NoopToolHook", "ToolHook", "ToolLifecycleEvent", "emit_safely", "lifecycle_event"]
