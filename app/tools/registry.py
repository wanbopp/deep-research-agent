"""工具 descriptor 与适配器的确定性注册表."""

from __future__ import annotations

from typing import Protocol

from app.tools.contracts import ToolDescriptor, ToolExecutionContext


class RuntimeTool(Protocol):
    """任何本地或远端工具适配器的最小接口."""

    async def invoke(self, arguments: dict[str, object], context: ToolExecutionContext) -> object:
        """使用可信上下文执行并返回原始结果."""
        ...


class RuntimeToolRegistry:
    """同时保存 descriptor 和实现，名称重复时 fail-fast."""

    def __init__(self) -> None:
        """初始化空注册表."""
        self._entries: dict[str, tuple[ToolDescriptor, RuntimeTool]] = {}

    def register(self, descriptor: ToolDescriptor, tool: RuntimeTool) -> None:
        """按公开 name 注册；namespace 仍用于审计和未来多源路由."""
        if descriptor.name in self._entries:
            raise ValueError(f"duplicate runtime tool name: {descriptor.name}")
        self._entries[descriptor.name] = (descriptor, tool)

    def resolve(self, name: str) -> tuple[ToolDescriptor, RuntimeTool]:
        """解析冻结快照；未知工具不泄漏完整内部注册表."""
        try:
            return self._entries[name]
        except KeyError:
            raise LookupError("unknown runtime tool") from None


__all__ = ["RuntimeTool", "RuntimeToolRegistry"]
