"""Registry for tools available to the chat Agent."""

from collections.abc import Sequence

from langchain_core.tools import BaseTool


class ToolRegistry:
    """保存 Chat Agent 允许模型调用的工具白名单."""

    def __init__(self, tools: Sequence[BaseTool]) -> None:
        """复制工具序列，并拒绝重复名称."""
        self._tools: dict[str, BaseTool] = {}

        for tool in tools:
            if tool.name in self._tools:
                raise ValueError(f"duplicate tool name: {tool.name}")
            self._tools[tool.name] = tool

    def names(self) -> tuple[str, ...]:
        """按照注册顺序返回全部工具名称."""
        return tuple(self._tools)

    def tools(self) -> tuple[BaseTool, ...]:
        """返回可安全传给 LLMService.call() 的工具快照."""
        # 不要直接暴露内部 dict。
        return tuple(self._tools.values())

    def resolve(self, name: str) -> BaseTool:
        """根据模型返回的名称解析已注册工具."""
        try:
            return self._tools[name]
        except KeyError:
            # 使用 from None 隐藏内部字典查找异常。
            raise LookupError(f"unknown tool: {name}; available: {self.names()}") from None
