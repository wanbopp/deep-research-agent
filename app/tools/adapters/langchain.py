"""现有 LangChain BaseTool 到统一运行时的适配器."""

from langchain_core.tools import BaseTool

from app.tools.contracts import ToolExecutionContext


class LangChainToolAdapter:
    """借用现有 BaseTool，不改变其 schema 或模型曝光方式."""

    def __init__(self, tool: BaseTool) -> None:
        """保存无请求状态工具对象."""
        self._tool = tool

    async def invoke(self, arguments: dict[str, object], context: ToolExecutionContext) -> object:
        """执行工具；可信身份预留给未来用户资源工具，不写入模型参数."""
        del context
        return await self._tool.ainvoke(arguments)


__all__ = ["LangChainToolAdapter"]
