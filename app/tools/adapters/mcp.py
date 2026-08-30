"""MCP 工具接口占位；本阶段不连接 Server、不读取凭据."""

from typing import Protocol

from app.tools.contracts import ToolExecutionContext


class MCPToolClient(Protocol):
    """未来 MCP 客户端必须实现的窄调用接口."""

    async def call_tool(self, name: str, arguments: dict[str, object]) -> object:
        """调用已经由部署层连接和授权的工具."""
        ...


class MCPToolAdapter:
    """显式注入 client 的适配器，不创建连接或持有密钥."""

    def __init__(self, *, client: MCPToolClient, remote_name: str) -> None:
        """保存已装配客户端和固定远端名称."""
        if not remote_name:
            raise ValueError("remote MCP tool name must not be empty")
        self._client = client
        self._remote_name = remote_name

    async def invoke(self, arguments: dict[str, object], context: ToolExecutionContext) -> object:
        """模型不能覆盖远端名称或可信用户身份."""
        del context
        return await self._client.call_tool(self._remote_name, arguments)


__all__ = ["MCPToolAdapter", "MCPToolClient"]
