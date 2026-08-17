"""Time tools for the chat Agent."""

from datetime import UTC, datetime

from langchain_core.tools import tool


# @tool 会把普通 Python 函数包装成 BaseTool
@tool
def get_current_utc_time() -> str:
    """Return the current UTC timestamp in ISO 8601 format."""
    # 使用带时区的 UTC 时间，避免产生含义不明确的 naive datetime。
    # 转换为字符串后，可以安全地放进后续的 ToolMessage.content。
    return datetime.now(UTC).isoformat()
