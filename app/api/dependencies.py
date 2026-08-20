"""FastAPI application dependencies.

集中管理项目运行所依赖的对象、服务或依赖注入逻辑
"""

from functools import lru_cache

from app.agents.chat.runtime import create_chat_runtime
from app.services.chat import ChatService


@lru_cache(maxsize=1)
def get_chat_service() -> ChatService:
    """创建并复用进程内唯一的聊天应用服务."""
    graph = create_chat_runtime()
    return ChatService(graph)
