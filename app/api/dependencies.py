"""FastAPI application dependencies.

集中管理项目运行所依赖的对象、服务或依赖注入逻辑
"""

from collections.abc import AsyncIterator
from functools import lru_cache

from fastapi import Request
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.chat.runtime import create_chat_runtime
from app.infrastructure.lifespan import get_application_resources
from app.services.chat import ChatService


@lru_cache(maxsize=1)
def get_chat_service() -> ChatService:
    """创建并复用进程内唯一的聊天应用服务."""
    graph = create_chat_runtime()
    return ChatService(graph)


async def get_db_session(request: Request) -> AsyncIterator[AsyncSession]:
    """为一次普通 HTTP 请求创建并关闭独立 AsyncSession.

    dependency 只负责 Session 生命周期，不自动 commit。事务是业务语义，后续
    application service 应通过 ``async with session.begin()`` 决定哪些 Repository
    写操作必须一起成功或一起回滚。

    对于 LangGraph 并行节点，不应把这里生成的同一个 Session 并发传给多个节点；
    每个并行工作单元都应从 ``orm_session_factory`` 创建自己的短生命周期 Session。
    """
    resources = get_application_resources(request.app)
    async with resources.orm_session_factory() as session:
        yield session
