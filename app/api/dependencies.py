"""FastAPI application dependencies.

集中管理项目运行所依赖的对象、服务或依赖注入逻辑
"""

from collections.abc import AsyncIterator
from functools import lru_cache
from typing import Annotated

from fastapi import Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.chat.runtime import create_chat_runtime
from app.core.config import settings
from app.infrastructure.lifespan import get_application_resources
from app.services.auth import AuthService, PasswordHasher, TokenService
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


@lru_cache(maxsize=1)
def get_password_hasher() -> PasswordHasher:
    """创建并复用无状态 Argon2id 配置对象.

    每次 hash 仍会生成独立随机 salt；复用 PasswordHasher 不是复用密码结果，只是
    避免每个请求重复构造相同的算法配置。
    """
    return PasswordHasher()


@lru_cache(maxsize=1)
def get_token_service() -> TokenService:
    """从进程启动时加载的 Settings 创建并复用 JWT 服务.

    如果 secret 缺失或不安全，TokenService 会明确拒绝构造。生产环境修改 secret
    后必须重启所有实例，并确保同一轮部署中的实例使用相同 key。
    """
    return TokenService.from_settings(settings)


def get_auth_service(
    session: Annotated[AsyncSession, Depends(get_db_session)],
    password_hasher: Annotated[PasswordHasher, Depends(get_password_hasher)],
    token_service: Annotated[TokenService, Depends(get_token_service)],
) -> AuthService:
    """为一次 HTTP 请求组合 AuthService 及其依赖.

    Session 是请求级对象；PasswordHasher 与 TokenService 是无状态进程级对象。
    FastAPI 会先解析三个依赖，再把它们显式注入 application service。
    """
    return AuthService(
        session=session,
        password_hasher=password_hasher,
        token_service=token_service,
    )
