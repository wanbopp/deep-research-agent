"""FastAPI application dependencies.

集中管理项目运行所依赖的对象、服务或依赖注入逻辑
"""

from collections.abc import AsyncIterator
from functools import lru_cache
from typing import Annotated
from uuid import UUID

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.infrastructure.lifespan import (
    get_application_chat_cleanup_service,
    get_application_chat_service,
    get_application_resources,
)
from app.repositories import UserRepository
from app.schemas.auth import AuthenticatedUser
from app.services.auth import (
    AuthService,
    InvalidAccessTokenError,
    PasswordHasher,
    TokenService,
)
from app.services.chat import ChatService
from app.services.chat_session_cleanup import ChatSessionCleanupService
from app.services.chat_sessions import ChatSessionService


# 当前 /auth/login 接收 JSON，并不是 OAuth2 Password Flow 规定的 form token
# endpoint，因此使用 HTTPBearer 准确描述“客户端携带 Bearer token”的协议。
# ``auto_error=False`` 让缺失 header 和错误认证方案也交给 get_current_user，
# 从而与篡改、过期、未知用户 token 产生同一种公开 401。
http_bearer = HTTPBearer(auto_error=False)


def get_chat_service(request: Request) -> ChatService:
    """返回当前 FastAPI 应用在 startup 创建的共享聊天服务."""
    return get_application_chat_service(request.app)


def get_chat_session_cleanup_service(
    request: Request,
) -> ChatSessionCleanupService:
    """返回 startup 创建的共享会话清理协调器.

    Args:
        request: 当前 FastAPI 请求。这里只读取 ``request.app``，不会读取请求
            body、token 或客户端提供的内部 thread key。

    Returns:
        lifespan 已组合好的 ``ChatSessionCleanupService``。它与 ``ChatService``
        共用 checkpointer、Redis guard 和 internal thread ID 映射。

    Raises:
        RuntimeError: 应用尚未完成 startup，或者已经进入 shutdown。正常 HTTP
            请求不会处于这两个阶段，因此该错误表示应用生命周期接线问题。

    Notes:
        这里不能像请求级 ``ChatSessionService`` 那样重新构造 cleanup service。
        清理操作必须使用 lifespan 中与 Graph 完全相同的 saver 和 guard，才能
        保证删除与同一会话的 Agent 执行属于同一个并发域。
    """
    return get_application_chat_cleanup_service(request.app)


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


def get_chat_session_service(
    session: Annotated[AsyncSession, Depends(get_db_session)],
) -> ChatSessionService:
    """为一次 HTTP 请求构造业务会话 application service.

    Args:
        session: FastAPI 从 get_db_session 注入的请求级 AsyncSession。相同请求中的
            get_current_user 会复用它，但认证事务会在 route 执行前结束。

    Returns:
        使用该 Session 协调会话事务与所有权查询的 ChatSessionService。
    """
    return ChatSessionService(session)


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


def _authentication_required() -> HTTPException:
    """创建不泄漏认证失败原因的统一 401 异常.

    Returns:
        每次调用得到一个新的 HTTPException。响应包含标准
        ``WWW-Authenticate: Bearer`` header，提示客户端需要 bearer token。

    Notes:
        不复用同一个异常实例，避免并发请求共享 traceback 状态。公开文案也不会
        区分 token 缺失、篡改、过期或对应用户不存在。
    """
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )


async def get_current_user(
    credentials: Annotated[
        HTTPAuthorizationCredentials | None,
        Depends(http_bearer),
    ],
    session: Annotated[AsyncSession, Depends(get_db_session)],
    token_service: Annotated[TokenService, Depends(get_token_service)],
) -> AuthenticatedUser:
    """把外部 Bearer token 转换为数据库确认过的可信用户上下文.

    Args:
        credentials: HTTPBearer 从 Authorization header 提取的认证方案与凭据。
            header 缺失或不是 Bearer 方案时为 ``None``；本方法不会记录该值。
        session: 当前 HTTP 请求独享的 AsyncSession，用于确认 token 对应的用户
            仍然存在，并读取数据库中的最新规范化邮箱。
        token_service: 进程级 JWT 服务，负责验签、算法 allowlist、用途和时间校验。

    Returns:
        只含 ``user_id`` 与 ``email`` 的 AuthenticatedUser。原 token、claims、密码
        和 password hash 都不会继续进入 route、service 或未来 Agent runtime。

    Raises:
        HTTPException: token 缺失、畸形、篡改、过期，或其 ``sub`` 在数据库中
            已不存在时统一抛出 401。数据库连接故障不会伪装成 401，而会继续交给
            全局 500 handler，便于运维发现真实基础设施问题。
    """
    if credentials is None:
        raise _authentication_required()

    try:
        claims = token_service.decode_access_token(credentials.credentials)
    except InvalidAccessTokenError:
        # 不把底层异常串进 HTTPException，避免错误链或响应泄漏 token 诊断细节。
        raise _authentication_required() from None

    # SQLAlchemy 的 SELECT 也会自动开启事务。如果这里不显式结束事务，同一请求
    # 后续的 ChatSessionService 再执行 session.begin() 就会报 transaction already
    # begun。认证查询使用一个短事务，结束后同一请求级 Session 可以顺序开启业务事务。
    async with session.begin():
        user = await UserRepository(session).get_by_id(claims.sub)
        if user is None:
            # 有效签名并不保证账户仍然存在。删除用户后，旧 access token 会在这里失效。
            raise _authentication_required()

        # 在事务内转换为不依赖 ORM 的可信上下文；route 不持有 User ORM 实体。
        authenticated_user = AuthenticatedUser(user_id=user.id, email=user.email)

    return authenticated_user


# 类型别名同时保留 Python 类型和 FastAPI 依赖来源。Route 只声明“需要可信用户”，
# 不需要知道 JWT、Session 或 Repository 的组合过程。
# 声明类型是 AuthenticatedUser 值从 get_current_user 中获取
CurrentUserDependency = Annotated[AuthenticatedUser, Depends(get_current_user)]


def require_current_user_id(
    user_id: UUID,
    current_user: CurrentUserDependency,
) -> AuthenticatedUser:
    """要求路径中的用户 ID 与当前已认证用户一致.

    Args:
        user_id: FastAPI 从当前 route 的 ``{user_id}`` 路径参数解析出的 UUID。
        current_user: get_current_user 已完成 JWT 验签和数据库确认的可信身份。

    Returns:
        ID 相同时原样返回 current_user，供后续 route 或 service 继续使用。

    Raises:
        HTTPException: 身份有效但试图访问另一个用户的用户级资源时抛出 403。

    Notes:
        这个 helper 适用于“客户端明确请求另一个用户作用域”的场景。会话、文档和
        研究任务采用 ``resource_id + current_user.user_id`` SQL 过滤，跨用户时统一
        表现为资源不存在，以免额外泄漏某个资源 UUID 是否真实存在。
    """
    if user_id != current_user.user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient permissions",
        )
    return current_user


MatchingUserDependency = Annotated[
    AuthenticatedUser,
    Depends(require_current_user_id),
]
