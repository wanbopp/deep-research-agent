"""FastAPI application dependencies.

集中管理项目运行所依赖的对象、服务或依赖注入逻辑
"""

from collections.abc import AsyncIterator
from functools import lru_cache
from ipaddress import ip_address
from typing import Annotated
from uuid import UUID

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.infrastructure.lifespan import (
    get_application_cache,
    get_application_chat_cleanup_service,
    get_application_chat_service,
    get_application_file_storage,
    get_application_rate_limiter,
    get_application_rate_limit_policies,
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
from app.services.knowledge import KnowledgeService
from app.services.rate_limit import (
    RateLimiter,
    RateLimitIdentityUnavailableError,
    RateLimitPolicies,
    enforce_rate_limit,
)
from app.services.research import ResearchService


# 当前 /auth/login 接收 JSON，并不是 OAuth2 Password Flow 规定的 form token
# endpoint，因此使用 HTTPBearer 准确描述“客户端携带 Bearer token”的协议。
# ``auto_error=False`` 让缺失 header 和错误认证方案也交给 get_current_user，
# 从而与篡改、过期、未知用户 token 产生同一种公开 401。
http_bearer = HTTPBearer(auto_error=False)


def get_chat_service(request: Request) -> ChatService:
    """返回当前 FastAPI 应用在 startup 创建的共享聊天服务."""
    return get_application_chat_service(request.app)


def get_rate_limiter(request: Request) -> RateLimiter:
    """返回 startup 创建并借用共享 Redis client 的限流器."""
    return get_application_rate_limiter(request.app)


def get_rate_limit_policies(request: Request) -> RateLimitPolicies:
    """返回 startup 已校验的不可变限流策略集合."""
    return get_application_rate_limit_policies(request.app)


def _get_client_ip_identity(request: Request) -> str:
    """从当前直连边界取得规范化客户端 IP.

    Args:
        request: Starlette 根据实际 ASGI 连接建立的请求对象。

    Returns:
        规范化后的 IPv4 或 IPv6 字符串，只会在限流适配器内参与摘要计算。

    Raises:
        RateLimitIdentityUnavailableError: ASGI server 没有提供客户端地址，或地址
            不是合法 IP。此时无法安全计数，不能用公共常量把所有用户混在一起。

    Security:
        当前没有配置可信反向代理边界，因此刻意不读取 ``X-Forwarded-For``。
        直接相信该 header 会允许客户端伪造不同 IP 绕过注册和登录限流。
    """
    if request.client is None:
        raise RateLimitIdentityUnavailableError

    try:
        return str(ip_address(request.client.host))
    except ValueError:
        raise RateLimitIdentityUnavailableError from None


async def enforce_auth_rate_limit(
    request: Request,
    limiter: Annotated[RateLimiter, Depends(get_rate_limiter)],
    policies: Annotated[RateLimitPolicies, Depends(get_rate_limit_policies)],
) -> None:
    """在注册或登录业务执行前按可信客户端 IP 消费一次额度.

    Args:
        request: 用于读取 ASGI 直连客户端地址；不会读取 email 或 password。
        limiter: lifespan 发布的共享 Redis 限流适配器。
        policies: startup 已校验的固定策略集合。

    Raises:
        RateLimitExceededError: Redis 明确确认 IP 已耗尽认证入口额度。
        RateLimitBackendUnavailableError: Redis 无法完成原子判断。
        RateLimitIdentityUnavailableError: 无法从可信边界建立客户端 IP。
    """
    identity = _get_client_ip_identity(request)
    await enforce_rate_limit(
        limiter,
        policy=policies.auth,
        identity=f"ip:{identity}",
    )


# 匿名认证入口只需要“依赖成功执行”这个事实，不需要把判断对象传给 route。
AnonymousAuthRateLimitDependency = Annotated[None, Depends(enforce_auth_rate_limit)]


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
    request: Request,
    session: Annotated[AsyncSession, Depends(get_db_session)],
) -> ChatSessionService:
    """为一次 HTTP 请求构造业务会话 application service.

    Args:
        request: 当前 FastAPI 请求。这里只用 ``request.app`` 读取 lifespan 已发布
            的共享 Cache，不读取客户端 body 或身份字段。
        session: FastAPI 从 get_db_session 注入的请求级 AsyncSession。相同请求中的
            get_current_user 会复用它，但认证事务会在 route 执行前结束。

    Returns:
        使用请求级 Session 与共享 Cache 协调事务、所有权查询和列表缓存的
        ChatSessionService。
    """
    return ChatSessionService(
        session,
        cache=get_application_cache(request.app),
        cache_ttl_seconds=settings.CACHE_TTL_SECONDS,
    )


def get_knowledge_service(
    request: Request,
    session: Annotated[AsyncSession, Depends(get_db_session)],
) -> KnowledgeService:
    """为一次 HTTP 请求组合知识文档应用服务.

    Args:
        request: 用于读取 lifespan 发布的共享 FileStorage。
        session: 当前请求独享的 AsyncSession；认证依赖会复用同一个对象，但在
            route 执行前已经结束认证查询事务。

    Returns:
        使用请求级事务和固定上传策略的 KnowledgeService。
    """
    return KnowledgeService(
        session,
        storage=get_application_file_storage(request.app),
        allowed_content_types=settings.KNOWLEDGE_ALLOWED_CONTENT_TYPES,
        max_upload_bytes=settings.KNOWLEDGE_MAX_UPLOAD_BYTES,
        read_chunk_bytes=settings.KNOWLEDGE_UPLOAD_READ_CHUNK_BYTES,
        index_max_attempts=settings.KNOWLEDGE_INDEX_MAX_ATTEMPTS,
    )


def get_research_service(
    session: Annotated[AsyncSession, Depends(get_db_session)],
) -> ResearchService:
    """为一次 HTTP 请求创建研究任务应用服务.

    Session 只用于任务、事件和所有权的短事务；真正的长时间研究由独立 worker
    从数据库领取，绝不能在 API 请求的 Session 或 BackgroundTasks 中执行。
    """
    return ResearchService(session)


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


async def enforce_agent_rate_limit(
    current_user: CurrentUserDependency,
    limiter: Annotated[RateLimiter, Depends(get_rate_limiter)],
    policies: Annotated[RateLimitPolicies, Depends(get_rate_limit_policies)],
) -> None:
    """在 ChatService 与 Agent Graph 执行前按可信 user_id 消费额度.

    Args:
        current_user: ``get_current_user`` 验签并查询数据库后建立的可信身份。
        limiter: lifespan 发布的共享 Redis 限流适配器。
        policies: startup 已校验的固定策略集合。

    Raises:
        RateLimitExceededError: 用户已耗尽当前 Agent 请求窗口。
        RateLimitBackendUnavailableError: Redis 无法可靠判断额度。

    Notes:
        这里不能使用客户端可自由创建的 ``thread_id``，否则同一用户只需不断换
        thread 就能绕过模型成本保护。FastAPI 还会缓存同一请求中的 dependency
        结果，所以 route 再声明 CurrentUserDependency 不会重复查询用户。
    """
    await enforce_rate_limit(
        limiter,
        policy=policies.agent,
        identity=f"user:{current_user.user_id}",
    )


# dependency 在 route 函数体之前完成。拒绝时不会构造 ChatService 调用、进入
# execution guard、运行 LangGraph 或向真实模型 provider 发送请求。
AgentRateLimitDependency = Annotated[None, Depends(enforce_agent_rate_limit)]


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
