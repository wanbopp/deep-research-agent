"""用户注册与登录 HTTP 入口."""

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, status
from fastapi.responses import JSONResponse

from app.api.dependencies import (
    CurrentUserDependency,
    MatchingUserDependency,
    get_auth_service,
)
from app.core.exception_handlers import build_error_content
from app.schemas.auth import (
    AuthenticatedUser,
    LoginRequest,
    RegisterRequest,
    TokenResponse,
)
from app.schemas.base import ErrorResponse
from app.services.auth import (
    AuthService,
    EmailAlreadyRegisteredError,
    InvalidCredentialsError,
)


router = APIRouter(prefix="/auth", tags=["auth"])

# Annotated 把 Python 业务类型和 FastAPI 获取方式绑定在一起。route 只依赖
# AuthService，不知道 Session、Repository、Argon2 或 JWT service 如何构造。
AuthServiceDependency = Annotated[AuthService, Depends(get_auth_service)]


def _auth_error_response(
    *,
    status_code: int,
    code: str,
    message: str,
    headers: dict[str, str] | None = None,
) -> JSONResponse:
    """把认证业务错误转换为项目统一错误协议.

    只接受固定 code/message；调用方不能把邮箱、密码、hash、token 或底层异常文本
    传入。request_id 继续由统一 build_error_content 从请求上下文取得。
    """
    return JSONResponse(
        status_code=status_code,
        content=build_error_content(code=code, message=message),
        headers=headers,
    )


@router.post(
    "/register",
    response_model=TokenResponse,
    status_code=status.HTTP_201_CREATED,
    responses={
        status.HTTP_409_CONFLICT: {
            "model": ErrorResponse,
            "description": "Email is already registered",
        },
    },
)
async def register_user(
    request: RegisterRequest,
    service: AuthServiceDependency,
) -> TokenResponse | JSONResponse:
    """注册用户，并在数据库提交成功后返回 bearer access token."""
    try:
        return await service.register(request)
    except EmailAlreadyRegisteredError:
        return _auth_error_response(
            status_code=status.HTTP_409_CONFLICT,
            code="EMAIL_ALREADY_REGISTERED",
            message="Email is already registered",
        )


@router.post(
    "/login",
    response_model=TokenResponse,
    responses={
        status.HTTP_401_UNAUTHORIZED: {
            "model": ErrorResponse,
            "description": "Email or password is incorrect",
        },
    },
)
async def login_user(
    request: LoginRequest,
    service: AuthServiceDependency,
) -> TokenResponse | JSONResponse:
    """验证邮箱和密码；未知邮箱与错误密码返回完全相同的公开错误."""
    try:
        return await service.login(request)
    except InvalidCredentialsError:
        return _auth_error_response(
            status_code=status.HTTP_401_UNAUTHORIZED,
            code="INVALID_CREDENTIALS",
            message="Email or password is incorrect",
            headers={"WWW-Authenticate": "Bearer"},
        )


@router.get(
    "/me",
    response_model=AuthenticatedUser,
    responses={
        status.HTTP_401_UNAUTHORIZED: {
            "model": ErrorResponse,
            "description": "Bearer token is missing or invalid",
        },
    },
)
async def read_current_user(
    current_user: CurrentUserDependency,
) -> AuthenticatedUser:
    """返回服务端已经验签并经数据库确认的当前用户.

    Args:
        current_user: FastAPI 调用 get_current_user 后注入的可信身份。Route 不接触
            原始 Authorization header，也不会自行解码 JWT。

    Returns:
        只包含 user_id 和数据库当前 email 的公开安全模型。token、claims 和
        password_hash 不会进入响应。
    """
    return current_user


@router.get(
    "/users/{user_id}",
    response_model=AuthenticatedUser,
    responses={
        status.HTTP_401_UNAUTHORIZED: {
            "model": ErrorResponse,
            "description": "Bearer token is missing or invalid",
        },
        status.HTTP_403_FORBIDDEN: {
            "model": ErrorResponse,
            "description": "Authenticated user cannot access this user scope",
        },
    },
)
async def read_user_identity(
    user_id: UUID,
    current_user: MatchingUserDependency,
) -> AuthenticatedUser:
    """读取与当前身份相同的用户作用域，并明确演示 403 边界.

    Args:
        user_id: 客户端在路径中指定的用户 UUID。require_current_user_id 会在 route
            执行前把它与可信 current_user.user_id 比较。
        current_user: 只有完成认证且路径 ID 与登录用户相同时才会注入。

    Returns:
        当前已认证用户的安全身份模型。

    Notes:
        该路由用于需要显式用户作用域的 API。资源 UUID 查询不应先查询资源再返回
        403，而应直接在 Repository 中组合资源 ID 和用户 ID，跨用户时返回 not found。
    """
    # user_id 已由 dependency 校验。保留参数可以让 OpenAPI 明确展示路径输入，
    # current_user 才是后续业务代码应信任的身份来源。
    _ = user_id
    return current_user
