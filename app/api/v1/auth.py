"""用户注册与登录 HTTP 入口."""

from typing import Annotated

from fastapi import APIRouter, Depends, status
from fastapi.responses import JSONResponse

from app.api.dependencies import get_auth_service
from app.core.exception_handlers import build_error_content
from app.schemas.auth import LoginRequest, RegisterRequest, TokenResponse
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
