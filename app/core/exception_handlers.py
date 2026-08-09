"""FastAPI 全局异常处理.

负责将框架异常转换为统一的 HTTP 响应。
"""

from typing import Any, cast

from asgi_correlation_id import correlation_id
from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.types import HTTPExceptionHandler
from app.core.logging import logger


def get_request_id() -> str | None:
    """获取当前请求的 correlation_id.

    CorrelationIdMiddleware 在请求进入时设置这个 ContextVar。
    handler 不生成新 ID，避免响应和日志失去关联。
    """
    return correlation_id.get()


def build_error_content(
    *,
    code: str,
    message: str,
    details: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """构造所有异常处理器共享的错误响应体.

    details 为 None 或空列表时不输出该字段；只有真正存在
    字段级补充信息时，才把 details 加入 error 对象。
    """
    # 先构造 error 内层字典。
    error: dict[str, Any] = {"code": code, "message": message}

    # details 为 None 或空列表时，不添加该字段。
    if details:
        error["details"] = details

    return {"error": error, "request_id": get_request_id()}


async def http_exception_handler(
    request: Request,
    exc: StarletteHTTPException,
) -> JSONResponse:
    """处理 404 以及路由主动抛出的 HTTP 错误."""
    # method 和 path 是排查问题所需的上下文，更适合记录到日志。
    logger.warning(
        "http_exception",
        status_code=exc.status_code,
        method=request.method,
        path=request.url.path,
    )
    # HTTPException.detail 通常是字符串。
    # 如果以后有人传入字典或列表，这里先使用通用文案，
    # 避免把不可控的内部结构直接作为公开错误信息。
    message = exc.detail if isinstance(exc.detail, str) else "HTTP request failed"

    return JSONResponse(
        # 保留原始状态码，例如 404 仍然返回 404。
        status_code=exc.status_code,
        # content 才是真正返回给客户端的 JSON body。
        content=build_error_content(
            code="HTTP_ERROR",
            message=message,
        ),
        # 保留原异常携带的响应头。
        headers=exc.headers,
    )


async def validation_exception_handler(
    request: Request,
    exc: RequestValidationError,
) -> JSONResponse:
    """将 FastAPI/Pydantic 校验错误转换为稳定、可读的字段列表."""
    details: list[dict[str, Any]] = []

    for item in exc.errors():
        # loc 通常类似 ("query", "limit") 或 ("body", "user", "email")。
        # 每一部分不一定都是字符串，例如列表下标可能是整数，
        # 因此先调用 str()，再拼接为客户端容易理解的字段路径。
        field = " -> ".join(str(part) for part in item["loc"])

        details.append(
            {
                "field": field,
                "message": item["msg"],
                "type": item["type"],
            }
        )

    # 只记录错误数量和请求信息。
    # 不要直接记录 exc.errors()，因为其中可能包含用户提交的 input。
    logger.warning(
        "validation_exception",
        error_count=len(details),
        method=request.method,
        path=request.url.path,
    )

    return JSONResponse(
        # 422 表示请求格式能够被服务器理解，
        # 但参数内容没有通过校验。
        status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
        content=build_error_content(
            code="VALIDATION_ERROR",
            # message 是整个错误响应的概括，必须是字符串。
            message="Request validation failed",
            # 具体有哪些字段错误，通过 details 单独传递。
            details=details,
        ),
    )


async def unhandled_exception_handler(
    request: Request,
    exc: Exception,
) -> JSONResponse:
    """处理未预期异常，隐藏内部细节并保留服务端 traceback."""
    # logger.exception 会记录当前异常的 traceback。
    # str(exc) 可以进入服务端日志，但不能进入客户端响应。
    logger.exception(
        "unhandled_exception",
        error_type=type(exc).__name__,
        error=str(exc),
        method=request.method,
        path=request.url.path,
    )

    content = build_error_content(
        code="INTERNAL_SERVER_ERROR",
        message="Internal server error",
    )
    request_id = content["request_id"]

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=content,
        # 最外层发送 500 时绕过了 CorrelationIdMiddleware，
        # 因此需要显式补充同一个 request ID。
        headers=({"X-Request-ID": request_id} if isinstance(request_id, str) else None),
    )


def register_exception_handlers(app: FastAPI) -> None:
    """集中注册异常类型与 handler 的对应关系."""
    # Starlette 的类型声明无法表达“异常类与 handler 参数类型相关联”。
    # FastAPI 会在运行时按异常类型正确分发，因此只在注册边界进行类型转换。
    app.add_exception_handler(
        StarletteHTTPException,
        cast(HTTPExceptionHandler, http_exception_handler),
    )

    app.add_exception_handler(
        RequestValidationError,
        cast(HTTPExceptionHandler, validation_exception_handler),
    )

    app.add_exception_handler(
        Exception,
        unhandled_exception_handler,
    )
