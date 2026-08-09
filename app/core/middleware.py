"""FastAPI 请求日志中间件.

中间件负责记录请求开始与结束日志、计算请求耗时，并把 method 和 path
绑定到当前日志上下文。request_id 由 CorrelationIdMiddleware 提供。
"""

import time
from typing import override

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

from app.core.logging import bind_context, clear_context, logger


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """记录每个 HTTP 请求的结构化日志."""

    @override
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        """记录单个请求的上下文、结果和耗时.

        call_next(request) 会把请求交给后续中间件或真正的路由函数。
        """
        start_time = time.perf_counter()
        status_code = 500

        # 把所有请求都会用到的字段绑定到日志上下文中
        # 后续这个请求的任何logger.info  都会自动带上这些字段
        bind_context(method=request.method, path=request.url.path)
        # request_id 不需要手动绑定，已经通过correlation_id.get 获取，在main中获取注册中间件即可

        logger.info("request_started")

        try:
            response = await call_next(request)
            status_code = response.status_code
            return response
        except Exception:
            logger.exception("request_failed")
            raise
        finally:
            # 无论请求成功、异常还是 404，都要记录 completed 并清理上下文。
            # 这能保证 request_started/request_completed 成对出现。
            duration_ms = round((time.perf_counter() - start_time) * 1000, 2)

            logger.info(
                "request_completed",
                status_code=status_code,
                duration_ms=duration_ms,
            )

            # 清空本次请求的上下文
            clear_context()
