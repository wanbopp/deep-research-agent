"""FastAPI 请求日志中间件.

中间件负责记录请求开始与结束日志、计算请求耗时，并把 method 和 path
绑定到当前日志上下文。request_id 由 CorrelationIdMiddleware 提供。
"""

import time
from typing import override

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

from app.core.logging import logger, logging_context
from app.observability import metrics


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """记录每个 HTTP 请求的结构化日志."""

    @override
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        """记录单个请求的上下文、结果和耗时.

        call_next(request) 会把请求交给后续中间件或真正的路由函数。
        """
        start_time = time.perf_counter()
        status_code = 500

        # token 作用域会在请求结束后恢复外层 api component，而不是清空整个
        # ContextVar；随后 Uvicorn 写 access log 时仍能路由到 api JSONL。
        with logging_context(method=request.method, path=request.url.path):
            # request_id 不需要手动绑定，已由 correlation_id processor 获取。
            logger.info("request_started")

            try:
                response = await call_next(request)
                status_code = response.status_code
                return response
            except Exception:
                logger.exception("request_failed")
                raise
            finally:
                # 无论请求成功、异常还是 404，都要记录 completed，保证事件成对。
                duration_ms = round((time.perf_counter() - start_time) * 1000, 2)

                logger.info(
                    "request_completed",
                    status_code=status_code,
                    duration_ms=duration_ms,
                )

                # 路由匹配发生在 call_next 内部，因此必须在返回后读取 route.path。
                # 404 或路由前异常没有模板时统一使用 unmatched，绝不能把包含用户 ID、
                # research_id 或任意 path segment 的原始 URL 写入 Prometheus 标签。
                route = request.scope.get("route")
                route_template = getattr(route, "path", "unmatched")
                metrics.observe_http(
                    method=request.method,
                    route=route_template,
                    status_code=status_code,
                    duration_seconds=duration_ms / 1000,
                )
