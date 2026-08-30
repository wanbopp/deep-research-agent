"""DeepResearch FastAPI 应用入口."""

from asgi_correlation_id import CorrelationIdMiddleware
from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware

from app.api.v1.api import api_router
from app.api.metrics import router as metrics_router
from app.core.config import settings
from app.core.exception_handlers import register_exception_handlers
from app.core.middleware import RequestLoggingMiddleware
from app.infrastructure.lifespan import lifespan

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description=settings.DESCRIPTION,
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    # lifespan 是应用级资源所有权边界。Uvicorn 启动时先进入它，只有 startup
    # 成功并执行到 yield 后才开始接收请求；关闭时再执行 yield 后的清理逻辑。
    lifespan=lifespan,
)

# 注册异常处理的handler
register_exception_handlers(app)

# 注意中间件执行顺序：
# Starlette/FastAPI 中后添加的 middleware 更靠外层，会更早处理请求。
# 因此 CorrelationIdMiddleware 要在 RequestLoggingMiddleware 之后添加，
# 这样 request_started/request_completed 也能读到 request_id。
app.add_middleware(RequestLoggingMiddleware)

# CorrelationMiddleware 会为每个请求生成/读取 request_id
# Logging.py 中的 correlation_id.get() 会自动把它写进日志
app.add_middleware(CorrelationIdMiddleware)

# CORS 允许浏览器前端访问API
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册app router
app.include_router(api_router, prefix=settings.API_V1_STR)
app.include_router(metrics_router)
