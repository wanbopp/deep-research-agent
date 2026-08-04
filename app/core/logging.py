"""DeepResearch 日志配置模块

设计思路：
    - 基于 structlog 结构化日志
    - 控制台使用 ConsoleRenderer，方便开发时阅读
    - 文件使用 JSONRenderer，输出干净 JSONL，方便日志采集和检索
    - 使用 ContextVar 绑定请求级上下文，例如 request_id、session_id、user_id
"""
import logging
import sys
from contextvars import ContextVar
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import structlog
from asgi_correlation_id import correlation_id  # 获取fastAPI请求iD

from app.core.config import Environment, settings

# 确保日志目录存在。mkdir(..., exist_ok=True) 可重复执行，不会因为目录已存在而报错。
settings.LOG_DIR.mkdir(parents=True, exist_ok=True)

# ContextVar 保存的是“当前执行上下文”的日志字段。
# 在 FastAPI 中，每个请求会有自己的上下文，不会串到其他并发请求里。
_request_context: ContextVar[Optional[Dict[str, Any]]] = ContextVar("request_context", default=None)


def bind_context(**kwargs: Any) -> None:
    """把字段绑定到当前请求/任务的日志上下文中。

    示例：
        bind_context(session_id="xxx", user_id=123)

    之后同一上下文里的每一条 structlog 日志都会自动带上这些字段。
    """
    # 获取当前上下文；如果还没有绑定过字段，则从空 dict 开始。
    context = _request_context.get() or {}
    # 合并新字段。同名字段以后传入的为准，例如新的 session_id 会覆盖旧值。
    _request_context.set({**context, **kwargs})


def clear_context() -> None:
    """清空当前上下文，避免请求结束后字段泄漏到后续请求。"""
    _request_context.set(None)


def get_context() -> Dict[str, Any]:
    """读取当前日志上下文；没有上下文时返回空 dict，方便 processor 直接使用。"""
    return _request_context.get() or {}


def add_context_to_event_dict(
    logger: Any,
    method_name: str,
    event_dict: dict[str, Any],
) -> dict[str, Any]:
    """structlog processor：把 ContextVar 中的字段合并到日志事件里。"""
    context = get_context()
    if context:
        event_dict.update(context)
    return event_dict


def add_request_id_to_event_dict(
    logger: Any,
    method_name: str,
    event_dict: dict[str, Any],
) -> dict[str, Any]:
    """structlog processor：把 asgi-correlation-id 的 request_id 加入日志。"""
    request_id = correlation_id.get()
    if request_id:
        event_dict["request_id"] = request_id
    return event_dict


def get_log_file_path() -> Path:
    """按环境和日期生成当前 JSONL 日志文件路径。"""
    env_prefix = settings.ENVIRONMENT.value
    # JSONL 每一行都是一个独立 JSON 对象，天然适合追加写入、tail 查看和日志采集。
    return settings.LOG_DIR / f"{env_prefix}-{datetime.now().strftime('%Y-%m-%d')}.jsonl"


def get_structlog_processors(include_file_info: bool = True) -> List[Any]:
    """获取 structlog 公共 processors。

    processors 只负责“补字段”和“规范化异常”等公共处理，不负责最终渲染。
    最终渲染由每个 handler 上的 ProcessorFormatter 决定：
        - console_handler -> ConsoleRenderer
        - file_handler    -> JSONRenderer

    Args:
        include_file_info: 是否把 filename、lineno、func_name 等调用位置写入日志。
    """
    # 这组 processor 会同时作用于控制台日志和文件日志。
    processors = [
        structlog.stdlib.filter_by_level,  # 过滤日志等级
        structlog.stdlib.add_logger_name,  # 添加logger名称
        structlog.stdlib.add_log_level,  # 添加日志等级
        structlog.stdlib.PositionalArgumentsFormatter(),  # 添加参数格式化
        structlog.processors.TimeStamper(fmt='iso'),  # 添加时间
        structlog.processors.StackInfoRenderer(),  # 添加调用栈
        structlog.processors.format_exc_info,  # 处理 logger.exception 输出
        structlog.processors.UnicodeDecoder(),
        add_context_to_event_dict,  # 添加上下文信息
        add_request_id_to_event_dict,  # 添加request_id
    ]

    if include_file_info:
        # 开发/测试环境保留调用位置，方便从日志直接跳回问题代码。
        processors.append(
            structlog.processors.CallsiteParameterAdder(
                {
                    structlog.processors.CallsiteParameter.FILENAME,
                    structlog.processors.CallsiteParameter.FUNC_NAME,
                    structlog.processors.CallsiteParameter.LINENO,
                    structlog.processors.CallsiteParameter.MODULE,
                    structlog.processors.CallsiteParameter.PATHNAME,
                }
            )
        )

    # 给所有日志补环境字段，便于多环境日志混合采集时过滤。
    processors.append(lambda _, __, event_dict: {**event_dict, "environment": settings.ENVIRONMENT.value})

    return processors


def setup_logging() -> None:
    """初始化整个应用的日志系统。

    设计目标：
        1. 控制台输出适合开发者阅读。
        2. 文件输出保持干净 JSONL，方便日志采集和后续检索。
        3. structlog 只做字段加工，最终渲染交给不同 handler。
    """
    # 根据配置决定日志等级
    log_level = logging.DEBUG if settings.DEBUG else logging.INFO

    # 确保日志输出的目录存在
    settings.LOG_DIR.mkdir(parents=True, exist_ok=True)

    # 公共 processors 为控制台和文件补同一批结构化字段。
    # development/test 环境保留文件名、函数名、行号，production 环境日志更轻量。
    shared_processors = get_structlog_processors(
        include_file_info=settings.ENVIRONMENT in [Environment.DEVELOPMENT, Environment.TEST]
    )

    # 文件 handler 只负责写入文件；JSON 格式由 ProcessorFormatter + JSONRenderer 完成。
    # 这样 session_id、step 等字段会作为 JSON 顶层字段保留下来，而不是藏在 message 字符串里。
    file_handler = logging.FileHandler(
        get_log_file_path(),
        encoding="utf-8",
    )
    file_handler.setLevel(log_level)
    file_handler.setFormatter(
        structlog.stdlib.ProcessorFormatter(
            processor=structlog.processors.JSONRenderer(ensure_ascii=False),
            foreign_pre_chain=shared_processors,
        )
    )

    # 控制台 handler 面向开发者阅读，颜色和对齐只属于终端，不写入 JSONL 文件。
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(
        structlog.stdlib.ProcessorFormatter(
            processor=structlog.dev.ConsoleRenderer(),
            foreign_pre_chain=shared_processors,
        )
    )

    # 注册两个 handler 到标准库 logging。
    # force=True 用于开发阶段重复运行脚本时清理旧 handler，避免日志重复输出。
    logging.basicConfig(
        level=log_level,
        handlers=[file_handler, console_handler],
        force=True,
    )

    # structlog 这里只做字段加工，不做最终渲染。
    # wrap_for_formatter 会把 event_dict 交给各 handler 的 ProcessorFormatter：
    #   - console_handler -> ConsoleRenderer
    #   - file_handler    -> JSONRenderer
    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


# 模块导入时初始化日志系统。其他模块只需要导入 logger 即可使用。
setup_logging()

# 创建全局 logger 实例。
logger = structlog.get_logger()
log_level_name = "DEBUG" if settings.DEBUG else "INFO"
logger.info(
    "logging_initialized",
    environment=settings.ENVIRONMENT.value,
    log_level=log_level_name,
    log_format=settings.LOG_FORMAT,
    debug=settings.DEBUG,
)
