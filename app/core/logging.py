"""DeepResearch 日志配置模块.

设计思路：
    - 基于 structlog 结构化日志
    - 控制台使用 ConsoleRenderer，方便开发时阅读
    - 文件使用 JSONRenderer，输出干净 JSONL，方便日志采集和检索
    - 使用 ContextVar 绑定请求级上下文，例如 request_id、session_id、user_id.
"""

import logging
import sys
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
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

# 组件名同时是日志文件路由协议，必须保持有限集合，不能使用用户、任务或工具输入
# 动态生成文件名。未显式绑定的启动和第三方日志统一进入 runtime。
LOG_COMPONENTS = ("runtime", "api", "research-worker", "index-worker")


def bind_context(**kwargs: Any) -> None:
    """把字段绑定到当前请求/任务的日志上下文中.

    示例：
        bind_context(session_id="xxx", user_id=123)

    之后同一上下文里的每一条 structlog 日志都会自动带上这些字段.
    """
    # 获取当前上下文；如果还没有绑定过字段，则从空 dict 开始。
    context = _request_context.get() or {}
    # 合并新字段。同名字段以后传入的为准，例如新的 session_id 会覆盖旧值。
    _request_context.set({**context, **kwargs})


@contextmanager
def logging_context(**kwargs: Any) -> Iterator[None]:
    """临时增量绑定日志字段，退出时恢复进入前的完整上下文.

    与 ``clear_context`` 不同，这个作用域不会删除 Supervisor 绑定的 component。
    因此请求中间件结束后，随后由 Uvicorn 写出的 access log 仍能回到 api 文件。
    """
    current = _request_context.get() or {}
    token = _request_context.set({**current, **kwargs})
    try:
        yield
    finally:
        _request_context.reset(token)


def clear_context() -> None:
    """清空当前上下文，避免请求结束后字段泄漏到后续请求."""
    _request_context.set(None)


@contextmanager
def component_context(component: str) -> Iterator[None]:
    """在当前同步/异步调用链中绑定一个有限日志组件.

    普通 context manager 可以安全跨越 ``await``：ContextVar 状态属于当前 Task，
    新建的子 Task 会继承它。退出时使用 token 精确恢复外层上下文，不会误删请求
    或 Supervisor 已经绑定的其他字段。

    Args:
        component: ``api``、``research-worker``、``index-worker`` 或 ``runtime``。

    Raises:
        ValueError: component 不在固定允许列表中。
    """
    if component not in LOG_COMPONENTS:
        raise ValueError("unsupported log component")
    with logging_context(component=component):
        yield


def get_context() -> Dict[str, Any]:
    """读取当前日志上下文；没有上下文时返回空 dict，方便 processor 直接使用."""
    return _request_context.get() or {}


def add_context_to_event_dict(
    logger: Any,
    method_name: str,
    event_dict: dict[str, Any],
) -> dict[str, Any]:
    """Structlog processor：把 ContextVar 中的字段合并到日志事件里."""
    context = get_context()
    if context:
        event_dict.update(context)
    # 未绑定组件的模块导入、Supervisor 和第三方日志进入 runtime 文件。这样每条
    # JSONL 都带有明确组件，同时避免为动态值创建无界文件集合。
    event_dict.setdefault("component", "runtime")
    return event_dict


def add_request_id_to_event_dict(
    logger: Any,
    method_name: str,
    event_dict: dict[str, Any],
) -> dict[str, Any]:
    """Structlog processor：把 asgi-correlation-id 的 request_id 加入日志."""
    request_id = correlation_id.get()
    if request_id:
        event_dict["request_id"] = request_id
    return event_dict


def get_log_file_path(component: str = "runtime") -> Path:
    """按环境、组件和日期生成当前 JSONL 日志文件路径."""
    if component not in LOG_COMPONENTS:
        raise ValueError("unsupported log component")
    env_prefix = settings.ENVIRONMENT.value
    # JSONL 每一行都是一个独立 JSON 对象，天然适合追加写入、tail 查看和日志采集。
    return settings.LOG_DIR / f"{env_prefix}-{component}-{datetime.now().strftime('%Y-%m-%d')}.jsonl"


def _record_component(record: logging.LogRecord) -> str:
    """从 structlog 事件或当前 ContextVar 读取安全组件名."""
    if isinstance(record.msg, Mapping):
        value = record.msg.get("component")
        if isinstance(value, str) and value in LOG_COMPONENTS:
            return value
    value = get_context().get("component")
    if isinstance(value, str) and value in LOG_COMPONENTS:
        return value
    return "runtime"


class ComponentLogFilter(logging.Filter):
    """只允许目标组件进入对应 JSONL handler."""

    def __init__(self, component: str) -> None:
        """保存一个经过有限集合校验的目标组件."""
        super().__init__()
        if component not in LOG_COMPONENTS:
            raise ValueError("unsupported log component")
        self._component = component

    def filter(self, record: logging.LogRecord) -> bool:
        """返回当前记录是否属于这个 handler."""
        return _record_component(record) == self._component


def get_structlog_processors(include_file_info: bool = True) -> List[Any]:
    """获取 structlog 公共 processors.

    processors 只负责“补字段”和“规范化异常”等公共处理，不负责最终渲染。
    最终渲染由每个 handler 上的 ProcessorFormatter 决定：
        - console_handler -> ConsoleRenderer
        - file_handler    -> JSONRenderer

    Args:
        include_file_info: 是否把 filename、lineno、func_name 等调用位置写入日志。
    """
    # 这组 processor 会同时作用于控制台日志和文件日志。
    processors = [
        structlog.stdlib.add_logger_name,  # 添加logger名称
        structlog.stdlib.add_log_level,  # 添加日志等级
        structlog.stdlib.PositionalArgumentsFormatter(),  # 添加参数格式化
        structlog.processors.TimeStamper(fmt="iso"),  # 添加时间
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
    """初始化整个应用的日志系统.

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

    # 每个 handler 只写一个有限组件。过滤发生在渲染前，既支持 structlog 的 dict
    # 消息，也支持 Uvicorn 等标准库日志通过当前 Task 的 component ContextVar 路由。
    file_handlers: list[logging.Handler] = []
    for component in LOG_COMPONENTS:
        file_handler = logging.FileHandler(
            get_log_file_path(component),
            encoding="utf-8",
        )
        file_handler.setLevel(log_level)
        file_handler.addFilter(ComponentLogFilter(component))
        file_handler.setFormatter(
            structlog.stdlib.ProcessorFormatter(
                processor=structlog.processors.JSONRenderer(ensure_ascii=False),
                foreign_pre_chain=shared_processors,
            )
        )
        file_handlers.append(file_handler)

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
        handlers=[*file_handlers, console_handler],
        force=True,
    )

    # 数据库驱动的原始连接诊断可能包含主机、端口或连接参数，不适合直接进入
    # 应用日志。依赖状态统一由 infrastructure probe 记录稳定状态和安全错误代码，
    # 因此关闭这些驱动自己的日志输出，不影响项目内其他标准库日志。
    for logger_name in ("neo4j", "psycopg", "psycopg.pool", "redis"):
        logging.getLogger(logger_name).setLevel(logging.CRITICAL + 1)

    # structlog 这里只做字段加工，不做最终渲染。
    # wrap_for_formatter 会把 event_dict 交给各 handler 的 ProcessorFormatter：
    #   - console_handler -> ConsoleRenderer
    #   - file_handler    -> JSONRenderer
    structlog.configure(
        processors=[
            # 这里只处理 structlog 日志，因此 logger 一定存在。
            structlog.stdlib.filter_by_level,  # 过滤日志等级
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
