"""统一启动 FastAPI 与 Research Worker 的 Runtime CLI."""

import argparse
import asyncio
import os
import selectors
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Coroutine, Literal, Protocol, cast

import uvicorn

type RuntimeMode = Literal["all", "api", "worker"]
type WorkerRunner = Callable[[], Coroutine[Any, Any, None]]


class ApiServer(Protocol):
    """Supervisor 依赖的最小 API Server 生命周期协议."""

    should_exit: bool
    started: bool

    async def serve(self) -> None:
        """运行服务器直到收到停止信号."""


@dataclass(frozen=True, slots=True)
class RuntimeOptions:
    """统一 Runtime 的命令行配置."""

    mode: RuntimeMode
    host: str
    port: int
    log_level: str


def build_parser() -> argparse.ArgumentParser:
    """构造 CLI parser；不解析参数，便于测试和未来复用."""
    parser = argparse.ArgumentParser(
        prog="deep-research-runtime",
        description="启动 DeepResearch API、Research Worker 或完整 Web Runtime。",
    )
    parser.add_argument(
        "--mode",
        choices=("all", "api", "worker"),
        default=os.getenv("DEEP_RESEARCH_RUNTIME_MODE", "all"),
        help="运行组件；默认 all，同时启动 API 和 Research Worker。",
    )
    parser.add_argument(
        "--host",
        default=os.getenv("DEEP_RESEARCH_RUNTIME_HOST", "127.0.0.1"),
        help="API 监听地址；默认只绑定本机。",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("DEEP_RESEARCH_RUNTIME_PORT", "8000")),
        help="API 监听端口，默认 8000。",
    )
    parser.add_argument(
        "--log-level",
        choices=("critical", "error", "warning", "info", "debug", "trace"),
        default=os.getenv("DEEP_RESEARCH_RUNTIME_LOG_LEVEL", "info"),
        help="Uvicorn 日志等级，默认 info。",
    )
    return parser


def parse_options(argv: Sequence[str] | None = None) -> RuntimeOptions:
    """解析并校验 Runtime 参数.

    Args:
        argv: 不包含程序名的参数；None 表示读取当前进程命令行。

    Returns:
        已完成基本边界校验的不可变 RuntimeOptions。
    """
    namespace = build_parser().parse_args(argv)
    if not 1 <= namespace.port <= 65535:
        raise ValueError("port must be between 1 and 65535")
    if not namespace.host.strip():
        raise ValueError("host must not be empty")
    return RuntimeOptions(
        mode=cast(RuntimeMode, namespace.mode),
        host=namespace.host,
        port=namespace.port,
        log_level=namespace.log_level,
    )


def create_api_server(options: RuntimeOptions) -> uvicorn.Server:
    """根据统一配置创建不启用 reload 的 Uvicorn Server.

    ``all`` 模式由本进程 Supervisor 管理生命周期，不能再交给 Uvicorn reload
    父进程复制应用，否则每次 reload 都可能额外启动一个队列消费者。
    """
    config = uvicorn.Config(
        "app.main:app",
        host=options.host,
        port=options.port,
        log_level=options.log_level,
        reload=False,
    )
    return uvicorn.Server(config)


async def _run_standalone_worker() -> None:
    """惰性导入并运行独立 Worker，保持 ``--help`` 没有配置副作用."""
    from app.entrypoints.research_worker import run_worker

    await run_worker()


async def _run_supervised_worker() -> None:
    """运行由 FastAPI lifespan 统一拥有 tracing 的 Worker."""
    from app.entrypoints.research_worker import run_worker

    await run_worker(manage_tracing=False)


async def supervise_all(api_server: ApiServer, worker_runner: WorkerRunner = _run_supervised_worker) -> None:
    """并发运行 API 与 Worker，任一组件结束时收敛整个 Runtime.

    API 正常响应 Ctrl+C 后会返回，本函数随即取消 Worker；Worker 的 finally 会
    关闭连接池且不会把进程关闭伪装成用户取消。若 Worker 初始化或运行异常，
    Supervisor 会要求 Uvicorn 优雅停止并向进程入口传播异常，使 systemd 能够
    根据非零退出码重启，而不是留下“API 正常、任务永远 pending”的半健康服务。
    """
    api_task = asyncio.create_task(api_server.serve(), name="deep-research-api")
    worker_task = asyncio.create_task(worker_runner(), name="deep-research-worker")
    failure: BaseException | None = None

    try:
        done, _ = await asyncio.wait((api_task, worker_task), return_when=asyncio.FIRST_COMPLETED)
        if worker_task in done and not worker_task.cancelled():
            failure = worker_task.exception() or RuntimeError("Research Worker stopped unexpectedly")
        if api_task in done and not api_task.cancelled() and api_task.exception() is not None:
            failure = api_task.exception()
        elif api_task in done and not api_server.started:
            failure = RuntimeError("FastAPI stopped before startup completed")
    finally:
        # 无论谁先退出，都先阻止 API 继续接收请求，再取消仍在运行的 Worker。
        # Uvicorn 看到 should_exit 后会执行 FastAPI lifespan 的完整 shutdown。
        api_server.should_exit = True
        if not worker_task.done():
            worker_task.cancel()
        if not api_task.done():
            await api_task
        await asyncio.gather(worker_task, return_exceptions=True)

    if failure is not None:
        raise RuntimeError("DeepResearch Runtime component failed") from failure


async def run(options: RuntimeOptions) -> None:
    """按照 mode 启动对应组件."""
    if options.mode == "worker":
        await _run_standalone_worker()
        return

    api_server = create_api_server(options)
    if options.mode == "api":
        await api_server.serve()
        return

    await supervise_all(api_server)


def _event_loop_factory() -> asyncio.AbstractEventLoop:
    """为 Windows psycopg async 创建兼容的 Selector event loop."""
    if sys.platform == "win32":
        return asyncio.SelectorEventLoop(selectors.SelectSelector())
    return asyncio.new_event_loop()


def main(argv: Sequence[str] | None = None) -> int:
    """解析命令并运行统一 Runtime."""
    options = parse_options(argv)
    try:
        asyncio.run(run(options), loop_factory=_event_loop_factory)
    except KeyboardInterrupt:
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
