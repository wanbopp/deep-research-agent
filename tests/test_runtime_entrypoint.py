"""统一 Runtime CLI 与组件监督边界测试."""

import asyncio

import pytest

from app.entrypoints.runtime import parse_options, supervise_all


class _WaitingApiServer:
    """等待 Supervisor 设置 should_exit 的测试 API Server."""

    def __init__(self) -> None:
        self.should_exit = False
        self.started = False
        self.startup_entered = asyncio.Event()

    async def serve(self) -> None:
        """模拟一个已经启动并持续服务的 API."""
        self.started = True
        self.startup_entered.set()
        while not self.should_exit:
            await asyncio.sleep(0)


class _CompletingApiServer:
    """模拟收到 Ctrl+C 后正常返回的 API Server."""

    def __init__(self) -> None:
        self.should_exit = False
        self.started = True

    async def serve(self) -> None:
        """让出一次调度机会，保证 Worker 已进入运行状态后结束."""
        await asyncio.sleep(0)


def test_runtime_options_default_to_all(monkeypatch: pytest.MonkeyPatch) -> None:
    """无命令行和环境变量时必须启动完整 Web Runtime."""
    for name in (
        "DEEP_RESEARCH_RUNTIME_MODE",
        "DEEP_RESEARCH_RUNTIME_HOST",
        "DEEP_RESEARCH_RUNTIME_PORT",
        "DEEP_RESEARCH_RUNTIME_LOG_LEVEL",
    ):
        monkeypatch.delenv(name, raising=False)

    options = parse_options([])

    assert options.mode == "all"
    assert options.host == "127.0.0.1"
    assert options.port == 8000
    assert options.log_level == "info"


@pytest.mark.parametrize("mode", ["all", "api", "worker"])
def test_runtime_options_accept_each_explicit_mode(mode: str) -> None:
    """保留三种模式作为统一代码库的运维入口."""
    assert parse_options(["--mode", mode]).mode == mode


def test_runtime_options_reject_invalid_port() -> None:
    """端口越界时应在创建 Uvicorn Server 前失败."""
    with pytest.raises(ValueError, match="port"):
        parse_options(["--port", "70000"])


def test_supervisor_stops_api_when_worker_fails() -> None:
    """Worker 失败必须终止 all 模式，不能留下假健康 API."""

    async def scenario() -> None:
        api_server = _WaitingApiServer()

        async def failing_worker() -> None:
            await api_server.startup_entered.wait()
            raise ValueError("worker startup failed")

        with pytest.raises(RuntimeError, match="component failed") as captured:
            await supervise_all(api_server, failing_worker)

        assert api_server.should_exit is True
        assert isinstance(captured.value.__cause__, ValueError)

    asyncio.run(scenario())


def test_supervisor_fails_when_api_never_started() -> None:
    """Uvicorn lifespan 失败不能被误判为一次正常 Ctrl+C 退出."""

    class StartupFailedApiServer:
        should_exit = False
        started = False

        async def serve(self) -> None:
            return

    async def scenario() -> None:
        worker_cleaned = asyncio.Event()

        async def waiting_worker() -> None:
            try:
                await asyncio.Event().wait()
            finally:
                worker_cleaned.set()

        with pytest.raises(RuntimeError, match="component failed") as captured:
            await supervise_all(StartupFailedApiServer(), waiting_worker)

        assert isinstance(captured.value.__cause__, RuntimeError)
        assert worker_cleaned.is_set()

    asyncio.run(scenario())


def test_supervisor_cancels_worker_after_api_stops() -> None:
    """API 正常停止后，Worker 应收到取消并执行清理分支."""

    async def scenario() -> None:
        api_server = _CompletingApiServer()
        worker_started = asyncio.Event()
        worker_cleaned = asyncio.Event()

        async def waiting_worker() -> None:
            worker_started.set()
            try:
                await asyncio.Event().wait()
            finally:
                worker_cleaned.set()

        await supervise_all(api_server, waiting_worker)

        assert worker_started.is_set()
        assert worker_cleaned.is_set()
        assert api_server.should_exit is True

    asyncio.run(scenario())
