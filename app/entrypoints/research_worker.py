"""持续运行的 Research Worker 入口实现."""

import asyncio
import json
import selectors
import sys
from uuid import uuid4

from app.agents.research.runtime import create_research_runtime
from app.core.config import settings
from app.graphrag.runtime import create_graphrag_runtime
from app.infrastructure.factory import create_application_resources
from app.infrastructure.file_storage import LocalFileStorage
from app.observability import build_trace_sink, tracing
from app.rag.runtime import create_rag_runtime
from app.workers.research_task import ResearchTaskWorker, ResearchWorkerResult


async def run_worker(*, idle_sleep_seconds: float = 1.0, manage_tracing: bool = True) -> None:
    """持续领取持久 Research Task，并在退出时关闭进程拥有的资源.

    这个函数既可以由独立 ``worker`` 模式调用，也可以由统一 Runtime 的
    Supervisor 调用。两种模式始终经过同一个数据库领取边界，不会退化成仅存在
    于 API 内存中的后台任务。

    Args:
        idle_sleep_seconds: 队列暂时为空时再次轮询前的等待秒数。
        manage_tracing: 是否由本函数创建并关闭进程级 tracing sink。独立 Worker
            使用 True；``all`` 模式与 API 共享进程，由 FastAPI lifespan 统一拥有。

    Raises:
        ValueError: 轮询间隔不是正数。
        Exception: Worker 初始化失败时向 Supervisor 传播，由统一入口停止整个
            ``all`` 模式，避免出现 API 可用但队列无人消费的半健康状态。
    """
    if idle_sleep_seconds <= 0:
        raise ValueError("idle_sleep_seconds must be greater than zero")

    if manage_tracing:
        tracing.configure(build_trace_sink(settings))
    resources = create_application_resources(settings)
    await resources.postgres_pool.open()
    await resources.checkpointer.setup()
    try:
        graphrag = create_graphrag_runtime(config=settings, neo4j_driver=resources.neo4j_driver)
        await graphrag.repository.setup_schema()
        _, hybrid = create_rag_runtime(
            config=settings,
            session_factory=resources.orm_session_factory,
            storage=LocalFileStorage(settings.KNOWLEDGE_STORAGE_ROOT),
            worker_id="research-worker-unused-index",
            graphrag_runtime=graphrag,
        )
        graph = create_research_runtime(
            config=settings,
            hybrid_retriever=hybrid,
            graphrag_runtime=graphrag,
            checkpointer=resources.checkpointer,
        )
        worker_id = f"research-{uuid4().hex[:12]}"
        worker = ResearchTaskWorker(
            session_factory=resources.orm_session_factory,
            graph=graph,
            worker_id=worker_id,
        )
        # 输出固定字段而不是数据库连接或任务内容，便于 Supervisor/systemd 判断
        # 队列消费者已经完成初始化，同时避免凭据和研究主题进入启动日志。
        print(json.dumps({"research_worker_status": "ready", "worker_id": worker_id}), flush=True)
        while True:
            result = await worker.run_once()
            if result is ResearchWorkerResult.IDLE:
                await asyncio.sleep(idle_sleep_seconds)
            else:
                print(json.dumps({"research_worker_result": result.value}), flush=True)
    finally:
        if manage_tracing:
            tracing.close()
        await resources.redis_client.aclose()
        await resources.neo4j_driver.close()
        await resources.orm_engine.dispose()
        await resources.postgres_pool.close()


def _event_loop_factory() -> asyncio.AbstractEventLoop:
    """创建当前平台使用的事件循环.

    psycopg async 在 Windows 上不能使用默认 ProactorEventLoop，因此所有正式
    Worker 入口统一选择 SelectorEventLoop；其他平台保留 asyncio 默认实现。
    """
    if sys.platform == "win32":
        return asyncio.SelectorEventLoop(selectors.SelectSelector())
    return asyncio.new_event_loop()


def main() -> int:
    """运行独立 Worker 模式，并把 Ctrl+C 收敛为正常退出."""
    try:
        asyncio.run(run_worker(), loop_factory=_event_loop_factory)
    except KeyboardInterrupt:
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
