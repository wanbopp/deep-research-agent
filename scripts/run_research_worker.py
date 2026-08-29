"""持续领取持久 DeepResearch 任务的独立 worker 入口."""

import asyncio
import json
import selectors
from uuid import uuid4

from app.agents.research.runtime import create_research_runtime
from app.core.config import settings
from app.graphrag.runtime import create_graphrag_runtime
from app.infrastructure.factory import create_application_resources
from app.infrastructure.file_storage import LocalFileStorage
from app.rag.runtime import create_rag_runtime
from app.workers.research_task import ResearchTaskWorker, ResearchWorkerResult


async def run_worker(*, idle_sleep_seconds: float = 1.0) -> None:
    """运行常驻 worker，并在退出时按资源所有权关闭全部客户端.

    PostgreSQL 是任务事实来源；本进程不需要与创建任务的 API 进程共享内存。
    多个 worker 同时运行时，数据库行锁会把不同任务分配给不同进程。
    """
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
        worker = ResearchTaskWorker(
            session_factory=resources.orm_session_factory,
            graph=graph,
            worker_id=f"research-{uuid4().hex[:12]}",
        )
        while True:
            result = await worker.run_once()
            if result is ResearchWorkerResult.IDLE:
                await asyncio.sleep(idle_sleep_seconds)
            else:
                print(json.dumps({"research_worker_result": result.value}))
    finally:
        await resources.redis_client.aclose()
        await resources.neo4j_driver.close()
        await resources.orm_engine.dispose()
        await resources.postgres_pool.close()


if __name__ == "__main__":
    try:
        asyncio.run(
            run_worker(),
            loop_factory=lambda: asyncio.SelectorEventLoop(selectors.SelectSelector()),
        )
    except KeyboardInterrupt:
        pass
