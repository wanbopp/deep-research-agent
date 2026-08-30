"""知识库 Index Worker 的一次性与常驻调度入口."""

import asyncio
import json
import selectors
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from uuid import uuid4

from neo4j import AsyncGraphDatabase

from app.core.config import settings
from app.core.logging import component_context, logger
from app.graphrag.runtime import create_graphrag_runtime
from app.infrastructure.database import create_orm_runtime
from app.infrastructure.file_storage import LocalFileStorage
from app.rag.runtime import create_rag_runtime
from app.services.index_worker import IndexWorker, IndexWorkerResult


@asynccontextmanager
async def _index_worker_runtime() -> AsyncIterator[tuple[IndexWorker, str]]:
    """创建一套可跨多次轮询复用的 Index Worker 资源.

    ORM Engine、Neo4j Driver、Embedding/RAG 对象只在调度器启动时创建一次，退出
    时按所有权统一关闭。队列为空时不会重复建立连接或重新构造模型客户端。
    """
    engine, sessions = create_orm_runtime(settings)
    neo4j_driver = AsyncGraphDatabase.driver(
        settings.NEO4J_URI,
        auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD),
    )
    try:
        graphrag_runtime = create_graphrag_runtime(
            config=settings,
            neo4j_driver=neo4j_driver,
        )
        await graphrag_runtime.repository.setup_schema()
        worker_id = f"index-{uuid4().hex[:12]}"
        worker, _ = create_rag_runtime(
            config=settings,
            session_factory=sessions,
            storage=LocalFileStorage(settings.KNOWLEDGE_STORAGE_ROOT),
            worker_id=worker_id,
            graphrag_runtime=graphrag_runtime,
        )
        yield worker, worker_id
    finally:
        await neo4j_driver.close()
        await engine.dispose()


async def run_until_idle() -> dict[str, object]:
    """处理当前所有可领取 IndexJob，队列暂时为空后返回摘要."""
    completed = failed = stale = 0
    async with _index_worker_runtime() as (worker, worker_id):
        logger.info("index_worker_ready", worker_id=worker_id, mode="until_idle")
        while True:
            result = await worker.run_once()
            if result is IndexWorkerResult.IDLE:
                break
            if result is IndexWorkerResult.COMPLETED:
                completed += 1
            elif result is IndexWorkerResult.FAILED:
                failed += 1
            else:
                stale += 1
            logger.info("index_worker_result", result=result.value)
    return {"ok": failed == 0, "completed": completed, "failed": failed, "stale": stale}


async def run_scheduler(*, idle_sleep_seconds: float | None = None) -> None:
    """常驻领取 IndexJob，队列为空时按固定间隔等待.

    Args:
        idle_sleep_seconds: 空队列轮询间隔。None 使用 Settings 中的统一配置。

    Raises:
        ValueError: 轮询间隔不是正数。
        Exception: 基础设施初始化或不可预期的 Worker 错误向 Supervisor 传播，
            使 ``all`` 模式失败退出而不是静默失去索引消费者。
    """
    interval = settings.KNOWLEDGE_INDEX_POLL_INTERVAL_SECONDS if idle_sleep_seconds is None else idle_sleep_seconds
    if interval <= 0:
        raise ValueError("idle_sleep_seconds must be greater than zero")

    async with _index_worker_runtime() as (worker, worker_id):
        logger.info(
            "index_worker_ready",
            worker_id=worker_id,
            mode="scheduler",
            idle_sleep_seconds=interval,
        )
        while True:
            result = await worker.run_once()
            if result is IndexWorkerResult.IDLE:
                await asyncio.sleep(interval)
                continue
            logger.info("index_worker_result", result=result.value)


def _event_loop_factory() -> asyncio.AbstractEventLoop:
    """创建与 Windows psycopg async 兼容的事件循环."""
    if sys.platform == "win32":
        return asyncio.SelectorEventLoop(selectors.SelectSelector())
    return asyncio.new_event_loop()


def main_until_idle() -> int:
    """运行兼容旧脚本的一次性索引入口."""
    with component_context("index-worker"):
        summary = asyncio.run(run_until_idle(), loop_factory=_event_loop_factory)
    print(json.dumps(summary), flush=True)
    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main_until_idle())
