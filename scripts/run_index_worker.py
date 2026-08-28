"""处理当前知识库中所有可领取索引任务的一次性 worker 入口."""

import asyncio
import json
import selectors
from uuid import uuid4

from app.core.config import settings
from app.infrastructure.database import create_orm_runtime
from app.infrastructure.file_storage import LocalFileStorage
from app.rag.runtime import create_rag_runtime
from app.services.index_worker import IndexWorkerResult


async def run_until_idle() -> dict[str, object]:
    """依次领取任务直到队列暂时为空，并始终关闭 ORM Engine.

    单个进程每次只处理一个 job；多进程扩容时 PostgreSQL ``SKIP LOCKED`` 会把
    不同 job 分配给不同 worker。该脚本不常驻轮询，生产队列调度将在 Lab 29
    接入；当前可由运维调度器重复启动。
    """
    engine, sessions = create_orm_runtime(settings)
    worker_id = f"manual-{uuid4().hex[:12]}"
    worker, _ = create_rag_runtime(
        config=settings,
        session_factory=sessions,
        storage=LocalFileStorage(settings.KNOWLEDGE_STORAGE_ROOT),
        worker_id=worker_id,
    )
    completed = failed = stale = 0
    try:
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
    finally:
        await engine.dispose()
    return {"ok": failed == 0, "completed": completed, "failed": failed, "stale": stale}


if __name__ == "__main__":
    summary = asyncio.run(
        run_until_idle(),
        loop_factory=lambda: asyncio.SelectorEventLoop(selectors.SelectSelector()),
    )
    print(json.dumps(summary))
    raise SystemExit(0 if summary["ok"] else 1)
