"""使用真实模型、公开网页和持久任务执行 Phase 7 端到端 smoke.

本脚本不使用 fake LLM 或 fake retriever。它复用开发环境已有用户身份，只创建一
条带随机幂等键的临时研究任务；最终无论成功失败，都会删除任务、事件和对应
checkpoint。输出只包含计数、状态和布尔验收结果，不输出用户资料、网页正文、
模型原文、连接串或 API key。
"""

import asyncio
import json
import selectors
from uuid import UUID, uuid4

from sqlmodel import select

from app.agents.research.runtime import create_research_runtime
from app.core.config import settings
from app.graphrag.runtime import create_graphrag_runtime
from app.infrastructure.factory import create_application_resources
from app.infrastructure.file_storage import LocalFileStorage
from app.models import ResearchTask, User
from app.rag.runtime import create_rag_runtime
from app.schemas.research import ResearchConfig, ResearchCreateRequest
from app.services.research import ResearchService
from app.workers.research_task import ResearchTaskWorker, ResearchWorkerResult


async def run_smoke() -> dict[str, object]:
    """创建、领取并完成一条真实任务，再验证事件、报告和引用."""
    resources = create_application_resources(settings)
    await resources.postgres_pool.open()
    await resources.checkpointer.setup()
    task_id: UUID | None = None
    checkpoint_thread_id: str | None = None
    try:
        graphrag = create_graphrag_runtime(config=settings, neo4j_driver=resources.neo4j_driver)
        await graphrag.repository.setup_schema()
        _, hybrid = create_rag_runtime(
            config=settings,
            session_factory=resources.orm_session_factory,
            storage=LocalFileStorage(settings.KNOWLEDGE_STORAGE_ROOT),
            worker_id="phase7-smoke-unused-index",
            graphrag_runtime=graphrag,
        )
        graph = create_research_runtime(
            config=settings,
            hybrid_retriever=hybrid,
            graphrag_runtime=graphrag,
            checkpointer=resources.checkpointer,
        )

        # user_id 来自数据库可信记录，不放入请求正文。Smoke 不创建或输出账户信息，
        # 只借用一个 owner 验证 Repository/worker/Graph 的身份传递链。
        async with resources.orm_session_factory() as session, session.begin():
            user_id = (await session.execute(select(User.id).order_by(User.created_at).limit(1))).scalar_one()

        request = ResearchCreateRequest(
            topic=(
                "As of August 2026, identify the latest stable Python release from current public web sources "
                "and explain the result briefly."
            ),
            config=ResearchConfig(
                max_steps=1,
                max_iterations=1,
                max_evidence_per_step=5,
                max_total_evidence=8,
                timeout_seconds=240,
                require_independent_sources=1,
            ),
        )
        idempotency_key = f"phase7-smoke-{uuid4().hex}"

        async def create_same_task() -> tuple[UUID, bool]:
            """用独立 Session 模拟两个真正同时到达的 HTTP 创建请求."""
            async with resources.orm_session_factory() as session:
                task, created = await ResearchService(session).create(
                    user_id=user_id,
                    request=request,
                    idempotency_key=idempotency_key,
                )
                return task.research_id, created

        # 两个请求必须得到同一个 task_id，并且只能有一个请求声称创建成功。
        # 这验证的是 PostgreSQL 事务锁与唯一约束，不是进程内 Python 锁。
        create_results = await asyncio.gather(create_same_task(), create_same_task())
        created_ids = {research_id for research_id, _ in create_results}
        created_count = sum(1 for _, created in create_results if created)
        idempotency_ok = len(created_ids) == 1 and created_count == 1
        task_id = create_results[0][0]

        async with resources.orm_session_factory() as session, session.begin():
            stored = await session.get(ResearchTask, task_id)
            if stored is None:
                raise RuntimeError("phase7 smoke task disappeared")
            checkpoint_thread_id = stored.checkpoint_thread_id

        worker = ResearchTaskWorker(
            session_factory=resources.orm_session_factory,
            graph=graph,
            worker_id=f"phase7-smoke-{uuid4().hex[:10]}",
        )
        worker_result = await worker.run_once()

        async with resources.orm_session_factory() as session:
            service = ResearchService(session)
            completed = await service.get(task_id=task_id, user_id=user_id)
            events = await service.events(task_id=task_id, user_id=user_id)

        report = completed.report
        event_names = tuple(event.event for event in events)
        required_events = {
            "task_created",
            "task_started",
            "node_completed",
            "task_completed",
        }
        completed_nodes = {
            node
            for event in events
            if event.event == "node_completed" and isinstance((node := getattr(event.payload, "node", None)), str)
        }
        required_nodes = {"planner", "retrieve", "validate", "write"}
        citation_count = len(report.citations) if report is not None else 0
        citation_ids = {item.citation_id for item in report.citations} if report is not None else set()
        used_ids = (
            {citation_id for section in report.sections for citation_id in section.citation_ids}
            if report is not None
            else set()
        )
        ok = (
            idempotency_ok
            and worker_result is ResearchWorkerResult.COMPLETED
            and completed.status == "completed"
            and report is not None
            and citation_count >= 1
            and bool(used_ids)
            and used_ids <= citation_ids
            and required_events <= set(event_names)
            and required_nodes <= completed_nodes
            and completed.markdown_report is not None
        )
        return {
            "ok": ok,
            "real_model": settings.DEFAULT_LLM_MODEL,
            "concurrent_idempotency_ok": idempotency_ok,
            "worker_result": worker_result.value,
            "task_status": completed.status,
            "event_count": len(events),
            "required_events_present": required_events <= set(event_names),
            "required_nodes_present": required_nodes <= completed_nodes,
            "citation_count": citation_count,
            "citations_valid": bool(used_ids) and used_ids <= citation_ids,
            "markdown_available": completed.markdown_report is not None,
        }
    finally:
        # 先删 checkpoint，再删业务任务。research_events 由数据库外键级联删除；
        # 清理失败会使 smoke 失败，而不是静默留下测试数据。
        if checkpoint_thread_id is not None:
            await resources.checkpointer.adelete_thread(checkpoint_thread_id)
        if task_id is not None:
            async with resources.orm_session_factory() as session, session.begin():
                task = await session.get(ResearchTask, task_id)
                if task is not None:
                    await session.delete(task)
        await resources.redis_client.aclose()
        await resources.neo4j_driver.close()
        await resources.orm_engine.dispose()
        await resources.postgres_pool.close()


if __name__ == "__main__":
    summary = asyncio.run(
        run_smoke(),
        loop_factory=lambda: asyncio.SelectorEventLoop(selectors.SelectSelector()),
    )
    print(json.dumps(summary))
    raise SystemExit(0 if summary["ok"] else 1)
