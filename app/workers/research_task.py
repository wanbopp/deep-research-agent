"""从 PostgreSQL 队列领取并执行可取消 DeepResearch 任务."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from datetime import timedelta
from enum import StrEnum
from typing import cast
from uuid import UUID

from langchain_core.runnables import RunnableConfig
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.agents.research.context import ResearchRuntimeContext
from app.agents.research.graph import ResearchGraph
from app.agents.research.state import ResearchState
from app.agents.research.writer import render_report_markdown
from app.core.logging import logger
from app.models import ResearchTask, ResearchTaskStatus, utc_now
from app.observability import metrics
from app.repositories import ResearchTaskRepository
from app.schemas.research import ResearchConfig, ResearchReport, ResearchStatus


class ResearchWorkerResult(StrEnum):
    """一次 worker 轮询的稳定结果."""

    IDLE = "idle"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    CLAIM_LOST = "claim_lost"


class ResearchCancellationRequested(RuntimeError):
    """表示数据库中已有用户取消意图，不等同于进程任务被系统取消."""


class ResearchRunClaimLost(RuntimeError):
    """当前 Worker 的 run_id 已失效，禁止继续写任务状态和事件."""


class ResearchTaskWorker:
    """以短事务领取任务，在事务外运行模型，并持久化节点进度."""

    def __init__(
        self,
        *,
        session_factory: async_sessionmaker[AsyncSession],
        graph: ResearchGraph,
        worker_id: str,
        cancellation_poll_seconds: float = 0.5,
        heartbeat_interval_seconds: float = 10.0,
        stale_after_seconds: float = 120.0,
    ) -> None:
        """保存共享工厂、研究图和当前 worker 身份.

        Args:
            session_factory: 每个短事务创建独立 Session 的工厂。
            graph: lifespan/worker startup 编译一次的共享研究图。
            worker_id: 运维可识别但不包含密钥的进程实例名称。
            cancellation_poll_seconds: 检查持久取消标记的间隔。
            heartbeat_interval_seconds: Graph 长节点运行时续写 worker 存活时间的间隔。
            stale_after_seconds: 认为旧 worker 已失联的心跳时长。
        """
        if cancellation_poll_seconds <= 0 or heartbeat_interval_seconds <= 0 or stale_after_seconds <= 0:
            raise ValueError("worker timing values must be greater than zero")
        if heartbeat_interval_seconds >= stale_after_seconds:
            raise ValueError("heartbeat interval must be shorter than stale timeout")
        self._session_factory = session_factory
        self._graph = graph
        self._worker_id = worker_id
        self._cancellation_poll_seconds = cancellation_poll_seconds
        self._heartbeat_interval_seconds = heartbeat_interval_seconds
        self._stale_after_seconds = stale_after_seconds

    async def run_once(self) -> ResearchWorkerResult:
        """恢复过期任务、原子领取一条任务并执行，队列为空时返回 idle."""
        async with self._session_factory() as session, session.begin():
            repository = ResearchTaskRepository(session)
            await repository.recover_stale(
                stale_before=utc_now() - timedelta(seconds=self._stale_after_seconds),
            )
            task = await repository.claim_next(worker_id=self._worker_id)
            if task is None:
                return ResearchWorkerResult.IDLE
            run_id = task.active_run_id
            if run_id is None:
                raise RuntimeError("claimed research task is missing active_run_id")
            await repository.append_event(
                task_id=task.id,
                user_id=task.user_id,
                event_type="task_started",
                payload={"run_id": str(run_id), "attempt_count": task.attempt_count},
            )

        metrics.research_worker_inflight.inc()
        try:
            try:
                result = await self._execute_claimed(task, run_id=run_id)
            except ResearchRunClaimLost:
                # claim 也可能恰好在最终写入或错误收尾前失效。统一在最外层转换
                # 为有限结果，确保旧 Worker 安静退出且 Gauge 正常归还。
                logger.warning("research_run_claim_lost", research_id=str(task.id), run_id=str(run_id))
                result = ResearchWorkerResult.CLAIM_LOST
            metrics.observe_research_task(outcome=result.value)
            return result
        finally:
            # Worker shutdown、finalize 失败和普通终态都必须归还 Gauge，避免进程内
            # 指标永久显示幽灵任务。任务真实恢复仍由数据库 lease 决定。
            metrics.research_worker_inflight.dec()

    async def _execute_claimed(self, task: ResearchTask, *, run_id: UUID) -> ResearchWorkerResult:
        """在领取事务之外执行一条任务，并返回有限的处理结果."""
        try:
            final_state = await self._run_graph_until_done_or_cancelled(task, run_id=run_id)
        except ResearchCancellationRequested:
            await self._finalize_cancelled(task.id, run_id=run_id)
            return ResearchWorkerResult.CANCELLED
        except ResearchRunClaimLost:
            # 新 Worker 已经恢复并领取了新 run。旧执行只能退出，不能把 claim lost
            # 伪装成业务失败或用户取消，也不能修改新 run 的状态。
            logger.warning("research_run_claim_lost", research_id=str(task.id), run_id=str(run_id))
            return ResearchWorkerResult.CLAIM_LOST
        except asyncio.CancelledError:
            # 服务关闭或运维取消 worker 时不能伪造“用户取消”。保留 running 状态，
            # 新 worker 会在心跳过期后按重试预算恢复任务。
            raise
        except Exception as error:
            logger.exception(
                "research_task_failed",
                research_id=str(task.id),
                error_type=type(error).__name__,
            )
            await self._finalize_failed(task.id, run_id=run_id, error_type=type(error).__name__)
            return ResearchWorkerResult.FAILED

        report_data = final_state.get("report")
        if report_data is None or final_state["status"] != ResearchStatus.COMPLETED.value:
            await self._finalize_failed(task.id, run_id=run_id, error_type="IncompleteResearchState")
            return ResearchWorkerResult.FAILED
        report = ResearchReport.model_validate(report_data)

        async with self._session_factory() as session, session.begin():
            repository = ResearchTaskRepository(session)
            current = await repository.get_claim_for_update(
                task_id=task.id,
                run_id=run_id,
                worker_id=self._worker_id,
            )
            if current is None:
                raise ResearchRunClaimLost
            await repository.finalize(
                current,
                run_id=run_id,
                status=ResearchTaskStatus.COMPLETED,
                report_json=report.model_dump(mode="json"),
                markdown_report=render_report_markdown(report),
            )
            await repository.append_event(
                task_id=current.id,
                user_id=current.user_id,
                event_type="task_completed",
                payload={"run_id": str(run_id), "status": ResearchTaskStatus.COMPLETED.value},
            )
        return ResearchWorkerResult.COMPLETED

    async def _run_graph_until_done_or_cancelled(self, task: ResearchTask, *, run_id: UUID) -> ResearchState:
        """竞速执行图与持久取消轮询；取消胜出时中止当前模型协程."""
        graph_task = asyncio.create_task(self._consume_graph(task, run_id=run_id))
        cancel_task = asyncio.create_task(self._wait_for_cancellation(task.id, run_id=run_id))
        done, _ = await asyncio.wait((graph_task, cancel_task), return_when=asyncio.FIRST_COMPLETED)
        if cancel_task in done:
            try:
                cancellation_requested = cancel_task.result()
            except BaseException:
                graph_task.cancel()
                await asyncio.gather(graph_task, return_exceptions=True)
                raise
            if cancellation_requested:
                graph_task.cancel()
                await asyncio.gather(graph_task, return_exceptions=True)
                raise ResearchCancellationRequested
        cancel_task.cancel()
        await asyncio.gather(cancel_task, return_exceptions=True)
        return await graph_task

    async def _consume_graph(self, task: ResearchTask, *, run_id: UUID) -> ResearchState:
        """流式执行节点，节点完成后立即把小型进度事件写入数据库."""
        config = ResearchConfig.model_validate(task.config_json)
        initial: ResearchState = {
            "topic": task.topic,
            # Graph checkpoint 是长期数据边界。只交给它 JSON 数据；每个节点会在
            # 使用前恢复为 ResearchConfig/ResearchPlan 等严格模型。
            "config": config.model_dump(mode="json"),
            "status": ResearchStatus.PLANNING.value,
            "current_iteration": 0,
            "evidence": (),
            "retrieval_failures": (),
        }
        runnable_config: RunnableConfig = {"configurable": {"thread_id": task.checkpoint_thread_id}}
        context = ResearchRuntimeContext(user_id=task.user_id, research_id=task.id)
        async with asyncio.timeout(config.timeout_seconds):
            async for update in self._graph.astream(
                initial,
                config=runnable_config,
                context=context,
                stream_mode="updates",
            ):
                if isinstance(update, Mapping):
                    for node_name, node_update in update.items():
                        await self._record_node_event(task, run_id, str(node_name), node_update)

        snapshot = await self._graph.aget_state(runnable_config)
        return cast(ResearchState, snapshot.values)

    async def _record_node_event(
        self,
        task: ResearchTask,
        run_id: UUID,
        node_name: str,
        update: object,
    ) -> None:
        """只保存节点名、状态和计数，不把证据正文复制进事件表."""
        payload: dict[str, object] = {"run_id": str(run_id), "node": node_name}
        if isinstance(update, Mapping):
            status = update.get("status")
            if isinstance(status, StrEnum):
                payload["status"] = status.value
            evidence = update.get("evidence")
            if isinstance(evidence, tuple):
                payload["evidence_count"] = len(evidence)
        async with self._session_factory() as session, session.begin():
            repository = ResearchTaskRepository(session)
            current = await repository.get_claim_for_update(
                task_id=task.id,
                run_id=run_id,
                worker_id=self._worker_id,
            )
            if current is None:
                raise ResearchRunClaimLost
            await repository.heartbeat(current, run_id=run_id)
            await repository.append_event(
                task_id=task.id,
                user_id=task.user_id,
                event_type=f"node_{node_name}_completed",
                payload=payload,
            )

    async def _wait_for_cancellation(self, task_id: UUID, *, run_id: UUID) -> bool:
        """轮询取消标记，并在长节点运行期间独立续写 worker 心跳."""
        loop = asyncio.get_running_loop()
        next_heartbeat_at = loop.time()
        while True:
            await asyncio.sleep(self._cancellation_poll_seconds)
            async with self._session_factory() as session, session.begin():
                repository = ResearchTaskRepository(session)
                task = await repository.get_claim_for_update(
                    task_id=task_id,
                    run_id=run_id,
                    worker_id=self._worker_id,
                )
                if task is None:
                    raise ResearchRunClaimLost
                if task.status in {ResearchTaskStatus.CANCELLING, ResearchTaskStatus.CANCELLED}:
                    return True
                if loop.time() >= next_heartbeat_at:
                    # 模型或网页调用可能长时间没有节点完成事件；独立心跳避免另一
                    # worker 把仍在执行的任务误判为失联并重复领取。
                    await repository.heartbeat(task, run_id=run_id)
                    next_heartbeat_at = loop.time() + self._heartbeat_interval_seconds

    async def _finalize_cancelled(self, task_id: UUID, *, run_id: UUID) -> None:
        """把取消中的任务推进到 cancelled 并写最终事件."""
        async with self._session_factory() as session, session.begin():
            repository = ResearchTaskRepository(session)
            task = await repository.get_claim_for_update(
                task_id=task_id,
                run_id=run_id,
                worker_id=self._worker_id,
            )
            if task is None:
                raise ResearchRunClaimLost
            await repository.finalize(
                task,
                run_id=run_id,
                status=ResearchTaskStatus.CANCELLED,
                error_code="CANCELLED_BY_USER",
            )
            await repository.append_event(
                task_id=task.id,
                user_id=task.user_id,
                event_type="task_cancelled",
                payload={"run_id": str(run_id), "status": ResearchTaskStatus.CANCELLED.value},
            )

    async def _finalize_failed(self, task_id: UUID, *, run_id: UUID, error_type: str) -> None:
        """保存安全错误类型；不持久化异常文本或 provider 响应."""
        async with self._session_factory() as session, session.begin():
            repository = ResearchTaskRepository(session)
            task = await repository.get_claim_for_update(
                task_id=task_id,
                run_id=run_id,
                worker_id=self._worker_id,
            )
            if task is None:
                raise ResearchRunClaimLost
            await repository.finalize(
                task,
                run_id=run_id,
                status=ResearchTaskStatus.FAILED,
                error_code=error_type[:128],
            )
            await repository.append_event(
                task_id=task.id,
                user_id=task.user_id,
                event_type="task_failed",
                payload={"run_id": str(run_id), "error_code": error_type[:128]},
            )


__all__ = [
    "ResearchCancellationRequested",
    "ResearchRunClaimLost",
    "ResearchTaskWorker",
    "ResearchWorkerResult",
]
