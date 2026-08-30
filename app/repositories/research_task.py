"""持久研究任务的领取、状态转换和事件重放数据边界."""

from datetime import datetime
from typing import Any, cast
from uuid import UUID, uuid4

from sqlalchemy import text
from sqlmodel import select

from app.models import ResearchEvent, ResearchTask, ResearchTaskStatus, utc_now
from app.repositories.base import RepositoryBase


class ResearchTaskRepository(RepositoryBase):
    """封装研究任务所有权查询和跨 worker 原子领取."""

    async def lock_idempotency_key(self, *, user_id: UUID, key: str) -> None:
        """在当前事务内串行化同一用户、同一幂等键的创建请求.

        唯一约束只能阻止重复数据，却不能让“输掉竞争”的请求自动读回原任务。
        PostgreSQL 事务级 advisory lock 把用户 ID 与幂等键映射成数据库锁：相同
        请求依次执行，不同用户或不同键仍可并发。事务结束后数据库自动释放锁，
        因此异常路径不需要手工清理，也不会遗留永久锁。
        """
        await self._session.execute(
            text("SELECT pg_advisory_xact_lock(hashtextextended(:lock_key, 0))"),
            {"lock_key": f"research-create:{user_id.hex}:{key}"},
        )

    async def create(
        self,
        *,
        user_id: UUID,
        topic: str,
        chat_session_id: UUID | None = None,
        status: ResearchTaskStatus = ResearchTaskStatus.PENDING,
        config_json: dict[str, object] | None = None,
        idempotency_key: str | None = None,
        max_attempts: int = 2,
    ) -> ResearchTask:
        """在当前事务创建任务；事务提交仍由 service 统一负责."""
        task = ResearchTask(
            user_id=user_id,
            chat_session_id=chat_session_id,
            topic=topic,
            status=status,
            config_json=config_json or {},
            idempotency_key=idempotency_key or f"legacy-{uuid4().hex}",
            checkpoint_thread_id="pending",
            max_attempts=max_attempts,
        )
        # 应用侧 UUID 在 flush 前已经存在，可以用它建立不含用户输入的内部
        # checkpoint 名称。用户隔离仍依赖可信 user_id + task.id 双重组成。
        task.checkpoint_thread_id = f"research:{user_id.hex}:{task.id.hex}"
        return await self._persist(task, resource="research_task")

    async def get_by_id(self, task_id: UUID, *, user_id: UUID) -> ResearchTask | None:
        """同时按任务 ID 和用户 ID 查询，跨用户与不存在均返回 None."""
        statement = select(ResearchTask).where(
            ResearchTask.id == task_id,
            ResearchTask.user_id == user_id,
        )
        result = await self._session.execute(statement)
        return result.scalar_one_or_none()

    async def get_by_idempotency_key(self, *, user_id: UUID, key: str) -> ResearchTask | None:
        """读取同一用户已经用相同幂等键创建的任务."""
        result = await self._session.execute(
            select(ResearchTask).where(
                ResearchTask.user_id == user_id,
                ResearchTask.idempotency_key == key,
            )
        )
        return result.scalar_one_or_none()

    async def list_by_user(self, user_id: UUID) -> tuple[ResearchTask, ...]:
        """按最新创建时间列出一个用户拥有的任务."""
        columns = cast(Any, ResearchTask).__table__.c
        statement = (
            select(ResearchTask)
            .where(ResearchTask.user_id == user_id)
            .order_by(columns.created_at.desc(), columns.id.desc())
        )
        result = await self._session.execute(statement)
        return tuple(result.scalars().all())

    async def claim_next(self, *, worker_id: str) -> ResearchTask | None:
        """使用行锁领取一个任务，多个 worker 不会取得同一行.

        ``SKIP LOCKED`` 表示另一个 worker 已锁定首条任务时，本 worker 继续寻找
        下一条，而不是排队等待。状态更新与 task_started 事件应在同一事务提交。
        """
        columns = cast(Any, ResearchTask).__table__.c
        statement = (
            select(ResearchTask)
            .where(columns.status.in_((ResearchTaskStatus.PENDING, ResearchTaskStatus.RETRYING)))
            .where(columns.attempt_count < columns.max_attempts)
            .order_by(columns.created_at, columns.id)
            .with_for_update(skip_locked=True)
            .limit(1)
        )
        result = await self._session.execute(statement)
        task = result.scalar_one_or_none()
        if task is None:
            return None
        now = utc_now()
        task.status = ResearchTaskStatus.RUNNING
        task.worker_id = worker_id
        task.active_run_id = uuid4()
        task.lifecycle_version += 1
        task.attempt_count += 1
        task.started_at = task.started_at or now
        task.heartbeat_at = now
        task.updated_at = now
        await self._session.flush()
        return task

    async def append_event(
        self,
        *,
        task_id: UUID,
        user_id: UUID,
        event_type: str,
        payload: dict[str, object] | None = None,
    ) -> ResearchEvent:
        """在任务状态事务中追加一个可重放进度事件."""
        event = ResearchEvent(
            research_task_id=task_id,
            user_id=user_id,
            event_type=event_type,
            payload_json=payload or {},
        )
        return await self._persist(event, resource="research_event")

    async def list_events(
        self,
        *,
        task_id: UUID,
        user_id: UUID,
        after_id: int = 0,
        limit: int = 100,
    ) -> tuple[ResearchEvent, ...]:
        """按自增 ID 返回断线后尚未看过的事件."""
        columns = cast(Any, ResearchEvent).__table__.c
        statement = (
            select(ResearchEvent)
            .where(
                ResearchEvent.research_task_id == task_id,
                ResearchEvent.user_id == user_id,
                columns.id > after_id,
            )
            .order_by(columns.id)
            .limit(limit)
        )
        result = await self._session.execute(statement)
        return tuple(result.scalars().all())

    async def lock_owned(self, *, task_id: UUID, user_id: UUID) -> ResearchTask | None:
        """锁定当前用户任务，供取消和重试执行条件状态转换."""
        result = await self._session.execute(
            select(ResearchTask).where(ResearchTask.id == task_id, ResearchTask.user_id == user_id).with_for_update()
        )
        return result.scalar_one_or_none()

    async def get_worker_view(self, task_id: UUID) -> ResearchTask | None:
        """供已领取 worker 检查取消标记；调用方必须已持有可信 task ID."""
        return await self._session.get(ResearchTask, task_id)

    async def get_claim_for_update(
        self,
        *,
        task_id: UUID,
        run_id: UUID,
        worker_id: str,
    ) -> ResearchTask | None:
        """锁定仍属于指定 run 的任务；旧 Worker 的 token 不会命中新领取."""
        columns = cast(Any, ResearchTask).__table__.c
        result = await self._session.execute(
            select(ResearchTask)
            .where(
                ResearchTask.id == task_id,
                ResearchTask.active_run_id == run_id,
                ResearchTask.worker_id == worker_id,
                columns.status.in_((ResearchTaskStatus.RUNNING, ResearchTaskStatus.CANCELLING)),
            )
            .with_for_update()
        )
        return result.scalar_one_or_none()

    async def heartbeat(self, task: ResearchTask, *, run_id: UUID) -> None:
        """在调用方已经用 run_id 锁定当前 claim 后续写心跳."""
        if task.active_run_id != run_id or task.worker_id is None:
            raise RuntimeError("heartbeat requires the current research run claim")
        task.heartbeat_at = utc_now()
        task.updated_at = task.heartbeat_at
        task.lifecycle_version += 1
        await self._session.flush()

    async def finalize(
        self,
        task: ResearchTask,
        *,
        run_id: UUID,
        status: ResearchTaskStatus,
        report_json: dict[str, object] | None = None,
        markdown_report: str | None = None,
        error_code: str | None = None,
    ) -> None:
        """把运行中任务转换为一个稳定终态，并清除 worker 租用信息."""
        if task.active_run_id != run_id:
            raise RuntimeError("finalize requires the current research run claim")
        if status not in {
            ResearchTaskStatus.COMPLETED,
            ResearchTaskStatus.FAILED,
            ResearchTaskStatus.CANCELLED,
        }:
            raise ValueError("finalize requires a terminal status")
        now = utc_now()
        task.status = status
        task.report_json = report_json
        task.markdown_report = markdown_report
        task.error_code = error_code
        task.worker_id = None
        task.active_run_id = None
        task.heartbeat_at = None
        task.completed_at = now
        task.updated_at = now
        task.lifecycle_version += 1
        await self._session.flush()

    async def recover_stale(self, *, stale_before: datetime) -> int:
        """把心跳过期任务转回 retrying，次数耗尽则失败."""
        columns = cast(Any, ResearchTask).__table__.c
        result = await self._session.execute(
            select(ResearchTask)
            .where(
                columns.status.in_((ResearchTaskStatus.RUNNING, ResearchTaskStatus.CANCELLING)),
                columns.heartbeat_at < stale_before,
            )
            .with_for_update(skip_locked=True)
        )
        tasks = tuple(result.scalars().all())
        for task in tasks:
            expired_run_id = task.active_run_id
            previous_status = task.status
            if task.status == ResearchTaskStatus.CANCELLING:
                next_status = ResearchTaskStatus.CANCELLED
            elif task.attempt_count < task.max_attempts:
                next_status = ResearchTaskStatus.RETRYING
            else:
                next_status = ResearchTaskStatus.FAILED
            task.status = next_status
            task.worker_id = None
            task.active_run_id = None
            task.heartbeat_at = None
            task.error_code = "WORKER_HEARTBEAT_EXPIRED"
            now = utc_now()
            task.updated_at = now
            task.lifecycle_version += 1
            if next_status in {ResearchTaskStatus.FAILED, ResearchTaskStatus.CANCELLED}:
                task.completed_at = now
            await self.append_event(
                task_id=task.id,
                user_id=task.user_id,
                event_type="run_lease_expired",
                payload={
                    "run_id": str(expired_run_id) if expired_run_id is not None else None,
                    # SQLModel 的 String 列在部分加载路径返回普通 str，在另一些
                    # 路径保留 StrEnum；统一用 str()，避免恢复流程依赖 ORM 细节。
                    "previous_status": str(previous_status),
                    "status": str(next_status),
                    "attempt_count": task.attempt_count,
                },
            )
        await self._session.flush()
        return len(tasks)


__all__ = ["ResearchTaskRepository"]
