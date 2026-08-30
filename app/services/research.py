"""持久研究任务的请求级应用服务."""

from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from app.models import ResearchEvent, ResearchTask, ResearchTaskStatus
from app.models.base import utc_now
from app.repositories import ResearchTaskRepository
from app.schemas.research import (
    ResearchConfig,
    ResearchCreateRequest,
    ResearchReport,
    ResearchTaskListResponse,
    ResearchTaskResponse,
)
from app.schemas.research_events import ResearchEventResponse, ResearchEventType, parse_research_event


class ResearchServiceError(RuntimeError):
    """可安全转换为 HTTP 响应的研究任务业务错误基类."""


class ResearchTaskNotFoundError(ResearchServiceError):
    """任务不存在或不属于当前用户."""


class ResearchTaskConflictError(ResearchServiceError):
    """幂等键或当前状态与请求操作冲突."""


class ResearchTaskNotRetryableError(ResearchServiceError):
    """任务不是可重试失败，或已经耗尽执行次数."""


class ResearchService:
    """协调一次 HTTP 请求中的任务状态事务和公开响应转换."""

    def __init__(self, session: AsyncSession, *, max_attempts: int = 2) -> None:
        """保存请求级 Session 和服务端最大尝试次数."""
        if max_attempts <= 0:
            raise ValueError("max_attempts must be greater than zero")
        self._session = session
        self._max_attempts = max_attempts

    async def create(
        self,
        *,
        user_id: UUID,
        request: ResearchCreateRequest,
        idempotency_key: str,
    ) -> tuple[ResearchTaskResponse, bool]:
        """创建 pending 任务；重复键返回原任务而不是重复调用模型.

        Returns:
            ``(task, created)``。created=False 表示客户端重试命中了原任务。
        """
        key = idempotency_key.strip()
        if not key or len(key) > 128:
            raise ResearchTaskConflictError("invalid idempotency key")

        async with self._session.begin():
            repository = ResearchTaskRepository(self._session)
            # 先按 user + key 取得事务锁，再执行“查询或创建”。这使两个同时到达的
            # 重试请求也只创建一行，后到者等待前一事务提交后读取原任务。
            await repository.lock_idempotency_key(user_id=user_id, key=key)
            existing = await repository.get_by_idempotency_key(user_id=user_id, key=key)
            if existing is not None:
                expected_config = request.config.model_dump(mode="json")
                if existing.topic != request.topic or existing.config_json != expected_config:
                    raise ResearchTaskConflictError("idempotency key belongs to a different request")
                return self._to_response(existing), False

            task = await repository.create(
                user_id=user_id,
                topic=request.topic,
                chat_session_id=request.chat_session_id,
                config_json=request.config.model_dump(mode="json"),
                idempotency_key=key,
                max_attempts=self._max_attempts,
            )
            await repository.append_event(
                task_id=task.id,
                user_id=user_id,
                event_type=ResearchEventType.TASK_CREATED,
                payload={"status": str(task.status)},
            )
        return self._to_response(task), True

    async def get(self, *, task_id: UUID, user_id: UUID) -> ResearchTaskResponse:
        """读取当前用户任务，跨用户时不泄漏任务是否存在."""
        async with self._session.begin():
            task = await ResearchTaskRepository(self._session).get_by_id(task_id, user_id=user_id)
            if task is None:
                raise ResearchTaskNotFoundError
        return self._to_response(task)

    async def list(self, *, user_id: UUID) -> ResearchTaskListResponse:
        """列出当前用户任务，不返回内部 worker 字段."""
        async with self._session.begin():
            tasks = await ResearchTaskRepository(self._session).list_by_user(user_id)
        return ResearchTaskListResponse(tasks=tuple(self._to_response(task) for task in tasks))

    async def cancel(self, *, task_id: UUID, user_id: UUID) -> ResearchTaskResponse:
        """持久化取消意图；运行中 worker 会轮询并取消当前图协程."""
        async with self._session.begin():
            repository = ResearchTaskRepository(self._session)
            task = await repository.lock_owned(task_id=task_id, user_id=user_id)
            if task is None:
                raise ResearchTaskNotFoundError
            if task.status in {ResearchTaskStatus.COMPLETED, ResearchTaskStatus.FAILED, ResearchTaskStatus.CANCELLED}:
                raise ResearchTaskConflictError("terminal research task cannot be cancelled")
            now = utc_now()
            task.cancellation_requested_at = now
            task.updated_at = now
            task.lifecycle_version += 1
            if task.status in {ResearchTaskStatus.PENDING, ResearchTaskStatus.RETRYING}:
                task.status = ResearchTaskStatus.CANCELLED
                task.completed_at = now
                event_type = ResearchEventType.TASK_CANCELLED
            else:
                task.status = ResearchTaskStatus.CANCELLING
                event_type = ResearchEventType.CANCELLATION_REQUESTED
            await repository.append_event(
                task_id=task.id,
                user_id=user_id,
                event_type=event_type,
                payload={"status": str(task.status)},
            )
        return self._to_response(task)

    async def retry(self, *, task_id: UUID, user_id: UUID) -> ResearchTaskResponse:
        """把未耗尽次数的 failed 任务重新放回持久队列."""
        async with self._session.begin():
            repository = ResearchTaskRepository(self._session)
            task = await repository.lock_owned(task_id=task_id, user_id=user_id)
            if task is None:
                raise ResearchTaskNotFoundError
            if task.status != ResearchTaskStatus.FAILED or task.attempt_count >= task.max_attempts:
                raise ResearchTaskNotRetryableError
            task.status = ResearchTaskStatus.RETRYING
            task.error_code = None
            task.completed_at = None
            task.cancellation_requested_at = None
            task.updated_at = utc_now()
            task.lifecycle_version += 1
            await repository.append_event(
                task_id=task.id,
                user_id=user_id,
                event_type=ResearchEventType.TASK_RETRYING,
                payload={"attempt_count": task.attempt_count},
            )
        return self._to_response(task)

    async def events(
        self,
        *,
        task_id: UUID,
        user_id: UUID,
        after_id: int = 0,
    ) -> tuple[ResearchEventResponse, ...]:
        """读取指定 ID 之后的持久事件，先验证任务所有权."""
        async with self._session.begin():
            repository = ResearchTaskRepository(self._session)
            if await repository.get_by_id(task_id, user_id=user_id) is None:
                raise ResearchTaskNotFoundError
            events = await repository.list_events(task_id=task_id, user_id=user_id, after_id=after_id)
        return tuple(self._to_event_response(event) for event in events)

    @staticmethod
    def _to_response(task: ResearchTask) -> ResearchTaskResponse:
        """把数据库模型转换为不暴露内部领取信息的公开结构."""
        return ResearchTaskResponse(
            research_id=task.id,
            topic=task.topic,
            status=str(task.status),
            config=ResearchConfig.model_validate(task.config_json),
            attempt_count=task.attempt_count,
            max_attempts=task.max_attempts,
            error_code=task.error_code,
            report=ResearchReport.model_validate(task.report_json) if task.report_json is not None else None,
            markdown_report=task.markdown_report,
            created_at=task.created_at,
            updated_at=task.updated_at,
            completed_at=task.completed_at,
        )

    @staticmethod
    def _to_event_response(event: ResearchEvent) -> ResearchEventResponse:
        """要求数据库事件已有自增 ID 后再公开."""
        if event.id is None:
            raise RuntimeError("persisted research event is missing its ID")
        return parse_research_event(
            {
                "event_id": event.id,
                "schema_version": event.schema_version,
                "event": event.event_type,
                "run_id": event.run_id,
                "payload": event.payload_json,
                "created_at": event.created_at,
            }
        )


__all__ = [
    "ResearchService",
    "ResearchServiceError",
    "ResearchTaskConflictError",
    "ResearchTaskNotFoundError",
    "ResearchTaskNotRetryableError",
]
