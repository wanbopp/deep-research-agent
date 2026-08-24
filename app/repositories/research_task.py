"""深度研究任务数据访问边界."""

from uuid import UUID

from sqlmodel import select

from app.models import ResearchTask, ResearchTaskStatus
from app.repositories.base import RepositoryBase


class ResearchTaskRepository(RepositoryBase):
    """封装研究任务的创建和用户所有权查询."""

    async def create(
        self,
        *,
        user_id: UUID,
        topic: str,
        chat_session_id: UUID | None = None,
        status: ResearchTaskStatus = ResearchTaskStatus.PENDING,
    ) -> ResearchTask:
        """在当前事务中创建研究任务记录."""
        task = ResearchTask(
            user_id=user_id,
            chat_session_id=chat_session_id,
            topic=topic,
            status=status,
        )
        return await self._persist(task, resource="research_task")

    async def get_by_id(self, task_id: UUID, *, user_id: UUID) -> ResearchTask | None:
        """同时按任务 ID 和用户 ID 查询，避免越权读取任务."""
        statement = select(ResearchTask).where(
            ResearchTask.id == task_id,
            ResearchTask.user_id == user_id,
        )
        result = await self._session.execute(statement)
        return result.scalar_one_or_none()

    async def list_by_user(self, user_id: UUID) -> tuple[ResearchTask, ...]:
        """稳定地列出一个用户拥有的全部研究任务."""
        statement = select(ResearchTask).where(ResearchTask.user_id == user_id).order_by("created_at", "id")
        result = await self._session.execute(statement)
        return tuple(result.scalars().all())
