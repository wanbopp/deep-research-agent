"""文档索引任务的 PostgreSQL 领取与状态转换边界."""

from datetime import datetime, timedelta
from uuid import UUID, uuid4

from sqlalchemy import and_, or_
from sqlmodel import col, select

from app.models import Document, DocumentStatus, IndexJob, IndexJobStatus, utc_now
from app.repositories.base import RepositoryBase


class IndexJobRepository(RepositoryBase):
    """使用数据库行锁协调多个索引 worker."""

    async def create(self, *, document_id: UUID, max_attempts: int) -> IndexJob:
        """在 Document 同一事务中创建唯一 pending job."""
        job = IndexJob(
            document_id=document_id,
            max_attempts=max_attempts,
            status=IndexJobStatus.PENDING,
        )
        return await self._persist(job, resource="index_job")

    async def get_by_document_id(
        self,
        document_id: UUID,
        *,
        user_id: UUID,
    ) -> IndexJob | None:
        """通过 Document owner 过滤读取 job，避免只凭 job UUID 越权."""
        statement = (
            select(IndexJob)
            .join(Document, col(Document.id) == col(IndexJob.document_id))
            .where(
                IndexJob.document_id == document_id,
                Document.user_id == user_id,
                Document.status != DocumentStatus.DELETING,
            )
        )
        result = await self._session.execute(statement)
        return result.scalar_one_or_none()

    async def list_by_document_ids(
        self,
        document_ids: tuple[UUID, ...],
    ) -> tuple[IndexJob, ...]:
        """批量读取列表页需要的 job，避免为每个 Document 单独发一次 SQL."""
        if not document_ids:
            return ()
        statement = select(IndexJob).where(col(IndexJob.document_id).in_(document_ids))
        result = await self._session.execute(statement)
        return tuple(result.scalars().all())

    async def get_for_update(self, *, document_id: UUID) -> IndexJob | None:
        """锁定指定文档的 job，供删除或人工重试检查状态."""
        statement = select(IndexJob).where(IndexJob.document_id == document_id).with_for_update()
        result = await self._session.execute(statement)
        return result.scalar_one_or_none()

    async def claim_next(
        self,
        *,
        worker_id: str,
        now: datetime,
        lease_seconds: float,
    ) -> IndexJob | None:
        """原子领取一个 pending 或租约已过期的 running job.

        ``FOR UPDATE SKIP LOCKED`` 让多个 worker 竞争时各自跳过别人已经锁定的行。
        状态和租约在同一事务提交，因此进程重启后 PostgreSQL 仍保留恢复依据。
        """
        claimable = or_(
            col(IndexJob.status) == IndexJobStatus.PENDING,
            and_(
                col(IndexJob.status) == IndexJobStatus.RUNNING,
                col(IndexJob.lease_expires_at).is_not(None),
                col(IndexJob.lease_expires_at) < now,
            ),
        )
        statement = (
            select(IndexJob)
            .join(Document, col(Document.id) == col(IndexJob.document_id))
            .where(
                claimable,
                col(IndexJob.attempt_count) < col(IndexJob.max_attempts),
                col(Document.status).in_(
                    (
                        DocumentStatus.PENDING,
                        DocumentStatus.INDEXING,
                        DocumentStatus.FAILED,
                    )
                ),
            )
            .order_by(col(IndexJob.created_at), col(IndexJob.id))
            # 这里只锁 IndexJob，随后 worker 再按固定顺序锁 Document。删除、重试
            # 和收尾也使用 Job -> Document 顺序，避免相反锁顺序形成数据库死锁。
            .with_for_update(skip_locked=True, of=IndexJob)
            .limit(1)
        )
        result = await self._session.execute(statement)
        job = result.scalar_one_or_none()
        if job is None:
            return None

        job.status = IndexJobStatus.RUNNING
        job.attempt_count += 1
        job.claim_token = uuid4()
        job.claimed_by = worker_id
        job.lease_expires_at = now + timedelta(seconds=lease_seconds)
        job.started_at = now
        job.finished_at = None
        job.error_code = None
        job.updated_at = now
        self._session.add(job)
        await self._session.flush()
        await self._session.refresh(job)
        return job

    async def get_claim_for_update(
        self,
        *,
        job_id: UUID,
        claim_token: UUID,
    ) -> IndexJob | None:
        """按 job 与 fencing token 锁定当前 claim，拒绝过期 worker 收尾."""
        statement = (
            select(IndexJob)
            .where(
                IndexJob.id == job_id,
                IndexJob.status == IndexJobStatus.RUNNING,
                IndexJob.claim_token == claim_token,
            )
            .with_for_update()
        )
        result = await self._session.execute(statement)
        return result.scalar_one_or_none()

    async def mark_completed(self, job: IndexJob, *, now: datetime) -> None:
        """清除租约并把仍属于当前 worker 的 job 标记完成."""
        job.status = IndexJobStatus.COMPLETED
        job.claim_token = None
        job.claimed_by = None
        job.lease_expires_at = None
        job.finished_at = now
        job.error_code = None
        job.updated_at = now
        self._session.add(job)
        await self._session.flush()

    async def mark_failed(self, job: IndexJob, *, now: datetime, error_code: str) -> None:
        """保存稳定错误码并释放 claim，不保存 provider 或文档正文."""
        job.status = IndexJobStatus.FAILED
        job.claim_token = None
        job.claimed_by = None
        job.lease_expires_at = None
        job.finished_at = now
        job.error_code = error_code
        job.updated_at = now
        self._session.add(job)
        await self._session.flush()

    async def reset_failed(self, job: IndexJob) -> None:
        """由显式人工重试把失败 job 放回 pending，但不抹掉已消耗次数."""
        job.status = IndexJobStatus.PENDING
        job.claim_token = None
        job.claimed_by = None
        job.lease_expires_at = None
        job.started_at = None
        job.finished_at = None
        job.error_code = None
        job.updated_at = utc_now()
        self._session.add(job)
        await self._session.flush()


__all__ = ["IndexJobRepository"]
