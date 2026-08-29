"""使用真实 PostgreSQL 验收研究任务的取消、重试、恢复和事件补发.

这里不需要调用模型：要验证的是数据库事务和任务状态变化，而不是回答质量。脚本
不使用 fake Repository；所有状态都经过正式 ResearchService/Repository 写入，并在
结束时删除临时任务。输出不包含用户资料、数据库地址或任务正文。
"""

import asyncio
import json
import selectors
from datetime import timedelta
from typing import Any, cast
from uuid import UUID, uuid4

from sqlalchemy import delete, func
from sqlmodel import select

from app.core.config import settings
from app.infrastructure.factory import create_application_resources
from app.models import ResearchEvent, ResearchTask, ResearchTaskStatus, User, utc_now
from app.repositories import ResearchTaskRepository
from app.schemas.research import ResearchConfig, ResearchCreateRequest
from app.services.research import ResearchService, ResearchTaskNotFoundError


async def run_smoke() -> dict[str, object]:
    """执行真实状态转换并返回不含业务数据的布尔摘要."""
    resources = create_application_resources(settings)
    task_ids: list[UUID] = []
    try:
        # 需要两个真实用户验证所有权边界，但不会输出它们的 ID 或账户信息。
        async with resources.orm_session_factory() as session, session.begin():
            user_columns = cast(Any, User).__table__.c
            users = tuple(
                (await session.execute(select(User.id).order_by(user_columns.created_at).limit(2))).scalars()
            )
        if len(users) < 2:
            raise RuntimeError("lifecycle smoke requires at least two existing users")
        owner_id, outsider_id = users

        request = ResearchCreateRequest(
            topic="验证持久研究任务的生命周期状态",
            config=ResearchConfig(max_steps=1, max_iterations=1),
        )

        async def create_task() -> UUID:
            """通过正式 Service 创建独立临时任务，并登记统一清理列表."""
            async with resources.orm_session_factory() as session:
                task, created = await ResearchService(session).create(
                    user_id=owner_id,
                    request=request,
                    idempotency_key=f"phase7-lifecycle-{uuid4().hex}",
                )
            if not created:
                raise RuntimeError("fresh lifecycle smoke key unexpectedly existed")
            task_ids.append(task.research_id)
            return task.research_id

        # pending 任务还没有进入 Graph，取消应立刻成为 cancelled，不需要等 Worker。
        cancelled_id = await create_task()
        async with resources.orm_session_factory() as session:
            cancelled = await ResearchService(session).cancel(task_id=cancelled_id, user_id=owner_id)
        cancel_ok = cancelled.status == ResearchTaskStatus.CANCELLED.value

        # 先模拟 Worker 真实领取，再通过 Repository 写入失败终态。重试请求必须把
        # 未耗尽次数的 failed 任务重新放回 retrying 队列。
        retry_id = await create_task()
        async with resources.orm_session_factory() as session, session.begin():
            repository = ResearchTaskRepository(session)
            claimed = await repository.claim_next(worker_id="phase7-lifecycle-retry")
            if claimed is None or claimed.id != retry_id:
                raise RuntimeError("retry smoke task was not claimed")
            await repository.finalize(
                claimed,
                status=ResearchTaskStatus.FAILED,
                error_code="SMOKE_EXPECTED_FAILURE",
            )
        async with resources.orm_session_factory() as session:
            retried = await ResearchService(session).retry(task_id=retry_id, user_id=owner_id)
        retry_ok = retried.status == ResearchTaskStatus.RETRYING.value
        # 结束这条已验证任务，避免它仍留在队首干扰下面的失联恢复场景。
        async with resources.orm_session_factory() as session, session.begin():
            repository = ResearchTaskRepository(session)
            reclaimed = await repository.claim_next(worker_id="phase7-lifecycle-retry-cleanup")
            if reclaimed is None or reclaimed.id != retry_id:
                raise RuntimeError("retried smoke task was not reclaimed")
            await repository.finalize(reclaimed, status=ResearchTaskStatus.CANCELLED)

        # 把已领取任务的心跳移到过去，模拟进程突然退出。新 Worker 的恢复扫描应
        # 将它转回 retrying；这条路径不依赖旧进程执行 finally。
        stale_id = await create_task()
        async with resources.orm_session_factory() as session, session.begin():
            repository = ResearchTaskRepository(session)
            claimed = await repository.claim_next(worker_id="phase7-lifecycle-stale")
            if claimed is None or claimed.id != stale_id:
                raise RuntimeError("stale smoke task was not claimed")
            claimed.heartbeat_at = utc_now() - timedelta(minutes=10)
            await session.flush()
        async with resources.orm_session_factory() as session, session.begin():
            recovered_count = await ResearchTaskRepository(session).recover_stale(
                stale_before=utc_now() - timedelta(minutes=1)
            )
        async with resources.orm_session_factory() as session:
            recovered = await ResearchService(session).get(task_id=stale_id, user_id=owner_id)
        recovery_ok = recovered_count >= 1 and recovered.status == ResearchTaskStatus.RETRYING.value

        # SSE 重连本质是用 Last-Event-ID 查询更大的事件 ID。这里直接验证持久事件
        # 查询边界，HTTP 编码层已由 encode_research_sse_event 负责。
        async with resources.orm_session_factory() as session:
            service = ResearchService(session)
            all_events = await service.events(task_id=cancelled_id, user_id=owner_id)
            replayed_events = await service.events(
                task_id=cancelled_id,
                user_id=owner_id,
                after_id=all_events[0].event_id,
            )
        replay_ok = bool(replayed_events) and all(event.event_id > all_events[0].event_id for event in replayed_events)

        # 不存在与属于其他用户统一抛 NotFound，避免 API 成为任务 ID 探测器。
        ownership_ok = False
        try:
            async with resources.orm_session_factory() as session:
                await ResearchService(session).get(task_id=cancelled_id, user_id=outsider_id)
        except ResearchTaskNotFoundError:
            ownership_ok = True

        return {
            "ok": cancel_ok and retry_ok and recovery_ok and replay_ok and ownership_ok,
            "pending_cancel_ok": cancel_ok,
            "failed_retry_ok": retry_ok,
            "stale_recovery_ok": recovery_ok,
            "event_replay_ok": replay_ok,
            "cross_user_rejected": ownership_ok,
        }
    finally:
        if task_ids:
            async with resources.orm_session_factory() as session, session.begin():
                task_columns = cast(Any, ResearchTask).__table__.c
                await session.execute(delete(ResearchTask).where(task_columns.id.in_(task_ids)))
            # 外键级联是任务清理协议的一部分，不能只假设数据库会替我们完成。
            async with resources.orm_session_factory() as session, session.begin():
                event_columns = cast(Any, ResearchEvent).__table__.c
                remaining = (
                    await session.execute(
                        select(func.count())
                        .select_from(ResearchEvent)
                        .where(event_columns.research_task_id.in_(task_ids))
                    )
                ).scalar_one()
                if remaining != 0:
                    raise RuntimeError("research event cascade cleanup failed")
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
