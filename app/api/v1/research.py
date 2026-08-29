"""持久 DeepResearch 任务的创建、查询、取消、重试与事件流 API."""

import asyncio
from collections.abc import AsyncIterator
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, Header, Request, Response, status
from fastapi.responses import StreamingResponse

from app.api.dependencies import AgentRateLimitDependency, CurrentUserDependency, get_research_service
from app.api.sse import encode_research_sse_event
from app.schemas.base import ErrorResponse
from app.schemas.research import ResearchCreateRequest, ResearchTaskListResponse, ResearchTaskResponse
from app.services.research import ResearchService

router = APIRouter(prefix="/research", tags=["research"])
ResearchServiceDependency = Annotated[ResearchService, Depends(get_research_service)]
TERMINAL_STATUSES = frozenset({"completed", "failed", "cancelled"})


@router.post(
    "",
    response_model=ResearchTaskResponse,
    status_code=status.HTTP_202_ACCEPTED,
    responses={
        status.HTTP_401_UNAUTHORIZED: {"model": ErrorResponse, "description": "Authentication required"},
        status.HTTP_409_CONFLICT: {"model": ErrorResponse, "description": "Idempotency conflict"},
        status.HTTP_429_TOO_MANY_REQUESTS: {
            "model": ErrorResponse,
            "description": "Research task rate limit exceeded",
        },
        status.HTTP_503_SERVICE_UNAVAILABLE: {"model": ErrorResponse, "description": "Rate limiter unavailable"},
    },
)
async def create_research(
    body: ResearchCreateRequest,
    service: ResearchServiceDependency,
    current_user: CurrentUserDependency,
    response: Response,
    idempotency_key: Annotated[str, Header(alias="Idempotency-Key", min_length=1, max_length=128)],
    _rate_limit: AgentRateLimitDependency,
) -> ResearchTaskResponse:
    """创建持久任务并立即返回 202；模型调用由 worker 稍后执行."""
    task, created = await service.create(
        user_id=current_user.user_id,
        request=body,
        idempotency_key=idempotency_key,
    )
    if not created:
        # 相同请求重试返回原任务，200 比再次声称“新建成功”更准确。
        response.status_code = status.HTTP_200_OK
    return task


@router.get("", response_model=ResearchTaskListResponse)
async def list_research_tasks(
    service: ResearchServiceDependency,
    current_user: CurrentUserDependency,
) -> ResearchTaskListResponse:
    """列出当前认证用户自己的研究任务."""
    return await service.list(user_id=current_user.user_id)


@router.get("/{research_id}", response_model=ResearchTaskResponse)
async def read_research_task(
    research_id: UUID,
    service: ResearchServiceDependency,
    current_user: CurrentUserDependency,
) -> ResearchTaskResponse:
    """读取任务状态和完成后的结构化/Markdown 报告."""
    return await service.get(task_id=research_id, user_id=current_user.user_id)


@router.post("/{research_id}/cancel", response_model=ResearchTaskResponse)
async def cancel_research_task(
    research_id: UUID,
    service: ResearchServiceDependency,
    current_user: CurrentUserDependency,
) -> ResearchTaskResponse:
    """持久化取消请求；运行中的 worker 会取消当前 Graph 协程."""
    return await service.cancel(task_id=research_id, user_id=current_user.user_id)


@router.post("/{research_id}/retry", response_model=ResearchTaskResponse)
async def retry_research_task(
    research_id: UUID,
    service: ResearchServiceDependency,
    current_user: CurrentUserDependency,
    _rate_limit: AgentRateLimitDependency,
) -> ResearchTaskResponse:
    """把符合条件的失败任务重新放入队列."""
    return await service.retry(task_id=research_id, user_id=current_user.user_id)


@router.get(
    "/{research_id}/stream",
    response_class=StreamingResponse,
    responses={200: {"content": {"text/event-stream": {}}}},
)
async def stream_research_events(
    research_id: UUID,
    request: Request,
    service: ResearchServiceDependency,
    current_user: CurrentUserDependency,
    last_event_id: Annotated[int | None, Header(alias="Last-Event-ID", ge=0)] = None,
) -> StreamingResponse:
    """从持久事件表持续发送进度；SSE 只是观察通道，不是任务事实来源."""
    # 在返回 StreamingResponse 前先验证所有权，让 404 仍能走普通 JSON handler。
    await service.get(task_id=research_id, user_id=current_user.user_id)

    async def generate() -> AsyncIterator[str]:
        """轮询新事件；断线不取消后台任务，终态事件发完后正常结束."""
        cursor = last_event_id or 0
        while not await request.is_disconnected():
            events = await service.events(
                task_id=research_id,
                user_id=current_user.user_id,
                after_id=cursor,
            )
            for event in events:
                cursor = event.event_id
                yield encode_research_sse_event(event)

            task = await service.get(task_id=research_id, user_id=current_user.user_id)
            if task.status in TERMINAL_STATUSES and not events:
                break
            if not events:
                # SSE 注释帧保持代理连接活跃，但不会占用业务 event ID。
                yield ": keep-alive\n\n"
                await asyncio.sleep(1.0)

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


__all__ = ["router"]
