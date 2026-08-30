"""版本化、可判别的持久 Research 事件协议."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Annotated, Literal, TypeAlias
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter


class ResearchEventType(StrEnum):
    """v1 允许持久化和公开的有限事件名称."""

    TASK_CREATED = "task_created"
    TASK_STARTED = "task_started"
    NODE_COMPLETED = "node_completed"
    CANCELLATION_REQUESTED = "cancellation_requested"
    TASK_RETRYING = "task_retrying"
    RUN_LEASE_EXPIRED = "run_lease_expired"
    TASK_COMPLETED = "task_completed"
    TASK_FAILED = "task_failed"
    TASK_CANCELLED = "task_cancelled"


class _EventModel(BaseModel):
    """事件协议共享的严格模型配置."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)


class TaskStatusPayload(_EventModel):
    """只携带公开任务状态的生命周期 payload."""

    status: str = Field(min_length=1, max_length=32)


class TaskStartedPayload(_EventModel):
    """一次 run 开始时的尝试次数."""

    attempt_count: int = Field(ge=1)


class NodeCompletedPayload(_EventModel):
    """节点完成后的安全摘要，不复制证据或模型正文."""

    node: str = Field(min_length=1, max_length=64)
    status: str | None = Field(default=None, max_length=32)
    evidence_count: int | None = Field(default=None, ge=0)


class TaskRetryingPayload(_EventModel):
    """人工或系统重试时已经消耗的尝试次数."""

    attempt_count: int = Field(ge=0)


class RunLeaseExpiredPayload(_EventModel):
    """Worker lease 过期后状态机执行的确定性恢复结果."""

    expired_run_id: UUID | None = None
    previous_status: str = Field(min_length=1, max_length=32)
    status: str = Field(min_length=1, max_length=32)
    attempt_count: int = Field(ge=0)


class TaskFailedPayload(_EventModel):
    """稳定错误码；禁止写入异常文本或 Provider 响应."""

    error_code: str = Field(min_length=1, max_length=128)


class _Envelope(_EventModel):
    """所有公开事件共享的定位和版本字段."""

    event_id: int = Field(ge=1)
    schema_version: Literal[1] = 1
    run_id: UUID | None = None
    created_at: datetime


class TaskCreatedEvent(_Envelope):
    event: Literal["task_created"]
    payload: TaskStatusPayload


class TaskStartedEvent(_Envelope):
    event: Literal["task_started"]
    payload: TaskStartedPayload


class NodeCompletedEvent(_Envelope):
    event: Literal["node_completed"]
    payload: NodeCompletedPayload


class CancellationRequestedEvent(_Envelope):
    event: Literal["cancellation_requested"]
    payload: TaskStatusPayload


class TaskRetryingEvent(_Envelope):
    event: Literal["task_retrying"]
    payload: TaskRetryingPayload


class RunLeaseExpiredEvent(_Envelope):
    event: Literal["run_lease_expired"]
    payload: RunLeaseExpiredPayload


class TaskCompletedEvent(_Envelope):
    event: Literal["task_completed"]
    payload: TaskStatusPayload


class TaskFailedEvent(_Envelope):
    event: Literal["task_failed"]
    payload: TaskFailedPayload


class TaskCancelledEvent(_Envelope):
    event: Literal["task_cancelled"]
    payload: TaskStatusPayload


ResearchEventV1: TypeAlias = Annotated[
    TaskCreatedEvent
    | TaskStartedEvent
    | NodeCompletedEvent
    | CancellationRequestedEvent
    | TaskRetryingEvent
    | RunLeaseExpiredEvent
    | TaskCompletedEvent
    | TaskFailedEvent
    | TaskCancelledEvent,
    Field(discriminator="event"),
]


class LegacyResearchEvent(_EventModel):
    """迁移前未知事件的只读兼容形状；新代码不得写 schema_version=0."""

    event_id: int = Field(ge=1)
    schema_version: Literal[0] = 0
    event: str = Field(min_length=1, max_length=64)
    run_id: UUID | None = None
    payload: dict[str, object]
    created_at: datetime


ResearchEventResponse: TypeAlias = ResearchEventV1 | LegacyResearchEvent
_EVENT_ADAPTER = TypeAdapter(ResearchEventV1)

_PAYLOAD_ADAPTERS: dict[ResearchEventType, TypeAdapter[object]] = {
    ResearchEventType.TASK_CREATED: TypeAdapter(TaskStatusPayload),
    ResearchEventType.TASK_STARTED: TypeAdapter(TaskStartedPayload),
    ResearchEventType.NODE_COMPLETED: TypeAdapter(NodeCompletedPayload),
    ResearchEventType.CANCELLATION_REQUESTED: TypeAdapter(TaskStatusPayload),
    ResearchEventType.TASK_RETRYING: TypeAdapter(TaskRetryingPayload),
    ResearchEventType.RUN_LEASE_EXPIRED: TypeAdapter(RunLeaseExpiredPayload),
    ResearchEventType.TASK_COMPLETED: TypeAdapter(TaskStatusPayload),
    ResearchEventType.TASK_FAILED: TypeAdapter(TaskFailedPayload),
    ResearchEventType.TASK_CANCELLED: TypeAdapter(TaskStatusPayload),
}


def validate_research_event_payload(
    event_type: ResearchEventType,
    payload: dict[str, object],
) -> dict[str, object]:
    """在数据库写入前严格验证 payload，并返回 JSON 可序列化结果."""
    model = _PAYLOAD_ADAPTERS[event_type].validate_python(payload)
    if not isinstance(model, BaseModel):
        raise TypeError("research event payload adapter returned a non-model value")
    return model.model_dump(mode="json", exclude_none=True)


def parse_research_event(data: dict[str, object]) -> ResearchEventResponse:
    """把数据库投影恢复为强类型事件；legacy 只用于未知旧记录."""
    if data.get("schema_version") == 0:
        return LegacyResearchEvent.model_validate(data)
    return _EVENT_ADAPTER.validate_python(data)


__all__ = [
    "LegacyResearchEvent",
    "ResearchEventResponse",
    "ResearchEventType",
    "ResearchEventV1",
    "parse_research_event",
    "validate_research_event_payload",
]
