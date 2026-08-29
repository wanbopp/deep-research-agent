"""研究图每次执行使用的可信、非持久化上下文."""

from dataclasses import dataclass
from uuid import UUID


@dataclass(frozen=True, slots=True)
class ResearchRuntimeContext:
    """服务端注入的一次研究执行身份.

    user_id 不进入可由模型更新的 ResearchState，避免模型输出、客户端正文或恢复
    checkpoint 时覆盖授权边界。任务 ID 用于日志、取消检查和持久事件关联。
    """

    user_id: UUID
    research_id: UUID


__all__ = ["ResearchRuntimeContext"]
