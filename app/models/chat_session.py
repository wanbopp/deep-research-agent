"""聊天会话业务表模型.

- 业务 ChatSession 不等于 SQLAlchemy AsyncSession。
- chat_sessions 不保存完整 LangGraph checkpoint。
- user_id 是业务所有权边界。
- 外键保证引用存在，索引加速按用户查询会话。
"""

from enum import StrEnum
from typing import Any
from uuid import UUID

from sqlalchemy import CheckConstraint, Column, String
from sqlmodel import Field

from app.models.base import UUIDTimestampModel


class ChatSessionStatus(StrEnum):
    """业务会话在可重试删除流程中的持久状态."""

    ACTIVE = "active"
    DELETING = "deleting"


class ChatSession(UUIDTimestampModel, table=True):
    """用户拥有的一次业务聊天会话.

    这个模型只保存产品层面的会话身份和标题。LangGraph messages、当前节点、
    interrupt 等执行状态属于 checkpointer，不能直接塞进这张业务表。
    """

    # 与 User 相同，SQLModel 运行时接受字符串表名，而类型桩使用动态
    # declared_attr；只对这个框架魔术属性放宽类型。
    __tablename__: Any = "chat_sessions"

    # 数据库约束是最后一道防线：即使后台脚本绕过 Pydantic/Service，也不能
    # 写入 cleanup coordinator 无法理解的第三种状态。
    __table_args__ = (
        CheckConstraint(
            "status IN ('active', 'deleting')",
            name="ck_chat_sessions_status",
        ),
    )

    # foreign_key="users.id" 在数据库层保证关联的用户必须存在。
    # nullable=False 表示会话不能脱离用户独立存在。
    # index=True 服务最常见的查询：列出某个用户的全部会话。
    user_id: UUID = Field(
        foreign_key="users.id",
        nullable=False,
        index=True,
    )

    # title 是业务展示信息，不是 Agent prompt，也不是完整消息历史。
    # 先提供稳定默认值，后续 service 可以根据用户输入或模型结果更新标题。
    title: str = Field(
        default="New chat",
        sa_column=Column(String(200), nullable=False),
    )

    # active 表示普通 API 和 Agent 可以访问；deleting 是持久 tombstone，表示
    # 业务行仍保留，但只允许 cleanup coordinator 继续幂等清理。不能只使用
    # 内存标志，否则进程在 checkpoint 删除中途重启后会遗失恢复方向。
    status: ChatSessionStatus = Field(
        default=ChatSessionStatus.ACTIVE,
        sa_column=Column(String(32), nullable=False, index=True),
    )
