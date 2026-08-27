"""聊天会话业务表模型.

- 业务 ChatSession 不等于 SQLAlchemy AsyncSession。
- chat_sessions 不保存完整 LangGraph checkpoint。
- user_id 是业务所有权边界。
- 外键保证引用存在，索引加速按用户查询会话。
"""

from enum import StrEnum
from datetime import datetime
from typing import Any
from uuid import UUID

from sqlalchemy import CheckConstraint, Column, String, Uuid
from sqlmodel import Field

from app.models.base import UTCDateTime, UUIDTimestampModel

DEFAULT_CHAT_SESSION_TITLE = "New chat"
MAX_CHAT_SESSION_TITLE_LENGTH = 200


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
        # token 与时间必须同时存在或同时为空。数据库约束保护所有写入入口，
        # 即使维护脚本绕过 Repository，也不能制造无法判断租约年龄的半状态。
        CheckConstraint(
            "(title_claim_token IS NULL AND title_claimed_at IS NULL) OR "
            "(title_claim_token IS NOT NULL AND title_claimed_at IS NOT NULL)",
            name="ck_chat_sessions_title_claim_pair",
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
        default=DEFAULT_CHAT_SESSION_TITLE,
        sa_column=Column(String(MAX_CHAT_SESSION_TITLE_LENGTH), nullable=False),
    )

    # 自动命名使用数据库租约，而不是 asyncio.Lock。token 标识本次占有者，
    # claimed_at 用于 worker 崩溃后的超时接管。二者都不公开给 API 或 Agent。
    # 谁拥有当前租约。
    title_claim_token: UUID | None = Field(
        default=None,
        sa_column=Column(Uuid(), nullable=True),
    )
    # 租约从什么时候开始
    title_claimed_at: datetime | None = Field(
        default=None,
        sa_type=UTCDateTime,
        nullable=True,
    )
    # 自动标题是否已经成功生成
    # 成功写入自动标题后记录完成时间。它与 title 分开保存，使系统能区分
    # “模型生成成功”和“用户从创建时就提供了自定义标题”两种业务来源。
    title_generated_at: datetime | None = Field(
        default=None,
        sa_type=UTCDateTime,
        nullable=True,
    )

    # active 表示普通 API 和 Agent 可以访问；deleting 是持久 tombstone，表示
    # 业务行仍保留，但只允许 cleanup coordinator 继续幂等清理。不能只使用
    # 内存标志，否则进程在 checkpoint 删除中途重启后会遗失恢复方向。
    status: ChatSessionStatus = Field(
        default=ChatSessionStatus.ACTIVE,
        sa_column=Column(String(32), nullable=False, index=True),
    )
