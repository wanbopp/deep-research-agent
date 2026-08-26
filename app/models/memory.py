"""长期记忆的 PostgreSQL/pgvector 业务表模型."""

from typing import Any
from uuid import UUID

from pgvector.sqlalchemy import Vector
from sqlalchemy import CheckConstraint, Column, Index, String, Text
from sqlmodel import Field

from app.models.base import UUIDTimestampModel

# 向量维度是数据库列类型的一部分，不只是运行时参数。修改该值必须同时新增
# Alembic migration，并重新生成已有向量；只修改环境变量会造成新旧向量不兼容。
MEMORY_EMBEDDING_DIMENSIONS = 1536


class Memory(UUIDTimestampModel, table=True):
    """用户级长期记忆及其语义向量.

    ORM model 描述数据库真实形状；API/Application schema 则隐藏 ``embedding``。
    这能避免高维向量被误放进响应、日志或 Agent state，既减少传输成本，也缩小
    用户文本派生数据的暴露面。
    """

    __tablename__: Any = "memories"

    __table_args__ = (
        # Python 枚举保护正常应用代码，数据库约束保护脚本和其他写入入口。
        CheckConstraint(
            "kind IN ('preference', 'fact', 'constraint')",
            name="ck_memories_kind",
        ),
        # 先按可信 user_id 缩小范围，再按 kind 过滤，是最常见的检索前置条件。
        # 该普通 B-tree 索引不会替代向量排序，只负责减少候选行。
        Index("ix_memories_user_id_kind", "user_id", "kind"),
    )

    # user_id 是长期记忆的安全 namespace。任何搜索、删除和来源校验都必须在 SQL
    # 中包含该字段，不能先全局查询再在 Python 中过滤。
    user_id: UUID = Field(
        foreign_key="users.id",
        nullable=False,
        index=True,
    )

    # 只保存提炼后的单条记忆，不保存完整聊天记录或 Prompt。
    content: str = Field(
        sa_column=Column(Text, nullable=False),
    )

    # ORM 使用数据库真实字符串，应用边界再转换为 MemoryKind。这样 models 层不会
    # 反向依赖 API schema，同时数据库 CheckConstraint 仍保证只有三个合法值。
    kind: str = Field(
        sa_column=Column(String(32), nullable=False),
    )

    # 来源会话用于审计和未来重建。外键只能证明会话存在，不能证明它属于同一
    # 用户；PostgresMemoryStore.add() 还必须执行 user_id 联合校验。
    source_thread_id: UUID = Field(
        foreign_key="chat_sessions.id",
        nullable=False,
        index=True,
    )

    # pgvector 的 VECTOR(1536) 在数据库层拒绝错误维度。Python 使用 list 是因为
    # SQLAlchemy 驱动读取后返回可变列表；该字段不会离开 infrastructure 层。
    embedding: list[float] = Field(
        sa_column=Column(
            Vector(MEMORY_EMBEDDING_DIMENSIONS),
            nullable=False,
        ),
    )


__all__ = ["MEMORY_EMBEDDING_DIMENSIONS", "Memory"]
