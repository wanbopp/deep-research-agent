"""用户业务表模型."""

from typing import Any

from sqlalchemy import Column, String, UniqueConstraint
from sqlmodel import Field

from app.models.base import UUIDTimestampModel


class User(UUIDTimestampModel, table=True):
    """应用用户的最小持久化实体.

    这里只描述数据库行和约束。邮箱格式校验属于 API schema，邮箱规范化与
    密码哈希属于 authentication service；ORM 只接收已经生成的 credential。
    """

    # SQLModel/SQLAlchemy 在运行时允许字符串表名，但基类类型桩把这个动态属性
    # 声明为 declared_attr。只对框架魔术属性使用 Any，避免为此关闭整个文件的
    # 类型检查；email 等业务字段仍然保留严格类型。
    __tablename__: Any = "users"

    # 显式命名唯一约束，能够让 Alembic migration 和数据库报错更容易审查。
    # index=True 服务按邮箱查找用户的常见路径；唯一约束则保证无论数据从
    # API、脚本还是后台任务写入，同一邮箱都不能出现两行。
    __table_args__ = (UniqueConstraint("email", name="uq_users_email"),)

    # 使用 str 而不是 EmailStr：ORM model 负责数据库形状，外部输入的邮箱格式
    # 应先由 API Pydantic schema 验证。String(320) 则把长度限制落实到数据库列。
    email: str = Field(
        sa_column=Column(String(320), nullable=False, index=True),
    )

    # 数据库只保存 Argon2id 编码串，不保存明文密码。该字符串已经内含算法、
    # 参数、随机 salt 和哈希结果，所以不需要额外 salt 列。它仍是敏感 credential：
    # 可以持久化和用于 verify，但不能进入 API 响应或普通日志。
    #
    # 不为本列建索引：认证先按唯一 email 找到一行，再读取其哈希；按哈希搜索
    # 没有业务意义，还会增加索引副本和暴露面。255 字符足以容纳编码后的哈希。
    password_hash: str = Field(
        sa_column=Column(String(255), nullable=False),
    )
