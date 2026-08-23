"""用户业务表模型."""

from typing import Any

from sqlalchemy import Column, String, UniqueConstraint
from sqlmodel import Field

from app.models.base import UUIDTimestampModel


class User(UUIDTimestampModel, table=True):
    """应用用户的最小持久化实体.

    这里只描述数据库行和约束。邮箱格式校验属于 API schema，邮箱规范化、
    密码哈希和注册规则属于后续 authentication service，不放进 ORM 模型。
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
