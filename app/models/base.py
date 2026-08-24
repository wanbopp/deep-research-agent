"""SQLModel 业务表共享的主键与时间字段."""

from datetime import UTC, datetime
from uuid import UUID, uuid4

from sqlalchemy import DateTime
from sqlmodel import Field, SQLModel

# 约束名是 migration 的稳定接口。没有命名规则时 PostgreSQL 会自行生成名字，
# 后续 downgrade 或修改外键就必须猜数据库方言的命名结果。
# CheckConstraint 已在具体模型中显式命名，因此这里不覆盖 ck 规则。
NAMING_CONVENTION = {
    "ix": "ix_%(table_name)s_%(column_0_name)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}
SQLModel.metadata.naming_convention = NAMING_CONVENTION


class UTCDateTime(DateTime):
    """始终创建 ``timezone=True`` 的 SQLAlchemy DateTime 类型.

    SQLModel 的 ``sa_type`` 要求接收“类型类”，随后会为每个字段单独实例化。
    因此这里把 timezone 配置封装进类型类，既不会跨表复用 Column，也能让
    Pyright 保持严格检查。
    """

    def __init__(self) -> None:
        """初始化带时区的数据库时间类型."""
        super().__init__(timezone=True)


def utc_now() -> datetime:
    """返回带时区的当前 UTC 时间.

    必须把函数本身传给 ``default_factory``，不能写成 ``utc_now()``。
    前者表示“每次创建模型时调用”，后者会在模块导入时立即执行，导致后续
    创建的对象可能错误地共享同一个旧时间。
    """
    return datetime.now(UTC)


class UUIDTimestampModel(SQLModel):
    """为业务 table model 提供 UUID 主键和审计时间.

    这里故意不写 ``table=True``，所以它只是字段基类，不会单独注册成数据库表。
    User 等子类继承它以后，字段才会进入各自的 SQLAlchemy Table metadata。
    """

    # UUID 在应用侧生成。这样对象在 flush/commit 前已经拥有稳定 ID，service
    # 可以立即用这个 ID 创建关联对象，也更适合在 Agent state 中只传递资源 ID。
    id: UUID = Field(default_factory=uuid4, primary_key=True)

    # Python 的 datetime 类型不能表达数据库列是否保存时区，因此通过
    # DateTime(timezone=True) 明确要求 PostgreSQL 使用带时区时间语义。
    #
    # 这里使用 sa_type，而不直接传 sa_column=Column(...)。公共基类会被多张
    # 表继承；如果复用一个具体 Column 对象，SQLAlchemy 会报“该列已经属于
    # 另一张表”。sa_type 会让 SQLModel 为每个子表分别创建独立 Column。
    created_at: datetime = Field(
        default_factory=utc_now,
        sa_type=UTCDateTime,
        nullable=False,
    )
    updated_at: datetime = Field(
        default_factory=utc_now,
        sa_type=UTCDateTime,
        nullable=False,
    )
