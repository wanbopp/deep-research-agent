"""SQLModel/SQLAlchemy 异步数据库运行时工厂."""

from sqlalchemy import URL
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.core.config import Settings

ORM_CONNECTION_TIMEOUT_SECONDS = 5


def build_orm_database_url(config: Settings) -> URL:
    """构造不会因密码特殊字符而损坏的 SQLAlchemy URL.

    使用 ``URL.create`` 而不是手工拼接字符串，可以正确处理密码中的 ``@``、
    ``:`` 或 ``/``。URL 对象在普通字符串表示中也会隐藏密码，降低误记日志风险。
    """
    return URL.create(
        drivername="postgresql+psycopg",
        username=config.POSTGRES_USER,
        password=config.POSTGRES_PASSWORD,
        host=config.POSTGRES_HOST,
        port=config.POSTGRES_PORT,
        database=config.POSTGRES_DB,
    )


def create_orm_runtime(
    config: Settings,
) -> tuple[AsyncEngine, async_sessionmaker[AsyncSession]]:
    """构造惰性 AsyncEngine 和应用级 session factory.

    create_async_engine() 只创建运行时对象，不会立即连接 PostgreSQL，也不会创建
    数据库表。第一次 checkout connection 时才会建立真实网络连接。
    """
    if config.POSTGRES_ORM_POOL_SIZE < 1:
        raise ValueError("POSTGRES_ORM_POOL_SIZE must be at least 1")
    if config.POSTGRES_ORM_MAX_OVERFLOW < 0:
        raise ValueError("POSTGRES_ORM_MAX_OVERFLOW must not be negative")

    engine = create_async_engine(
        build_orm_database_url(config),
        # Checkpoint 8A 确认的 ORM 连接预算：5 个常驻连接 + 5 个临时连接。
        pool_size=config.POSTGRES_ORM_POOL_SIZE,
        max_overflow=config.POSTGRES_ORM_MAX_OVERFLOW,
        pool_timeout=ORM_CONNECTION_TIMEOUT_SECONDS,
        # 连接从池中取出时先做轻量存活检查，避免复用已被服务端关闭的旧连接。
        pool_pre_ping=True,
        connect_args={"connect_timeout": ORM_CONNECTION_TIMEOUT_SECONDS},
    )

    session_factory = async_sessionmaker(
        bind=engine,
        class_=AsyncSession,
        # commit 后继续访问已加载字段时不强制重新查询，适合 API/service 返回结果。
        expire_on_commit=False,
        # Repository 显式 flush，减少一次普通查询意外触发待写对象落库的可能性。
        autoflush=False,
    )
    return engine, session_factory
