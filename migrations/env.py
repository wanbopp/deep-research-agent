"""Alembic environment for the application-owned SQLModel tables."""

import asyncio
import os
import selectors
import sys
from collections.abc import Mapping
from logging.config import fileConfig
from typing import Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import URL, pool
from sqlalchemy.engine import Connection, make_url
from sqlalchemy.ext.asyncio import create_async_engine
from sqlmodel import SQLModel

from alembic import context
import app.models  # noqa: F401  # 导入即把所有 table model 注册到 SQLModel.metadata。
from app.core.config import settings
from app.infrastructure.database import build_orm_database_url
from app.models.base import UTCDateTime

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = SQLModel.metadata

# 这是 Alembic 的表所有权白名单，不是“目前碰巧存在的表名单”。
# 新增业务表时必须先人工确认所有权，再显式更新这里。
ALEMBIC_MANAGED_TABLES = frozenset(
    {
        "users",
        "chat_sessions",
        "documents",
        "memories",
        "research_tasks",
    }
)


def _validate_target_metadata() -> None:
    """拒绝 metadata 与 Alembic 所有权白名单悄悄发生偏移."""
    metadata_tables = frozenset(target_metadata.tables)
    if metadata_tables != ALEMBIC_MANAGED_TABLES:
        raise RuntimeError(
            "Alembic managed-table list does not match SQLModel metadata: "
            f"missing={sorted(ALEMBIC_MANAGED_TABLES - metadata_tables)}, "
            f"unexpected={sorted(metadata_tables - ALEMBIC_MANAGED_TABLES)}"
        )


_validate_target_metadata()


def _database_url() -> URL:
    """读取迁移专用覆盖地址，默认复用应用数据库配置.

    ALEMBIC_DATABASE_URL 只用于隔离 migration smoke。普通运行不会把 URL 写回
    alembic.ini，因此配置文件、日志和版本库都不会保存真实密码。
    """
    override = os.getenv("ALEMBIC_DATABASE_URL")
    return make_url(override) if override else build_orm_database_url(settings)


def include_name(
    name: str | None,
    type_: str,
    parent_names: Mapping[str, str | None],
) -> bool:
    """在反射前过滤数据库对象，避免扫描或删除外部系统表."""
    del parent_names
    if type_ == "schema":
        return name in {None, "public"}
    if type_ == "table":
        return name in ALEMBIC_MANAGED_TABLES
    return True


def include_object(
    object_: Any,
    name: str | None,
    type_: str,
    reflected: bool,
    compare_to: Any,
) -> bool:
    """再次限制 metadata 和反射对象，只允许 Alembic 拥有的表."""
    del object_, reflected, compare_to
    return type_ != "table" or name in ALEMBIC_MANAGED_TABLES


def render_item(type_: str, object_: Any, autogen_context: Any) -> str | bool:
    """把自定义数据库类型渲染成 migration 可独立执行的代码."""
    if type_ == "type" and isinstance(object_, UTCDateTime):
        return "sa.DateTime(timezone=True)"
    if type_ == "type" and isinstance(object_, Vector):
        # pgvector 不是 SQLAlchemy 内置类型，autogenerate 必须同时写入 import；
        # 否则生成的 migration 虽然出现 Vector(1536)，执行时却找不到名称。
        autogen_context.imports.add("from pgvector.sqlalchemy import Vector")
        return f"Vector({object_.dim})"
    return False


def _configure_context(**kwargs: Any) -> None:
    """集中配置每种迁移模式共享的比较和所有权规则."""
    context.configure(
        target_metadata=target_metadata,
        include_name=include_name,
        include_object=include_object,
        render_item=render_item,
        compare_type=True,
        compare_server_default=True,
        **kwargs,
    )


def run_migrations_offline() -> None:
    """只生成 SQL 文本，不建立数据库连接."""
    _configure_context(
        url=_database_url(),
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    """在 Alembic 提供的同步桥接连接中执行迁移."""
    _configure_context(connection=connection)

    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    """使用异步 psycopg Engine 执行在线迁移，并确保最终释放连接."""
    connectable = create_async_engine(_database_url(), poolclass=pool.NullPool)
    try:
        async with connectable.connect() as connection:
            await connection.run_sync(do_run_migrations)
    finally:
        await connectable.dispose()


def _selector_loop_factory() -> asyncio.AbstractEventLoop:
    """在 Windows 上创建 psycopg async 支持的 Selector event loop."""
    return asyncio.SelectorEventLoop(selectors.SelectSelector())


def run_migrations_online() -> None:
    """运行在线迁移，兼容 Windows 与其他平台的事件循环实现."""
    if sys.platform == "win32":
        with asyncio.Runner(loop_factory=_selector_loop_factory) as runner:
            runner.run(run_async_migrations())
        return
    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
