"""Shared infrastructure resources owned by the FastAPI lifespan."""

from dataclasses import dataclass
from typing import Any

from neo4j import AsyncDriver
from psycopg import AsyncConnection
from psycopg_pool import AsyncConnectionPool
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker


@dataclass(frozen=True, slots=True)
class ApplicationResources:
    """FastAPI lifespan 创建并拥有的共享基础设施资源.

    frozen=True 防止运行期间把 postgres_pool、orm_engine、session factory、
    neo4j_driver 或 redis_client 引用意外替换掉。

    这不会冻结客户端内部状态；连接池和客户端仍然能够正常维护连接。
    Engine 管连接，session_maker 造 Session，Session 管一次业务事务。
    """

    postgres_pool: AsyncConnectionPool[AsyncConnection[Any]]
    orm_engine: AsyncEngine  # 驱动、连接池、SQL 执行入口
    orm_session_factory: async_sessionmaker[AsyncSession]  # 创建session的工厂
    neo4j_driver: AsyncDriver
    redis_client: Redis
