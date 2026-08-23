"""Construct lazy infrastructure clients for FastAPI lifespan ownership."""

from typing import Any

from neo4j import AsyncGraphDatabase
from psycopg import AsyncConnection
from psycopg.conninfo import make_conninfo
from psycopg_pool import AsyncConnectionPool
from redis.asyncio import Redis

from app.core.config import Settings
from app.infrastructure.resources import ApplicationResources

CONNECTION_TIMEOUT_SECONDS = 5.0


def create_application_resources(config: Settings) -> ApplicationResources:
    """根据应用配置构造惰性基础设施客户端."""
    # 使用 make_conninfo() 构造 PostgreSQL 连接信息。
    # 传入 host、port、dbname、user、password 和 connect_timeout。
    postgres_conninfo = make_conninfo(
        host=config.POSTGRES_HOST,
        port=config.POSTGRES_PORT,
        dbname=config.POSTGRES_DB,
        user=config.POSTGRES_USER,
        password=config.POSTGRES_PASSWORD,
        connect_timeout=int(CONNECTION_TIMEOUT_SECONDS),
    )

    # 创建 AsyncConnectionPool。
    #
    # 关键参数：
    # - conninfo=postgres_conninfo
    # - min_size=0
    # - max_size=config.POSTGRES_POOL_SIZE
    #   + config.POSTGRES_MAX_OVERFLOW
    # - open=False
    #
    # open=False 很重要：factory 只构造资源，
    # 真正打开连接池属于 lifespan startup。
    postgres_pool: AsyncConnectionPool[AsyncConnection[Any]] = AsyncConnectionPool(
        conninfo=postgres_conninfo,
        min_size=0,
        max_size=config.POSTGRES_POOL_SIZE + config.POSTGRES_MAX_OVERFLOW,
        open=False,
        timeout=CONNECTION_TIMEOUT_SECONDS,
    )

    # 使用 AsyncGraphDatabase.driver() 创建惰性 Neo4j Driver。
    #
    # 参数：
    # - config.NEO4J_URI
    # - auth=(config.NEO4J_USER, config.NEO4J_PASSWORD)
    # - connection_timeout=CONNECTION_TIMEOUT_SECONDS
    neo4j_driver = AsyncGraphDatabase.driver(
        config.NEO4J_URI,
        auth=(config.NEO4J_USER, config.NEO4J_PASSWORD),
        connection_timeout=CONNECTION_TIMEOUT_SECONDS,
        connection_acquisition_timeout=CONNECTION_TIMEOUT_SECONDS,
    )

    # 创建异步 Redis client。
    #
    # 参数：
    # - host、port、db
    # - password=config.REDIS_PASSWORD or None
    # - socket_connect_timeout
    # - socket_timeout
    # - decode_responses=True
    redis_client = Redis(
        host=config.REDIS_HOST,
        port=config.REDIS_PORT,
        db=config.REDIS_DB,
        password=config.REDIS_PASSWORD or None,
        socket_connect_timeout=CONNECTION_TIMEOUT_SECONDS,
        socket_timeout=CONNECTION_TIMEOUT_SECONDS,
        decode_responses=True,
    )

    # 把三个客户端放入 ApplicationResources。
    return ApplicationResources(
        postgres_pool=postgres_pool,
        neo4j_driver=neo4j_driver,
        redis_client=redis_client,
    )
