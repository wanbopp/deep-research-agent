"""Construct lazy infrastructure clients for FastAPI lifespan ownership."""

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from neo4j import AsyncGraphDatabase
from psycopg import AsyncConnection
from psycopg.conninfo import make_conninfo
from psycopg_pool import AsyncConnectionPool
from psycopg.rows import DictRow, dict_row
from redis.asyncio import Redis

from app.core.config import Settings
from app.infrastructure.database import create_orm_runtime
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
    # - max_size=config.POSTGRES_PSYCOPG_POOL_SIZE
    # - open=False
    #
    # open=False 很重要：factory 只构造资源，
    # 真正打开连接池属于 lifespan startup。
    postgres_pool: AsyncConnectionPool[AsyncConnection[DictRow]] = AsyncConnectionPool(
        conninfo=postgres_conninfo,
        # 这组连接同时服务 dependency probe 与 LangGraph checkpointer：
        # - autocommit=True 允许 saver.setup() 执行 CREATE INDEX CONCURRENTLY；
        # - prepare_threshold=0 与 AsyncPostgresSaver.from_conn_string() 的正式
        #   连接约定一致，避免 checkpoint 动态 SQL 的 prepared statement 问题；
        # - dict_row 让 pool 的行类型与 saver 读取 checkpoint 的 DictRow 一致。
        # Saver 自己仍会为每个 cursor 显式声明 dict_row，这里同时固定 pool 类型，
        # 方便 readiness 和未来其他原生 psycopg 调用得到一致行为。
        kwargs={
            "autocommit": True,
            "prepare_threshold": 0,
            "row_factory": dict_row,
        },
        min_size=0,
        max_size=config.POSTGRES_PSYCOPG_POOL_SIZE,
        open=False,
        timeout=CONNECTION_TIMEOUT_SECONDS,
    )

    # AsyncPostgresSaver 构造时会记录当前运行中的 event loop，但不会连接数据库
    # 或创建表。create_application_resources 因而必须由 async lifespan 调用，
    # 不能放在模块导入期。连接池 open 和 saver.setup 仍由 lifespan 按顺序负责。
    # Checkpoint 保存的是跨进程、跨版本的长期状态，因此研究图只写入 JSON 可表达
    # 的数据，不要求 saver 动态导入项目里的 Pydantic 类。节点入口再执行校验，既
    # 保留类型安全，也降低应用升级后旧 checkpoint 无法恢复的风险。
    checkpointer = AsyncPostgresSaver(postgres_pool)

    # SQLModel Repository 使用 SQLAlchemy AsyncEngine，而 LangGraph checkpointer
    # 和依赖探针继续使用上面的原生 psycopg pool。两套池职责不同、预算独立。
    # 此处只构造 Engine/sessionmaker，不建立连接，也不执行任何 DDL。
    orm_engine, orm_session_factory = create_orm_runtime(config)

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

    # 把所有应用级共享客户端放入 ApplicationResources。
    # 注意这里只保存 sessionmaker，绝不能创建一个全局 AsyncSession 放进 app.state。
    return ApplicationResources(
        postgres_pool=postgres_pool,
        checkpointer=checkpointer,
        orm_engine=orm_engine,
        orm_session_factory=orm_session_factory,
        neo4j_driver=neo4j_driver,
        redis_client=redis_client,
    )
