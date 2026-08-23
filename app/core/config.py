"""DeepSearch 配置管理模块.

设计思路：
    - 多环境支持（dev/staging/prod/test）
    - .env 文件按优先级加载
    - 所有配置项有安全默认值
    - 环境特定覆盖
"""

import os
from enum import Enum
from pathlib import Path

from dotenv import load_dotenv


# ============================================================
# 环境枚举
# 双继承 str + Enum：既可比较又可序列化，enum.value 用于拼接 .env 文件名
# ============================================================
class Environment(str, Enum):
    """应用支持的运行环境."""

    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TEST = "test"


# ============================================================
# 环境检测函数
# 从环境变量 APP_ENV 读取，默认 development
# 使用 match-case（Python 3.10+）做模式匹配
# 无法识别的值统一回退到 development
# ============================================================
def get_environment() -> Environment:
    """读取 APP_ENV，并转换为受支持的运行环境."""
    match os.getenv("APP_ENV", "development").lower():
        case "production" | "prod":
            return Environment.PRODUCTION
        case "staging" | "stage":
            return Environment.STAGING
        case "test":
            return Environment.TEST
        case _:
            return Environment.DEVELOPMENT


# ============================================================
# .env 文件加载
# 按优先级顺序尝试加载，找到第一个存在的就停
# 优先级：.env.{env}.local > .env.{env} > .env.local > .env
#
# 路径计算：config.py 在 app/core/ 下，需要向上 3 级到项目根目录
#   __file__              → .../deep-research/app/core/config.py
#   dirname 第 1 次       → .../deep-research/app/core
#   dirname 第 2 次       → .../deep-research/app
#   dirname 第 3 次       → .../deep-research          ← 根目录
# ============================================================
def load_env_file() -> str | None:
    """按优先级加载首个存在的环境变量文件.

    Returns:
        成功加载的文件路径；没有可用文件时返回 None。
    """
    env = get_environment()
    print(f"Loading environment: {env.value}")

    # 向上 3 级到达项目根目录 deep-research/
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

    # 按优先级构造文件列表（高优先级在前）
    # 注意：用 env.value 拿到字符串值（"development"），而非枚举对象本身
    env_files = [
        os.path.join(base_dir, f".env.{env.value}.local"),  # 个人本地覆盖，加入 .gitignore
        os.path.join(base_dir, f".env.{env.value}"),  # 团队共享的环境配置，提交到 git
        os.path.join(base_dir, ".env.local"),  # 个人全局覆盖
        os.path.join(base_dir, ".env"),  # 项目默认兜底
    ]

    # 遍历：找到第一个存在的文件就加载，然后返回
    for env_file in env_files:
        if os.path.isfile(env_file):
            load_dotenv(dotenv_path=env_file)  # 将配置加载到环境变量中
            print(f"Loaded environment from {env_file}")
            return env_file

    # 没有任何 .env 文件存在，回退到系统环境变量
    return None


# 模块加载时执行
ENV_FILE = load_env_file()


# ============================================================
# Settings 配置类
#
# 配置分组：
#   - 应用基本信息（PROJECT_NAME, VERSION, DEBUG）
#   - LLM 配置（API_KEY, BASE_URL, DEFAULT_MODEL, TEMPERATURE, MAX_RETRIES, TOTAL_TIMEOUT）
#   - 数据库配置（POSTGRES_HOST/PORT/DB/USER/PASSWORD）
#   - Neo4j 配置（NEO4J_URI/USER/PASSWORD）
#   - Redis 配置（REDIS_HOST/PORT）
#   - JWT 配置（SECRET_KEY, ALGORITHM, EXPIRE_DAYS）
#   - 限流配置（RATE_LIMIT_DEFAULT）
#   - 日志配置（LOG_LEVEL, LOG_FORMAT）
#   - Langfuse 可观测性（TRACING_ENABLED, PUBLIC_KEY, SECRET_KEY, HOST）
#
# 规则：
#   - 所有值从 os.getenv() 读取，带默认值，永远不崩溃
#   - 数值类型用 int()/float() 转换
#   - 布尔类型用 .lower() in ("true", "1", "yes") 判断
#   - __init__ 末尾调用 apply_environment_settings() 覆盖环境特定配置
# ============================================================
class Settings:
    """集中保存应用运行所需的配置项."""

    def __init__(self):
        """从环境变量加载配置，并应用当前环境的默认覆盖."""
        self.ENVIRONMENT = get_environment()

        # ---- 应用基本信息 ----
        self.PROJECT_NAME = os.getenv("PROJECT_NAME", "DeepResearch")
        self.VERSION = os.getenv("VERSION", "0.1.0")
        self.DESCRIPTION = os.getenv(
            "DESCRIPTION",
            "Intelligent research platform with LangGraph multi-agent orchestration",
        )
        self.API_V1_STR = os.getenv("API_V1_STR", "/api/v1")
        self.DEBUG = os.getenv("DEBUG", "false").lower() in ("true", "1", "yes")
        # 如果 元素在 in （） 则返回 true

        # ---- CORS ----
        origins = os.getenv("ALLOWED_ORIGINS", "*")
        self.ALLOWED_ORIGINS = [o.strip() for o in origins.split(",") if o.strip()]

        # ---- LLM 配置 ----
        self.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
        self.OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL") or None
        self.DEFAULT_LLM_MODEL = os.getenv("DEFAULT_LLM_MODEL", "gpt-4o-mini")
        self.DEFAULT_LLM_TEMPERATURE = float(os.getenv("DEFAULT_LLM_TEMPERATURE", "0.2"))
        self.MAX_LLM_CALL_RETRIES = int(os.getenv("MAX_LLM_CALL_RETRIES", "3"))
        self.LLM_TOTAL_TIMEOUT = int(os.getenv("LLM_TOTAL_TIMEOUT", "60"))
        self.MAX_TOKENS = int(os.getenv("MAX_TOKENS", "2000"))

        # ---- PostgreSQL 数据库 ----
        self.POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
        self.POSTGRES_PORT = int(os.getenv("POSTGRES_PORT", "5432"))
        self.POSTGRES_DB = os.getenv("POSTGRES_DB", "deep_research_db")
        self.POSTGRES_USER = os.getenv("POSTGRES_USER", "postgres")
        self.POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "postgres")
        # PostgreSQL 目前有两类独立消费者，必须分别设置连接预算：
        # 1. 原生 psycopg pool：供依赖探针和后续 LangGraph checkpointer 使用；
        # 2. SQLAlchemy pool：供 SQLModel Repository 和业务事务使用。
        #
        # 三个默认值采用 Checkpoint 8A 确认的 5/5/5 预算。单进程在突发情况下
        # 最多占用 5 + 5 + 5 = 15 个 PostgreSQL 连接，不能再把同一组配置复制
        # 给两套连接池，否则 worker 数量增加时会迅速耗尽数据库连接。
        self.POSTGRES_PSYCOPG_POOL_SIZE = int(os.getenv("POSTGRES_PSYCOPG_POOL_SIZE", "5"))
        self.POSTGRES_ORM_POOL_SIZE = int(os.getenv("POSTGRES_ORM_POOL_SIZE", "5"))
        self.POSTGRES_ORM_MAX_OVERFLOW = int(os.getenv("POSTGRES_ORM_MAX_OVERFLOW", "5"))

        # ---- Neo4j 图数据库 ----
        self.NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
        self.NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "neo4j")

        # ---- Redis 缓存 ----
        self.REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
        self.REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
        self.REDIS_DB = int(os.getenv("REDIS_DB", "0"))
        self.REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", "")
        self.CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", "60"))

        # ---- JWT 认证 ----
        self.JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "change-me-in-production")
        self.JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
        self.JWT_ACCESS_TOKEN_EXPIRE_DAYS = int(os.getenv("JWT_ACCESS_TOKEN_EXPIRE_DAYS", "30"))

        # ---- 限流 ----
        rate_limit = os.getenv("RATE_LIMIT_DEFAULT", "200 per day,50 per hour")
        self.RATE_LIMIT_DEFAULT = [r.strip() for r in rate_limit.split(",") if r.strip()]

        # ---- 日志 ----
        self.LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
        self.LOG_FORMAT = os.getenv("LOG_FORMAT", "json")  # "json" 或 "console"
        self.LOG_DIR = Path(os.getenv("LOG_DIR", "logs"))

        # ---- Langfuse 可观测性 ----
        self.LANGFUSE_TRACING_ENABLED = os.getenv("LANGFUSE_TRACING_ENABLED", "false").lower() in (
            "true",
            "1",
            "yes",
        )
        self.LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY", "")
        self.LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY", "")
        self.LANGFUSE_HOST = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")

        # 最后：根据环境覆盖特定配置
        self.apply_environment_settings()

    def apply_environment_settings(self):
        """根据环境覆盖配置值.

        仅当对应的环境变量没有被显式设置时才覆盖，
        这样用户可以通过 .env 文件中的 DEBUG=false 来阻止 dev 环境自动设 DEBUG=True。
        """
        env_overrides: dict[Environment, dict] = {
            Environment.DEVELOPMENT: {
                "DEBUG": True,
                "LOG_LEVEL": "DEBUG",
                "LOG_FORMAT": "console",
                "RATE_LIMIT_DEFAULT": ["1000 per day", "200 per hour"],
            },
            Environment.STAGING: {
                "DEBUG": False,
                "LOG_LEVEL": "INFO",
                "RATE_LIMIT_DEFAULT": ["500 per day", "100 per hour"],
            },
            Environment.PRODUCTION: {
                "DEBUG": False,
                "LOG_LEVEL": "WARNING",
                "RATE_LIMIT_DEFAULT": ["200 per day", "50 per hour"],
            },
            Environment.TEST: {
                "DEBUG": True,
                "LOG_LEVEL": "DEBUG",
                "LOG_FORMAT": "console",
                "RATE_LIMIT_DEFAULT": ["1000 per day", "1000 per hour"],
            },
        }

        overrides = env_overrides.get(self.ENVIRONMENT, {})
        for key, value in overrides.items():
            # 只在该环境变量没有被显式设置时才覆盖
            if key not in os.environ:
                setattr(self, key, value)


# 创建全局单例 —— 模块导入时自动执行
settings = Settings()
