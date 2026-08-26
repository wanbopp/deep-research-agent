"""应用业务数据库模型包.

集中导出模型有两个作用：
1. 业务代码可以从 ``app.models`` 使用稳定的导入路径；
2. Alembic 以后只要导入这个包，就能让所有 table model 注册到
   ``SQLModel.metadata``，避免迁移时漏掉某张业务表。
"""

from app.models.base import UUIDTimestampModel, utc_now
from app.models.chat_session import ChatSession, ChatSessionStatus
from app.models.document import Document, DocumentStatus
from app.models.research_task import ResearchTask, ResearchTaskStatus
from app.models.user import User

__all__ = [
    "ChatSession",
    "ChatSessionStatus",
    "Document",
    "DocumentStatus",
    "ResearchTask",
    "ResearchTaskStatus",
    "UUIDTimestampModel",
    "User",
    "utc_now",
]
