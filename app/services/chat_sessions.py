"""业务聊天会话 application service.

本模块位于 FastAPI route 与 ChatSessionRepository 之间，负责事务和层间模型转换。
它不执行 LangGraph、不读取 checkpoint，也不知道 HTTP 状态码；这样会话所有权规则
可以被普通 HTTP、后台任务或未来 Agent 工具复用，而不会绑定某一种传输协议。
"""

from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from app.models import ChatSession
from app.repositories import ChatSessionRepository
from app.schemas.chat_session import (
    ChatSessionCreateRequest,
    ChatSessionListResponse,
    ChatSessionResponse,
)


class ChatSessionServiceError(RuntimeError):
    """业务聊天会话可预期错误的基类."""


class ChatSessionNotFoundError(ChatSessionServiceError):
    """当前用户作用域中不存在指定业务会话."""

    def __init__(self) -> None:
        """使用固定文本，避免泄漏相同 UUID 是否属于其他用户."""
        super().__init__("Chat session was not found")


class ChatSessionService:
    """协调业务会话事务、所有权查询和公开响应转换.

    这里的 ``AsyncSession`` 是一次 HTTP 请求的数据库工作单元，不是业务
    ``ChatSession``。service 拥有 ``session.begin()``；Repository 只执行 SQL、
    ``flush`` 和 ``refresh``，不能自行 commit。
    """

    def __init__(self, session: AsyncSession) -> None:
        """保存请求级数据库 Session 并构造 Repository.

        Args:
            session: 当前请求独享的 AsyncSession。它可以在同一请求中顺序执行
                认证查询和业务事务，但不能跨请求或并发任务共享。
        """
        self._session = session
        self._chat_sessions = ChatSessionRepository(session)

    async def create(
        self,
        *,
        user_id: UUID,
        request: ChatSessionCreateRequest,
    ) -> ChatSessionResponse:
        """创建属于可信用户的业务会话.

        Args:
            user_id: 从已验签 Bearer token 和数据库用户查询得到的可信 UUID，不能
                来自请求 body、Prompt、模型输出或工具参数。
            request: 已由 Pydantic 清理并验证标题的公开创建请求。

        Returns:
            已成功提交的会话响应。公开 thread_id 等于业务 ChatSession.id。

        Raises:
            RepositoryConflictError: 用户在认证后被并发删除等数据库约束冲突。
                该异常不会在这里伪装成普通 404，避免隐藏基础设施或竞态问题。
        """
        async with self._session.begin():
            chat_session = await self._chat_sessions.create(
                user_id=user_id,
                title=request.title,
            )

        # 必须在 begin() 正常退出后才返回。到达这里表示事务已经 commit，客户端
        # 不会拿到一个最终因 rollback 而不存在的 thread_id。
        return self._to_response(chat_session)

    async def list_owned(self, *, user_id: UUID) -> ChatSessionListResponse:
        """列出可信用户拥有的全部业务会话.

        Args:
            user_id: 当前认证用户 UUID。

        Returns:
            稳定排序的会话集合；没有会话时返回空数组语义，而不是 404。
        """
        async with self._session.begin():
            chat_sessions = await self._chat_sessions.list_by_user(user_id)

        return ChatSessionListResponse(
            sessions=tuple(self._to_response(item) for item in chat_sessions),
        )

    async def get_owned(
        self,
        *,
        session_id: UUID,
        user_id: UUID,
    ) -> ChatSessionResponse:
        """在用户作用域内读取一条业务会话.

        Args:
            session_id: 客户端路径中的公开业务会话 UUID。
            user_id: 当前认证用户的可信 UUID。

        Returns:
            同时匹配 session_id 和 user_id 的会话响应。

        Raises:
            ChatSessionNotFoundError: 会话不存在或属于其他用户。两种情况故意使用
                相同异常，防止接口成为探测其他用户资源 UUID 的旁路。
        """
        async with self._session.begin():
            chat_session = await self._chat_sessions.get_by_id(
                session_id,
                user_id=user_id,
            )

        if chat_session is None:
            raise ChatSessionNotFoundError
        return self._to_response(chat_session)

    @staticmethod
    def _to_response(chat_session: ChatSession) -> ChatSessionResponse:
        """把数据库实体转换为不携带 ORM Session 的公开响应.

        Args:
            chat_session: Repository 已加载完成的业务 ORM 实体。

        Returns:
            只含公开 thread_id、标题和审计时间的不可变 Pydantic 模型。

        Notes:
            不直接把 ORM 对象交给 route，避免未来新增 user_id、删除标记或其他
            内部字段时被 response serialization 意外公开。
        """
        return ChatSessionResponse(
            thread_id=chat_session.id,
            title=chat_session.title,
            created_at=chat_session.created_at,
            updated_at=chat_session.updated_at,
        )


__all__ = [
    "ChatSessionNotFoundError",
    "ChatSessionService",
    "ChatSessionServiceError",
]
