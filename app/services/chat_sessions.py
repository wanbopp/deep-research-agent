"""业务聊天会话 application service.

本模块位于 FastAPI route 与 ChatSessionRepository 之间，负责事务和层间模型转换。
它不执行 LangGraph、不读取 checkpoint，也不知道 HTTP 状态码；这样会话所有权规则
可以被普通 HTTP、后台任务或未来 Agent 工具复用，而不会绑定某一种传输协议。
"""

from uuid import UUID

from pydantic import ValidationError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import logger
from app.models import ChatSession
from app.repositories import ChatSessionRepository
from app.schemas.chat_session import (
    ChatSessionCreateRequest,
    ChatSessionListResponse,
    ChatSessionResponse,
)
from app.services.chat_session_ownership import ChatSessionNotFoundError
from app.services.cache import Cache, CacheUnavailableError
from app.services.chat_session_cache import (
    build_chat_session_list_cache_key,
    invalidate_chat_session_list_cache,
)


class ChatSessionServiceError(RuntimeError):
    """业务聊天会话可预期错误的基类."""


class ChatSessionService:
    """协调业务会话事务、所有权查询和公开响应转换.

    这里的 ``AsyncSession`` 是一次 HTTP 请求的数据库工作单元，不是业务
    ``ChatSession``。service 拥有 ``session.begin()``；Repository 只执行 SQL、
    ``flush`` 和 ``refresh``，不能自行 commit。
    """

    def __init__(
        self,
        session: AsyncSession,
        *,
        cache: Cache,
        cache_ttl_seconds: int,
    ) -> None:
        """保存请求级数据库 Session、共享缓存协议和 Repository.

        Args:
            session: 当前请求独享的 AsyncSession。它可以在同一请求中顺序执行
                认证查询和业务事务，但不能跨请求或并发任务共享。
            cache: lifespan 共享的缓存能力。service 只依赖 Cache Protocol，
                不知道当前实现来自 Redis 还是进程内存。
            cache_ttl_seconds: 成功列表响应的正整数生存时间。TTL 只是性能和
                最终一致性边界，不能替代写后主动失效。

        Raises:
            ValueError: ``cache_ttl_seconds`` 为零或负数。该配置错误应在装配阶段
                暴露，不能等到第一次 Redis 写入时才成为请求级 500。
        """
        if cache_ttl_seconds <= 0:
            raise ValueError("cache_ttl_seconds must be greater than 0")

        self._session = session
        self._chat_sessions = ChatSessionRepository(session)
        self._cache = cache
        self._cache_ttl_seconds = cache_ttl_seconds

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

        # 必须在 begin() 正常退出后才失效缓存。到达这里表示事务已经 commit，
        # 客户端不会拿到最终因 rollback 而不存在的 thread_id；缓存失败也不能
        # 反向撤销已经提交的 PostgreSQL 事实。
        response = self._to_response(chat_session)
        await invalidate_chat_session_list_cache(
            self._cache,
            user_id=user_id,
            reason="created",
        )
        return response

    async def list_owned(self, *, user_id: UUID) -> ChatSessionListResponse:
        """列出可信用户拥有的全部业务会话.

        Args:
            user_id: 当前认证用户 UUID。

        Returns:
            稳定排序的会话集合；没有会话时返回空数组语义，而不是 404。
        """
        cache_key = build_chat_session_list_cache_key(user_id)
        cached_response = await self._read_cached_list(
            key=cache_key,
            user_id=user_id,
        )
        if cached_response is not None:
            return cached_response

        # 只有 cache miss、缓存不可用或缓存值损坏时才访问数据库。PostgreSQL
        # 查询仍然使用 owner-scoped user_id，它才是事实来源和授权边界。
        async with self._session.begin():
            chat_sessions = await self._chat_sessions.list_by_user(user_id)

        response = ChatSessionListResponse(
            sessions=tuple(self._to_response(item) for item in chat_sessions),
        )

        # 事务正常退出后再写缓存，确保不会缓存尚未提交或随后回滚的读取视图。
        # 空列表同样是成功结果，可以避免“没有数据”时每次都查询 PostgreSQL。
        try:
            await self._cache.set(
                cache_key,
                response.model_dump_json(),
                ttl_seconds=self._cache_ttl_seconds,
            )
        except CacheUnavailableError:
            # RedisCache 已记录脱敏驱动错误。读取业务结果仍然成功，因此这里只
            # 采用 fail-open，不把性能层故障升级成 API 失败。
            logger.warning(
                "chat_session_list_cache_write_skipped",
                reason="backend_unavailable",
            )

        return response

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

    async def _read_cached_list(
        self,
        *,
        key: str,
        user_id: UUID,
    ) -> ChatSessionListResponse | None:
        """读取并校验列表缓存，任何不可信结果都回退为 miss.

        Args:
            key: 已由可信 user_id 构造的版本化摘要 key。
            user_id: 当前认证用户 UUID。仅在发现损坏值时用于重新构造并删除
                同一个列表 key，不写入日志。

        Returns:
            命中且通过 Pydantic schema 校验时返回列表响应；普通 miss、Redis
            不可用或缓存值损坏时返回 ``None``，调用方随后回源 PostgreSQL。

        Notes:
            Redis 中的数据不是可信对象。即使 key 正确，也必须执行
            ``model_validate_json``，防止旧 schema、手工写入或不完整值绕过
            ChatSessionListResponse 的字段和时间约束。
        """
        try:
            cached_value = await self._cache.get(key)
        except CacheUnavailableError:
            logger.warning(
                "chat_session_list_cache_read_fallback",
                reason="backend_unavailable",
            )
            return None

        if cached_value is None:
            return None

        try:
            return ChatSessionListResponse.model_validate_json(cached_value)
        except ValidationError as error:
            # 不记录 cached_value 或 ValidationError 文本，因为它们可能包含被
            # 手工写入的敏感内容。error_type 足以区分 schema 损坏分支。
            logger.warning(
                "chat_session_list_cache_read_fallback",
                reason="invalid_payload",
                error_type=type(error).__name__,
            )
            await invalidate_chat_session_list_cache(
                self._cache,
                user_id=user_id,
                reason="invalid_payload",
            )
            return None

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
    "ChatSessionService",
    "ChatSessionServiceError",
]
