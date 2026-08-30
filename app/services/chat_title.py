"""会话自动命名、数据库租约和后台提交边界."""

import asyncio
from collections.abc import Sequence
from datetime import datetime, timedelta
from typing import Protocol
from uuid import UUID, uuid4

from langchain_core.messages import HumanMessage, SystemMessage
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.agents.prompts.loader import load_prompt_artifact, render_prompt_input
from app.core.logging import logger
from app.models import utc_now
from app.repositories import ChatSessionRepository
from app.schemas.chat_title import ChatSessionTitleResult
from app.services.background_tasks import BackgroundTaskSubmitter
from app.services.cache import Cache
from app.services.chat_session_cache import invalidate_chat_session_list_cache
from app.services.llm.service import LLMService

_CHAT_TITLE_TASK_NAME = "chat-session-title"
_MAX_TITLE_SOURCE_MESSAGE_LENGTH = 2000


class ChatTitleGenerator(Protocol):
    """定义把一轮已完成聊天概括成展示标题的能力."""

    async def generate(
        self,
        *,
        user_message: str,
        assistant_message: str,
    ) -> str:
        """根据有界对话文本返回已经过 schema 校验的标题."""
        ...


class ChatTitleWriter(Protocol):
    """定义 ChatService 可以调用的非阻塞会话命名边界."""

    def submit_turn(
        self,
        *,
        user_id: UUID,
        source_thread_id: UUID,
        user_message: str,
        assistant_message: str,
    ) -> None:
        """提交已完成对话的自动命名任务，不等待数据库或模型结果."""
        ...


class LLMChatTitleGenerator:
    """通过统一 LLMService 的真实 structured output 生成标题."""

    def __init__(
        self,
        llm_service: LLMService,
        *,
        aliases: Sequence[str],
    ) -> None:
        """保存无请求状态的模型服务与 fallback alias.

        Args:
            llm_service: 统一提供 timeout、retry、fallback 和结构化校验的服务。
            aliases: 按优先级尝试的稳定模型别名。

        Raises:
            ValueError: aliases 是字符串或空序列。
        """
        if isinstance(aliases, str) or not aliases:
            raise ValueError("aliases must contain at least one model alias")
        self._llm_service = llm_service
        self._aliases = tuple(aliases)

    async def generate(
        self,
        *,
        user_message: str,
        assistant_message: str,
    ) -> str:
        """发送最小化对话数据，并返回结构化校验后的单行标题.

        输入在进入 JSON 前截断，限制成本和暴露面。JSON 只是低信任数据载体，
        SystemMessage 才定义任务规则；完整 Prompt 和正文都不进入日志。
        """
        clean_user_message = user_message.strip()
        clean_assistant_message = assistant_message.strip()
        if not clean_user_message or not clean_assistant_message:
            raise ValueError("chat title source messages must not be empty")

        prompt = load_prompt_artifact("chat_title")
        payload = render_prompt_input(
            "chat_title",
            user_message=clean_user_message[:_MAX_TITLE_SOURCE_MESSAGE_LENGTH],
            assistant_message=clean_assistant_message[:_MAX_TITLE_SOURCE_MESSAGE_LENGTH],
        )
        result = await self._llm_service.call_structured(
            (
                SystemMessage(content=prompt.content),
                HumanMessage(content=payload),
            ),
            response_model=ChatSessionTitleResult,
            aliases=self._aliases,
            overrides={"temperature": 0.0},
            prompt=prompt,
        )
        return result.title


class BackgroundChatTitleWriter:
    """通过 PostgreSQL 租约把自动命名变成非阻塞、跨 worker 单赢家任务."""

    def __init__(
        self,
        *,
        generator: ChatTitleGenerator,
        session_factory: async_sessionmaker[AsyncSession],
        cache: Cache,
        task_submitter: BackgroundTaskSubmitter,
        claim_lease_seconds: float,
    ) -> None:
        """保存可跨请求共享且不含当前用户状态的依赖.

        Args:
            generator: 只有成功 claim 后才调用的真实标题生成边界。
            session_factory: lifespan 拥有的 ORM 工厂；每个数据库阶段创建独立
                AsyncSession，绝不把请求级 Session 交给后台任务。
            cache: 标题提交后用于尽力失效会话列表缓存的共享协议。
            task_submitter: 同时跟踪记忆与标题任务的进程内生命周期组件。
            claim_lease_seconds: worker 崩溃后允许其他 worker 接管的正数秒数。

        Raises:
            ValueError: claim_lease_seconds 小于或等于零。
        """
        if claim_lease_seconds <= 0:
            raise ValueError("claim_lease_seconds must be greater than 0")
        self._generator = generator
        self._session_factory = session_factory
        self._cache = cache
        self._task_submitter = task_submitter
        self._claim_lease_seconds = claim_lease_seconds

    def submit_turn(
        self,
        *,
        user_id: UUID,
        source_thread_id: UUID,
        user_message: str,
        assistant_message: str,
    ) -> None:
        """捕获不可变输入并提交一次自动命名尝试.

        该方法只是提交边界。重复轮次和多个 worker 都可以调用它，真正的单次
        付费保证来自后台任务开始时的 PostgreSQL 原子 claim。
        """
        clean_user_message = user_message.strip()
        clean_assistant_message = assistant_message.strip()
        if not clean_user_message or not clean_assistant_message:
            raise ValueError("background title messages must not be empty")

        async def claim_generate_and_store() -> None:
            """按 claim、模型调用和条件提交三个阶段执行命名工作流."""
            claim_token = uuid4()
            claimed_at = utc_now()
            stale_before = claimed_at - timedelta(seconds=self._claim_lease_seconds)

            claimed = await self._claim(
                session_id=source_thread_id,
                user_id=user_id,
                claim_token=claim_token,
                claimed_at=claimed_at,
                stale_before=stale_before,
            )
            if not claimed:
                # 普通跳过不是错误：可能已有 worker 正在生成、标题已完成、用户
                # 自定义了标题，或者会话已进入删除流程。
                logger.info("chat_title_generation_skipped", reason="not_claimed")
                return

            try:
                title = await self._generator.generate(
                    user_message=clean_user_message,
                    assistant_message=clean_assistant_message,
                )
                completed = await self._complete(
                    session_id=source_thread_id,
                    user_id=user_id,
                    claim_token=claim_token,
                    title=title,
                )
                if not completed:
                    logger.info("chat_title_generation_skipped", reason="claim_lost")
                    return

                # PostgreSQL 事务已经提交后才删除缓存。Redis 故障只能造成短暂
                # 陈旧，不能回滚已经成为权威事实的新标题。
                await invalidate_chat_session_list_cache(
                    self._cache,
                    user_id=user_id,
                    reason="title_generated",
                )
                logger.info("chat_title_generation_completed")
            except asyncio.CancelledError:
                # 即使失败后后续聊天仍旧会再次触发
                await self._release_safely(
                    session_id=source_thread_id,
                    user_id=user_id,
                    claim_token=claim_token,
                    reason="cancelled",
                )
                raise
            except Exception:
                await self._release_safely(
                    session_id=source_thread_id,
                    user_id=user_id,
                    claim_token=claim_token,
                    reason="failed",
                )
                raise

        self._task_submitter.submit(
            claim_generate_and_store,
            name=_CHAT_TITLE_TASK_NAME,
        )

    async def _claim(
        self,
        *,
        session_id: UUID,
        user_id: UUID,
        claim_token: UUID,
        claimed_at: datetime,
        stale_before: datetime,
    ) -> bool:
        """在独立短事务中原子申请租约."""
        async with self._session_factory() as session:
            async with session.begin():
                return await ChatSessionRepository(session).claim_title_generation(
                    session_id,
                    user_id=user_id,
                    claim_token=claim_token,
                    claimed_at=claimed_at,
                    stale_before=stale_before,
                )

    async def _complete(
        self,
        *,
        session_id: UUID,
        user_id: UUID,
        claim_token: UUID,
        title: str,
    ) -> bool:
        """在新事务中按 owner、默认标题和 token 条件提交结果."""
        async with self._session_factory() as session:
            async with session.begin():
                return await ChatSessionRepository(session).complete_title_generation(
                    session_id,
                    user_id=user_id,
                    claim_token=claim_token,
                    title=title,
                    completed_at=utc_now(),
                )

    async def _release_safely(
        self,
        *,
        session_id: UUID,
        user_id: UUID,
        claim_token: UUID,
        reason: str,
    ) -> None:
        """尽力释放当前 token；失败时仍依赖租约超时恢复."""

        async def release() -> None:
            async with self._session_factory() as session:
                async with session.begin():
                    await ChatSessionRepository(session).release_title_generation_claim(
                        session_id,
                        user_id=user_id,
                        claim_token=claim_token,
                    )

        try:
            # shield 防止外层取消在释放 SQL 执行到一半时再次打断。若数据库已经
            # 不可用，租约时间仍提供最终恢复路径。
            await asyncio.shield(release())
        except Exception as error:
            logger.warning(
                "chat_title_claim_release_failed",
                reason=reason,
                error_type=type(error).__name__,
            )


__all__ = [
    "BackgroundChatTitleWriter",
    "ChatTitleGenerator",
    "ChatTitleWriter",
    "LLMChatTitleGenerator",
]
