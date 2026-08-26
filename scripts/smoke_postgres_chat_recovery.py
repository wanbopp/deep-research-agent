"""验证普通聊天历史与 HITL 恢复跨运行时重建的正确性.

Checkpoint 10C 已证明 production Graph 绑定了 lifespan 的 PostgreSQL saver。本
smoke 继续验证 10D 的行为结果：先用 lifespan A 写入普通历史并停在 ask_human
interrupt，再完整关闭 A；随后为同一个随机数据库创建全新的 lifespan B，确认 B
在任何新模型调用前已经能读取 A 的状态，并最终完成普通追问和 HITL resume。

脚本不会输出 API key、连接串、数据库名、用户 UUID、thread ID、Prompt、ToolCall ID、
checkpoint 正文或模型完整响应；最终只输出对象身份、状态结构、恢复结果、清理状态和
耗时等安全摘要。
"""

import asyncio
import json
import os
import selectors
from pathlib import Path
from time import perf_counter
from unittest.mock import patch
from uuid import UUID, uuid4

import psycopg
from alembic import command
from alembic.config import Config
from fastapi import FastAPI
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.types import Interrupt, StateSnapshot
from psycopg import sql
from psycopg.conninfo import make_conninfo

from app.agents.chat.graph import ChatGraph
from app.agents.chat.runtime import create_chat_runtime
from app.agents.chat.tools.ask_human import ask_human
from app.core.config import Settings, settings
from app.infrastructure.database import build_orm_database_url
import app.infrastructure.lifespan as lifespan_module
from app.infrastructure.lifespan import (
    get_application_chat_service,
    get_application_resources,
    lifespan,
)
from app.infrastructure.resources import ApplicationResources
from app.models.chat_session import ChatSession
from app.models.user import User
from app.schemas.chat import ChatRequest, ChatResponse, ChatResumeRequest
from app.services.chat import ChatInterrupt, ChatService

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONNECTION_TIMEOUT_SECONDS = 10
TOTAL_TIMEOUT_SECONDS = 360.0

MEMORY_ACK = "POSTGRES_MEMORY_STORED"
HITL_FINAL_REPLY = "POSTGRES_HITL_RESUMED_OK"
HUMAN_RESPONSE = "approved"


def _elapsed_ms(started_at: float) -> float:
    """计算已用毫秒数，不暴露 provider 或数据库细节."""
    return round((perf_counter() - started_at) * 1000, 2)


def _conninfo(database: str) -> str:
    """构造仅供本 smoke 进程使用的 psycopg 管理员连接串.

    Args:
        database: 已存在的 PostgreSQL 数据库，用于执行 CREATE/DROP DATABASE。

    Returns:
        仅在本 smoke 进程内持有的连接串，调用方不得记录该值。
    """
    return make_conninfo(
        host=settings.POSTGRES_HOST,
        port=settings.POSTGRES_PORT,
        dbname=database,
        user=settings.POSTGRES_USER,
        password=settings.POSTGRES_PASSWORD,
        connect_timeout=CONNECTION_TIMEOUT_SECONDS,
    )


def _temporary_database_url(database: str) -> str:
    """构造仅在本次 smoke 期间使用的 Alembic 迁移 URL."""
    return build_orm_database_url(settings).set(database=database).render_as_string(hide_password=False)


def _create_database(admin_database: str, test_database: str) -> None:
    """在事务外安全地创建一个随机数据库.

    Args:
        admin_database: 用于打开管理员连接的已有数据库。
        test_database: 本进程生成的随机数据库名。
    """
    with psycopg.connect(_conninfo(admin_database), autocommit=True) as connection:
        connection.execute(
            sql.SQL("CREATE DATABASE {}").format(sql.Identifier(test_database)),
        )


def _drop_database(admin_database: str, test_database: str) -> None:
    """终止残留的 smoke 会话并仅删除随机数据库.

    Args:
        admin_database: 用于打开管理员连接的已有数据库。
        test_database: 本进程早前创建的随机数据库名。
    """
    with psycopg.connect(_conninfo(admin_database), autocommit=True) as connection:
        connection.execute(
            """
            SELECT pg_terminate_backend(pid)
            FROM pg_stat_activity
            WHERE datname = %s AND pid <> pg_backend_pid()
            """,
            (test_database,),
        )
        connection.execute(
            sql.SQL("DROP DATABASE IF EXISTS {}").format(
                sql.Identifier(test_database),
            )
        )


def _runtime_settings(database: str) -> Settings:
    """复制全局 settings 并将 PostgreSQL 指向随机数据库.

    Args:
        database: 已完成业务 Alembic 迁移的随机数据库。

    Returns:
        供每代 lifespan 使用的独立 Settings。provider 设置仍为真实的
        Git-ignored 开发值，脚本不会用伪造模型替换它们。
    """
    config = Settings()
    config.POSTGRES_DB = database
    config.POSTGRES_PSYCOPG_POOL_SIZE = 2
    config.POSTGRES_ORM_POOL_SIZE = 2
    config.POSTGRES_ORM_MAX_OVERFLOW = 0
    return config


def _selector_loop_factory() -> asyncio.AbstractEventLoop:
    """创建 psycopg 异步连接所需的 Windows 事件循环."""
    return asyncio.SelectorEventLoop(selectors.SelectSelector())


def _interrupts(snapshot: StateSnapshot) -> tuple[Interrupt, ...]:
    """从 LangGraph StateSnapshot 中收集所有 interrupt 记录.

    Args:
        snapshot: ``ChatGraph.aget_state`` 返回的类型化不可变快照。

    Returns:
        保留任务顺序的每个待处理任务的所有 interrupt。
    """
    return tuple(interrupt for task in snapshot.tasks for interrupt in task.interrupts)


async def _exercise_recovery(database: str) -> dict[str, bool | float | str]:
    """对同一个 PostgreSQL 数据库运行两代真实 production lifespan.

    第一代（A）写入普通聊天记忆并触发 HITL interrupt，然后完整关闭；
    第二代（B）使用全新对象重建运行时，验证能从 PostgreSQL 读取 A 的状态，
    并完成普通追问与 HITL resume。

    Args:
        database: 已完成业务迁移和 checkpointer 迁移的随机数据库。

    Returns:
        包含对象重建、普通历史恢复和 HITL 恢复检查项的安全摘要字典。

    Raises:
        RuntimeError: 某个阶段未产生预期的应用类型。
        Exception: 基础设施和 provider 故障会向上传播，顶层仍会清理临时数据库。
    """
    started_at = perf_counter()
    user_id = UUID("44444444-4444-4444-8444-444444444444")
    memory_thread_id = uuid4()
    hitl_thread_id = uuid4()
    memory_marker = f"RECOVERY-{uuid4().hex[:12].upper()}"
    hitl_question_marker = f"APPROVAL-{uuid4().hex[:12].upper()}"

    memory_prompt = (
        f"Remember the exact marker {memory_marker} for a later turn. Do not call any tool. "
        f"Reply with exactly {MEMORY_ACK} and nothing else."
    )
    memory_follow_up = (
        "What exact marker did I ask you to remember earlier in this conversation? "
        "Do not call any tool. Reply with the marker only and nothing else."
    )
    hitl_prompt = (
        "Call the ask_human tool exactly once and ask this exact question: "
        f"Approve recovery check {hitl_question_marker}? Do not answer directly. "
        "After receiving the human response, do not call any tool again and reply with "
        f"exactly {HITL_FINAL_REPLY}."
    )

    memory_config = ChatService._build_config(
        user_id=user_id,
        public_thread_id=memory_thread_id,
    )
    hitl_config = ChatService._build_config(
        user_id=user_id,
        public_thread_id=hitl_thread_id,
    )

    captured_checkpointers: list[BaseCheckpointSaver[str] | None] = []
    captured_graphs: list[ChatGraph] = []

    def capture_runtime(
        *,
        checkpointer: BaseCheckpointSaver[str] | None = None,
    ) -> ChatGraph:
        """捕获每代 lifespan 构建的真实 production Graph，仅记录身份.

        Args:
            checkpointer: 当前活跃 lifespan 代提供的 saver。

        Returns:
            包含真实模型和工具注册表的完整 production ChatGraph。
        """
        captured_checkpointers.append(checkpointer)
        graph = create_chat_runtime(checkpointer=checkpointer)
        captured_graphs.append(graph)
        return graph

    first_resources: ApplicationResources | None = None
    first_service: ChatService | None = None
    first_graph: ChatGraph | None = None
    first_pending_tool_call_id: str | None = None

    async with asyncio.timeout(TOTAL_TIMEOUT_SECONDS):
        # 第一代 A 拥有自己的 pool、saver、编译后的 Graph 和 ChatService。
        # patch 只捕获对象身份；create_chat_runtime 仍构建真实的 production Agent。
        with patch.object(lifespan_module, "create_chat_runtime", capture_runtime):
            first_app = FastAPI()
            async with lifespan(first_app, config=_runtime_settings(database)):
                first_resources = get_application_resources(first_app)
                first_service = get_application_chat_service(first_app)
                if len(captured_graphs) != 1:
                    raise RuntimeError("lifespan A 必须恰好构建一个 Chat 图")
                first_graph = captured_graphs[0]

                # Alembic 已创建业务表，但空数据库还没有 owner 与 session 行。
                # 先写入真实业务所有权，后续 ChatService 才有资格读取或写入
                # LangGraph checkpoint。第二代 lifespan 会复用这些持久化行。
                async with first_resources.orm_session_factory() as session:
                    session.add(
                        User(
                            id=user_id,
                            email=f"recovery-{uuid4().hex}@example.com",
                            password_hash="smoke-only-not-a-real-credential",
                        )
                    )
                    session.add_all(
                        [
                            ChatSession(
                                id=memory_thread_id,
                                user_id=user_id,
                                title="PostgreSQL memory recovery",
                            ),
                            ChatSession(
                                id=hitl_thread_id,
                                user_id=user_id,
                                title="PostgreSQL HITL recovery",
                            ),
                        ]
                    )
                    await session.commit()

                memory_result = await first_service.run_turn(
                    ChatRequest(
                        thread_id=memory_thread_id,
                        message=memory_prompt,
                    ),
                    user_id=user_id,
                )
                memory_seed_completed = (
                    isinstance(memory_result, ChatResponse) and memory_result.message.content.strip() == MEMORY_ACK
                )
                if not memory_seed_completed:
                    raise RuntimeError("lifespan A 未存储普通记忆种子")

                hitl_result = await first_service.run_turn(
                    ChatRequest(
                        thread_id=hitl_thread_id,
                        message=hitl_prompt,
                    ),
                    user_id=user_id,
                )
                hitl_paused_in_service = (
                    isinstance(hitl_result, ChatInterrupt) and hitl_question_marker in hitl_result.question
                )
                if not hitl_paused_in_service:
                    raise RuntimeError("lifespan A 未产生预期的 HITL interrupt")

                memory_snapshot_a = await first_graph.aget_state(memory_config)
                hitl_snapshot_a = await first_graph.aget_state(hitl_config)
                memory_messages_a = memory_snapshot_a.values.get("messages", [])
                hitl_messages_a = hitl_snapshot_a.values.get("messages", [])
                hitl_interrupts_a = _interrupts(hitl_snapshot_a)

                memory_checkpoint_contains_marker = isinstance(memory_messages_a, list) and any(
                    isinstance(message, HumanMessage)
                    and isinstance(message.content, str)
                    and memory_marker in message.content
                    for message in memory_messages_a
                )

                pending_ai = next(
                    (message for message in hitl_messages_a if isinstance(message, AIMessage) and message.tool_calls),
                    None,
                )
                pending_tool_calls = pending_ai.tool_calls if pending_ai is not None else []
                pending_tool_call = pending_tool_calls[0] if len(pending_tool_calls) == 1 else None
                if pending_tool_call is not None:
                    call_id = pending_tool_call.get("id")
                    if isinstance(call_id, str) and call_id:
                        first_pending_tool_call_id = call_id

                pending_tool_name_matches = (
                    pending_tool_call is not None and pending_tool_call.get("name") == ask_human.name
                )
                pending_question = (
                    pending_tool_call.get("args", {}).get("question") if pending_tool_call is not None else None
                )
                hitl_checkpoint_is_paused = (
                    hitl_snapshot_a.next == ("tools",)
                    and len(hitl_interrupts_a) == 1
                    and isinstance(pending_question, str)
                    and hitl_question_marker in pending_question
                    and hitl_interrupts_a[0].value == pending_question
                    and first_pending_tool_call_id is not None
                    and pending_tool_name_matches
                )
                if not memory_checkpoint_contains_marker or not hitl_checkpoint_is_paused:
                    raise RuntimeError("lifespan A 的 checkpoint 状态不符合预期")

            # 离开 A 会移除 app.state 并关闭 A 的连接池。
            # 这些检查在 B 启动前执行，防止仍处于活跃状态的 A 对象伪装成持久化结果。
            first_pool_closed_before_rebuild = first_resources.postgres_pool.closed
            first_state_removed_before_rebuild = not hasattr(
                first_app.state,
                "resources",
            ) and not hasattr(first_app.state, "chat_service")

            second_app = FastAPI()
            async with lifespan(second_app, config=_runtime_settings(database)):
                second_resources = get_application_resources(second_app)
                second_service = get_application_chat_service(second_app)
                if len(captured_graphs) != 2:
                    raise RuntimeError("lifespan B 必须构建一个新的 Chat 图")
                second_graph = captured_graphs[1]

                runtime_objects_were_rebuilt = (
                    second_resources is not first_resources
                    and second_resources.postgres_pool is not first_resources.postgres_pool
                    and second_resources.checkpointer is not first_resources.checkpointer
                    and second_service is not first_service
                    and second_graph is not first_graph
                    and len(captured_checkpointers) == 2
                    and captured_checkpointers[0] is first_resources.checkpointer
                    and captured_checkpointers[1] is second_resources.checkpointer
                    and second_graph.checkpointer is second_resources.checkpointer
                )

                # 在 B 调用模型之前先读取两个快照。
                # 这是 PostgreSQL（而非 provider 猜测）携带了 A 状态的确定性证据。
                memory_snapshot_before_b = await second_graph.aget_state(memory_config)
                hitl_snapshot_before_b = await second_graph.aget_state(hitl_config)
                memory_messages_before_b = memory_snapshot_before_b.values.get("messages", [])
                hitl_messages_before_b = hitl_snapshot_before_b.values.get("messages", [])
                hitl_interrupts_before_b = _interrupts(hitl_snapshot_before_b)

                ordinary_state_visible_before_provider = (
                    isinstance(memory_messages_before_b, list)
                    and len(memory_messages_before_b) >= 2
                    and any(
                        isinstance(message, HumanMessage)
                        and isinstance(message.content, str)
                        and memory_marker in message.content
                        for message in memory_messages_before_b
                    )
                )
                interrupt_visible_before_resume = (
                    isinstance(hitl_messages_before_b, list)
                    and hitl_snapshot_before_b.next == ("tools",)
                    and len(hitl_interrupts_before_b) == 1
                )

                recovered_memory_result = await second_service.run_turn(
                    ChatRequest(
                        thread_id=memory_thread_id,
                        message=memory_follow_up,
                    ),
                    user_id=user_id,
                )
                ordinary_history_recovered = (
                    isinstance(recovered_memory_result, ChatResponse)
                    and memory_marker in recovered_memory_result.message.content
                )

                resumed_result = await second_service.resume_turn(
                    ChatResumeRequest(
                        thread_id=hitl_thread_id,
                        response=HUMAN_RESPONSE,
                    ),
                    user_id=user_id,
                )
                hitl_service_completed = (
                    isinstance(resumed_result, ChatResponse)
                    and resumed_result.message.content.strip() == HITL_FINAL_REPLY
                )

                completed_hitl_snapshot = await second_graph.aget_state(hitl_config)
                completed_messages = completed_hitl_snapshot.values.get("messages", [])
                completed_interrupts = _interrupts(completed_hitl_snapshot)
                resumed_tool_message = next(
                    (
                        message
                        for message in completed_messages
                        if isinstance(message, ToolMessage) and message.tool_call_id == first_pending_tool_call_id
                    ),
                    None,
                )
                final_ai = (
                    completed_messages[-1]
                    if completed_messages and isinstance(completed_messages[-1], AIMessage)
                    else None
                )
                hitl_message_chain_recovered = (
                    resumed_tool_message is not None
                    and resumed_tool_message.status == "success"
                    and resumed_tool_message.content == HUMAN_RESPONSE
                    and final_ai is not None
                    and not final_ai.tool_calls
                    and isinstance(final_ai.content, str)
                    and final_ai.content.strip() == HITL_FINAL_REPLY
                    and not completed_interrupts
                    and not completed_hitl_snapshot.next
                )

            second_pool_closed_after_shutdown = second_resources.postgres_pool.closed
            second_state_removed_after_shutdown = not hasattr(
                second_app.state,
                "resources",
            ) and not hasattr(second_app.state, "chat_service")

    elapsed_ms = _elapsed_ms(started_at)
    return {
        "model": settings.DEFAULT_LLM_MODEL,
        "memory_seed_completed": memory_seed_completed,
        "hitl_paused_in_service": hitl_paused_in_service,
        "memory_checkpoint_contains_marker": memory_checkpoint_contains_marker,
        "hitl_checkpoint_is_paused": hitl_checkpoint_is_paused,
        "first_pool_closed_before_rebuild": first_pool_closed_before_rebuild,
        "first_state_removed_before_rebuild": first_state_removed_before_rebuild,
        "runtime_objects_were_rebuilt": runtime_objects_were_rebuilt,
        "ordinary_state_visible_before_provider": ordinary_state_visible_before_provider,
        "interrupt_visible_before_resume": interrupt_visible_before_resume,
        "ordinary_history_recovered": ordinary_history_recovered,
        "hitl_service_completed": hitl_service_completed,
        "hitl_message_chain_recovered": hitl_message_chain_recovered,
        "second_pool_closed_after_shutdown": second_pool_closed_after_shutdown,
        "second_state_removed_after_shutdown": second_state_removed_after_shutdown,
        "within_total_budget": elapsed_ms <= TOTAL_TIMEOUT_SECONDS * 1000,
        "elapsed_ms": elapsed_ms,
    }


def _run_smoke() -> dict[str, object]:
    """创建、迁移、验证并删除一个随机 PostgreSQL 数据库.

    Returns:
        仅包含模型名称、布尔值和耗时的安全摘要。

    Raises:
        RuntimeError: 测试后无法删除随机数据库。
        Exception: 迁移、基础设施和 provider 故障会在清理后向上传播。
    """
    started_at = perf_counter()
    admin_database = settings.POSTGRES_DB
    test_database = f"deep_research_chat_recovery_{uuid4().hex[:10]}"
    previous_override = os.environ.get("ALEMBIC_DATABASE_URL")
    database_created = False
    cleanup_ok = False
    checks: dict[str, bool | float | str]

    try:
        _create_database(admin_database, test_database)
        database_created = True
        os.environ["ALEMBIC_DATABASE_URL"] = _temporary_database_url(test_database)
        command.upgrade(Config(str(PROJECT_ROOT / "alembic.ini")), "head")
        checks = asyncio.run(
            _exercise_recovery(test_database),
            loop_factory=_selector_loop_factory,
        )
    finally:
        if previous_override is None:
            os.environ.pop("ALEMBIC_DATABASE_URL", None)
        else:
            os.environ["ALEMBIC_DATABASE_URL"] = previous_override

        if database_created:
            try:
                _drop_database(admin_database, test_database)
            except Exception:
                cleanup_ok = False
            else:
                cleanup_ok = True

    if database_created and not cleanup_ok:
        raise RuntimeError("临时 Chat 恢复数据库清理失败")

    boolean_checks = tuple(value for value in checks.values() if isinstance(value, bool))
    ok = bool(boolean_checks) and all(boolean_checks) and cleanup_ok
    return {
        "ok": ok,
        **checks,
        "cleanup_ok": cleanup_ok,
        "total_elapsed_ms": _elapsed_ms(started_at),
    }


def main() -> int:
    """打印一份安全 JSON 摘要并返回 shell 友好的进程退出码."""
    started_at = perf_counter()
    try:
        summary = _run_smoke()
    except Exception as error:
        # provider 和数据库异常可能包含请求或连接诊断信息。
        # 只有异常类名对于这个顶层控制台边界是安全的。
        print(
            json.dumps(
                {
                    "ok": False,
                    "error_type": type(error).__name__,
                    "elapsed_ms": _elapsed_ms(started_at),
                }
            )
        )
        return 1

    print(json.dumps(summary))
    return 0 if summary["ok"] is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
