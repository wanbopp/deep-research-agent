"""Verify trusted Chat Agent identity isolation with a real model provider.

本 smoke 聚焦 Checkpoint 9F 的边界，不重复测试 9E 已经完成的 JWT 密码学和
PostgreSQL 用户查询。它通过 FastAPI dependency override 提供两个已经认证完成的
``AuthenticatedUser``，随后让请求继续经过真实 route、ChatService、LangGraph、
InMemorySaver、工具节点和真实模型 provider。

执行顺序：

1. 三个 Chat HTTP 入口在没有可信身份时都返回 401，且不会调用模型；
2. 用户 A 在公开 thread ID 上触发真实 ``ask_human`` 中断；
3. 用户 B 使用相同公开 thread ID 完成一轮真实普通聊天；
4. 白盒读取两个内部 checkpoint，确认 key、消息和暂停状态互相隔离；
5. 用户 B 尝试恢复同名 thread，得到不泄漏 A 会话存在性的 404；
6. 用户 A 恢复自己的中断，真实模型生成最终回答。

控制台不会输出 token、用户 UUID、公开或内部 thread ID、Prompt、ToolCall ID、
模型完整响应或 checkpoint 内容，只输出布尔验收项、模型名称和耗时。
"""

import asyncio
import json
import os
from time import perf_counter
from typing import Any, cast
from uuid import UUID, uuid4

# Settings 在导入 app 模块时初始化。先写入受控 smoke 参数，dotenv 不会覆盖
# 已存在的环境变量；真实 API key 和 base URL 仍只从 Git 忽略配置读取。
os.environ["DEBUG"] = "false"
os.environ["LOG_LEVEL"] = "WARNING"
os.environ["MAX_LLM_CALL_RETRIES"] = "1"
os.environ["LLM_TOTAL_TIMEOUT"] = "180"
os.environ["MAX_TOKENS"] = "512"

from asgi_correlation_id import CorrelationIdMiddleware  # noqa: E402
from fastapi import FastAPI, HTTPException, Request, status  # noqa: E402
from httpx import ASGITransport, AsyncClient, Response  # noqa: E402
from langchain_core.messages import AIMessage, HumanMessage  # noqa: E402

from app.agents.chat.runtime import create_chat_runtime  # noqa: E402
from app.agents.chat.tools.ask_human import ask_human  # noqa: E402
from app.api.dependencies import get_chat_service, get_current_user  # noqa: E402
from app.api.v1.chat import router as chat_router  # noqa: E402
from app.core.config import settings  # noqa: E402
from app.core.exception_handlers import register_exception_handlers  # noqa: E402
from app.infrastructure.chat_guard import InProcessChatExecutionGuard  # noqa: E402
from app.schemas.auth import AuthenticatedUser  # noqa: E402
from app.services.chat import ChatService  # noqa: E402

HTTP_TIMEOUT_SECONDS = 300.0
GRAPH_TIMEOUT_SECONDS = 240.0
PUBLIC_THREAD_ID = f"identity-isolation-{uuid4().hex}"

# 这些值只存在于 smoke 进程内，用来代表“9E 已完成验签和数据库确认”的结果。
# token 是不具备真实认证能力的路由选择标记，因此不会被输出或用于生产代码。
USER_A_TOKEN = "smoke-authenticated-user-a"
USER_B_TOKEN = "smoke-authenticated-user-b"
USER_A = AuthenticatedUser(
    user_id=UUID("11111111-1111-4111-8111-111111111111"),
    email="smoke-user-a@example.com",
)
USER_B = AuthenticatedUser(
    user_id=UUID("22222222-2222-4222-8222-222222222222"),
    email="smoke-user-b@example.com",
)

USER_B_EXPECTED = "REAL_IDENTITY_USER_B_OK"
USER_A_RESUMED_EXPECTED = "REAL_IDENTITY_USER_A_RESUMED_OK"
HUMAN_RESPONSE = "approved"

USER_A_INTERRUPT_PROMPT = (
    "Call the ask_human tool exactly once to ask whether this identity isolation "
    "smoke is approved. Do not answer directly. After receiving the human "
    "response, do not call any tool again and reply with exactly "
    "REAL_IDENTITY_USER_A_RESUMED_OK."
)
USER_B_PROMPT = "Do not call any tool. Reply with exactly REAL_IDENTITY_USER_B_OK and nothing else."


def _json_object(response: Response) -> dict[str, Any]:
    """把 HTTP JSON body 收窄为对象.

    Args:
        response: httpx 从 FastAPI ASGI 应用取得的响应。

    Returns:
        可按字段读取的 JSON 对象。

    Raises:
        TypeError: body 不是 JSON object，说明公开 HTTP 协议已经偏离预期。
    """
    payload: object = response.json()
    if not isinstance(payload, dict):
        raise TypeError("chat identity smoke response must be a JSON object")
    return cast(dict[str, Any], payload)


def _authorization_header(token: str) -> dict[str, str]:
    """构造 smoke 请求使用的 Bearer header，不记录 token."""
    return {"Authorization": f"Bearer {token}"}


def _extract_bearer_token(request: Request) -> str | None:
    """从 smoke HTTP 请求提取严格 Bearer token.

    这里不是生产认证实现。生产路径仍由 ``get_current_user`` 完成 JWT 验签和查库；
    本 helper 只让 9F 的身份注入 smoke 从 HTTP 边界选择两个固定可信用户。
    """
    authorization = request.headers.get("Authorization")
    if authorization is None:
        return None

    scheme, separator, token = authorization.partition(" ")
    if separator != " " or scheme.lower() != "bearer" or not token:
        return None
    return token


async def _run_smoke() -> int:
    """执行真实身份隔离与 HITL 恢复闭环.

    Returns:
        所有断言通过时返回进程退出码 0，否则返回 1。
    """
    started_at = perf_counter()

    if not settings.OPENAI_API_KEY:
        print(json.dumps({"ok": False, "error_type": "MissingApiKey"}))
        return 1

    # graph 与 service 只能各创建一次。A/B 必须共享同一个 saver，才能证明隔离
    # 来自内部 checkpoint key，而不是“每个用户碰巧使用了不同 service”。
    graph = create_chat_runtime()
    service = ChatService(
        graph,
        # 本 smoke 的重点是用户 checkpoint 隔离；进程内 guard 只补齐执行边界，
        # 不替换后面的真实 provider、LangGraph 或身份检查。
        execution_guard=InProcessChatExecutionGuard(),
        graph_timeout_seconds=GRAPH_TIMEOUT_SECONDS,
    )

    app = FastAPI()
    register_exception_handlers(app)
    app.add_middleware(CorrelationIdMiddleware)
    app.include_router(chat_router, prefix="/api/v1")

    async def override_current_user(request: Request) -> AuthenticatedUser:
        """把两个 smoke Bearer 标记转换为可信用户对象.

        Args:
            request: FastAPI 当前请求；只读取 Authorization header。

        Returns:
            与 Bearer 标记对应的固定 AuthenticatedUser。

        Raises:
            HTTPException: header 缺失、格式错误或标记未知时统一返回 401。
        """
        token = _extract_bearer_token(request)
        if token == USER_A_TOKEN:
            return USER_A
        if token == USER_B_TOKEN:
            return USER_B
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

    def override_chat_service() -> ChatService:
        """让全部 HTTP 请求共享同一个真实 ChatService 和 checkpointer."""
        return service

    app.dependency_overrides[get_current_user] = override_current_user
    app.dependency_overrides[get_chat_service] = override_chat_service

    transport = ASGITransport(app=app, raise_app_exceptions=False)

    try:
        async with AsyncClient(
            transport=transport,
            base_url="http://testserver",
            timeout=HTTP_TIMEOUT_SECONDS,
        ) as client:
            # 这三次请求都必须在 route 调用 ChatService 之前被认证依赖拒绝。
            # 如果误调用了 provider，后续耗时和日志会暴露边界错误。
            unauthorized_responses = (
                await client.post(
                    "/api/v1/chat",
                    json={"thread_id": PUBLIC_THREAD_ID, "message": USER_B_PROMPT},
                ),
                await client.post(
                    "/api/v1/chat/resume",
                    json={"thread_id": PUBLIC_THREAD_ID, "response": HUMAN_RESPONSE},
                ),
                await client.post(
                    "/api/v1/chat/stream",
                    json={"thread_id": PUBLIC_THREAD_ID, "message": USER_B_PROMPT},
                ),
            )
            unauthorized_entries_rejected = all(
                response.status_code == status.HTTP_401_UNAUTHORIZED for response in unauthorized_responses
            )

            invalid_token_response = await client.post(
                "/api/v1/chat",
                headers=_authorization_header("unknown-smoke-token"),
                json={"thread_id": PUBLIC_THREAD_ID, "message": USER_B_PROMPT},
            )
            invalid_token_rejected = invalid_token_response.status_code == status.HTTP_401_UNAUTHORIZED

            if not unauthorized_entries_rejected or not invalid_token_rejected:
                print(
                    json.dumps(
                        {
                            "ok": False,
                            "stage": "authentication",
                            "unauthorized_entries_rejected": unauthorized_entries_rejected,
                            "invalid_token_rejected": invalid_token_rejected,
                        }
                    )
                )
                return 1

            # 用户 A 的真实模型调用必须产生 ask_human ToolCall，并暂停在 tools。
            user_a_interrupt_response = await client.post(
                "/api/v1/chat",
                headers=_authorization_header(USER_A_TOKEN),
                json={
                    "thread_id": PUBLIC_THREAD_ID,
                    "message": USER_A_INTERRUPT_PROMPT,
                },
            )
            user_a_interrupt_body = _json_object(user_a_interrupt_response)
            user_a_interrupted = (
                user_a_interrupt_response.status_code == status.HTTP_200_OK
                and user_a_interrupt_body.get("status") == "interrupted"
                and user_a_interrupt_body.get("thread_id") == PUBLIC_THREAD_ID
                and isinstance(user_a_interrupt_body.get("question"), str)
            )

            if not user_a_interrupted:
                print(
                    json.dumps(
                        {
                            "ok": False,
                            "stage": "user_a_interrupt",
                            "status_code": user_a_interrupt_response.status_code,
                            "response_status": user_a_interrupt_body.get("status"),
                        }
                    )
                )
                return 1

            # 用户 B 故意复用完全相同的公开 thread ID。只有内部 key 包含 user_id，
            # 这次调用才会从空历史开始，而不是续接 A 的暂停 checkpoint。
            user_b_response = await client.post(
                "/api/v1/chat",
                headers=_authorization_header(USER_B_TOKEN),
                json={
                    "thread_id": PUBLIC_THREAD_ID,
                    "message": USER_B_PROMPT,
                },
            )
            user_b_body = _json_object(user_b_response)
            user_b_message = user_b_body.get("message")
            user_b_completed = (
                user_b_response.status_code == status.HTTP_200_OK
                and user_b_body.get("status") == "completed"
                and user_b_body.get("thread_id") == PUBLIC_THREAD_ID
                and isinstance(user_b_message, dict)
                and user_b_message.get("content") == USER_B_EXPECTED
            )

            if not user_b_completed:
                print(
                    json.dumps(
                        {
                            "ok": False,
                            "stage": "user_b_completed",
                            "status_code": user_b_response.status_code,
                            "response_status": user_b_body.get("status"),
                        }
                    )
                )
                return 1

            # 白盒检查只读取状态形状，不输出状态正文。生产 API 不提供这个入口；
            # smoke 使用它证明两个公开同名 thread 实际落入两个不同内部 key。
            user_a_config = ChatService._build_config(
                user_id=USER_A.user_id,
                public_thread_id=PUBLIC_THREAD_ID,
            )
            user_b_config = ChatService._build_config(
                user_id=USER_B.user_id,
                public_thread_id=PUBLIC_THREAD_ID,
            )
            user_a_snapshot = await graph.aget_state(user_a_config)
            user_b_snapshot = await graph.aget_state(user_b_config)

            # RunnableConfig 的类型声明允许 configurable 缺失，因此即使我们的
            # helper 必然写入它，smoke 仍按公开 TypedDict 约束使用 get() 收窄。
            user_a_configurable = user_a_config.get("configurable", {})
            user_b_configurable = user_b_config.get("configurable", {})
            user_a_internal_id = user_a_configurable.get("thread_id")
            user_b_internal_id = user_b_configurable.get("thread_id")
            internal_keys_are_distinct = user_a_internal_id != user_b_internal_id

            user_a_messages = user_a_snapshot.values.get("messages", [])
            user_b_messages = user_b_snapshot.values.get("messages", [])
            user_a_interrupts = tuple(interrupt for task in user_a_snapshot.tasks for interrupt in task.interrupts)
            user_b_interrupts = tuple(interrupt for task in user_b_snapshot.tasks for interrupt in task.interrupts)

            checkpoint_states_are_isolated = (
                internal_keys_are_distinct
                and len(user_a_interrupts) == 1
                and not user_b_interrupts
                and user_a_snapshot.next == ("tools",)
                and not user_b_snapshot.next
                and isinstance(user_a_messages, list)
                and isinstance(user_b_messages, list)
                and any(isinstance(message, AIMessage) for message in user_a_messages)
                and len(user_b_messages) == 2
                and isinstance(user_b_messages[0], HumanMessage)
                and isinstance(user_b_messages[1], AIMessage)
            )

            # B 的内部 checkpoint 没有 interrupt，因此 service 在调用 Command 前
            # 安全拒绝。404 不告诉 B 同名公开 thread 是否在 A 的空间真实存在。
            cross_user_resume_response = await client.post(
                "/api/v1/chat/resume",
                headers=_authorization_header(USER_B_TOKEN),
                json={
                    "thread_id": PUBLIC_THREAD_ID,
                    "response": HUMAN_RESPONSE,
                },
            )
            cross_user_resume_rejected = cross_user_resume_response.status_code == status.HTTP_404_NOT_FOUND

            # A 使用相同可信身份和公开 thread ID，重新得到同一个内部 key。
            # Command(resume=...) 因此进入 A 的暂停任务，工具返回后模型完成回答。
            user_a_resume_response = await client.post(
                "/api/v1/chat/resume",
                headers=_authorization_header(USER_A_TOKEN),
                json={
                    "thread_id": PUBLIC_THREAD_ID,
                    "response": HUMAN_RESPONSE,
                },
            )
            user_a_resume_body = _json_object(user_a_resume_response)
            user_a_resume_message = user_a_resume_body.get("message")
            owner_resume_succeeded = (
                user_a_resume_response.status_code == status.HTTP_200_OK
                and user_a_resume_body.get("status") == "completed"
                and user_a_resume_body.get("thread_id") == PUBLIC_THREAD_ID
                and isinstance(user_a_resume_message, dict)
                and user_a_resume_message.get("content") == USER_A_RESUMED_EXPECTED
            )

    except Exception as error:
        # 第三方错误正文可能包含请求信息，因此只输出异常类型，不输出 str(error)。
        print(
            json.dumps(
                {
                    "ok": False,
                    "stage": "exception",
                    "error_type": type(error).__name__,
                    "elapsed_ms": round((perf_counter() - started_at) * 1000, 2),
                }
            )
        )
        return 1
    finally:
        app.dependency_overrides.clear()

    # 模型看到的 ask_human schema 只能包含 question。user_id 若出现在 args 中，
    # 模型就可能伪造授权身份，违反 runtime context 的核心设计。
    tool_schema_hides_user_id = "user_id" not in ask_human.args
    singleton_service_holds_no_user = "user_id" not in vars(service)

    ok = (
        unauthorized_entries_rejected
        and invalid_token_rejected
        and user_a_interrupted
        and user_b_completed
        and checkpoint_states_are_isolated
        and cross_user_resume_rejected
        and owner_resume_succeeded
        and tool_schema_hides_user_id
        and singleton_service_holds_no_user
    )

    print(
        json.dumps(
            {
                "ok": ok,
                "model": settings.DEFAULT_LLM_MODEL,
                "unauthorized_entries_rejected": unauthorized_entries_rejected,
                "invalid_token_rejected": invalid_token_rejected,
                "user_a_interrupted": user_a_interrupted,
                "user_b_completed": user_b_completed,
                "internal_keys_are_distinct": internal_keys_are_distinct,
                "checkpoint_states_are_isolated": checkpoint_states_are_isolated,
                "cross_user_resume_status": cross_user_resume_response.status_code,
                "cross_user_resume_rejected": cross_user_resume_rejected,
                "owner_resume_succeeded": owner_resume_succeeded,
                "tool_schema_hides_user_id": tool_schema_hides_user_id,
                "singleton_service_holds_no_user": singleton_service_holds_no_user,
                "elapsed_ms": round((perf_counter() - started_at) * 1000, 2),
            }
        )
    )

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_run_smoke()))
