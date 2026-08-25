"""Use real provider calls to verify the non-streaming Chat API boundary.

这个 smoke 不绕过 FastAPI 直接调用 ChatService，而是通过 httpx 的 ASGI
transport 发送 HTTP 请求。这样可以同时覆盖 middleware、Pydantic 请求校验、
dependency 缓存、route、ChatService、LangGraph、工具和真实模型调用。

脚本执行四个阶段：

1. 发送非法请求，确认统一 422 协议；该阶段不会调用模型。
2. 发送普通聊天请求，确认真实模型返回 completed。
3. 要求真实模型调用 ask_human，确认 HTTP 返回 interrupted。
4. 使用相同 thread_id 调用 resume，确认 Agent 从 checkpoint 继续并完成。

控制台只输出状态、布尔检查、模型名称、缓存统计和耗时，不输出 API key、
完整 prompt、问题正文、人工回答、调用 ID、checkpoint 或完整模型内容。
"""

import asyncio
import json
import os
from time import perf_counter
from typing import Any, cast
from uuid import UUID, uuid4

# Settings 在 app.core.config 首次导入时创建，所以受控运行参数必须先写入
# os.environ。dotenv 默认不会覆盖已经存在的环境变量。
os.environ["DEBUG"] = "false"
os.environ["LOG_LEVEL"] = "WARNING"
os.environ["MAX_LLM_CALL_RETRIES"] = "1"
os.environ["LLM_TOTAL_TIMEOUT"] = "30"
os.environ["MAX_TOKENS"] = "512"

from httpx import ASGITransport, AsyncClient, Response  # noqa: E402

from app.api.dependencies import get_chat_service, get_current_user  # noqa: E402
from app.core.config import settings  # noqa: E402
from app.main import app  # noqa: E402
from app.schemas.auth import AuthenticatedUser  # noqa: E402

COMPLETED_EXPECTED = "REAL_HTTP_CHAT_OK"
RESUMED_EXPECTED = "REAL_HTTP_HITL_OK"
HUMAN_RESPONSE = "approved"
HTTP_TIMEOUT_SECONDS = 120.0
SMOKE_USER = AuthenticatedUser(
    user_id=UUID("00000000-0000-4000-8000-000000000001"),
    email="legacy-chat-smoke@example.com",
)

# 普通完成路径明确禁止工具调用，避免它与 HITL 路径相互干扰。
COMPLETED_PROMPT = "Do not call any tool. Reply with exactly REAL_HTTP_CHAT_OK and nothing else."

# 该提示要求真实模型先生成 ask_human ToolCall。恢复后，模型会收到对应的
# ToolMessage，再生成没有 tool_calls 的最终 AIMessage。
HITL_PROMPT = (
    "Call the ask_human tool exactly once to ask whether this bounded "
    "HTTP smoke action is approved. Do not answer directly. "
    "After receiving the human response, do not call any tool again and "
    "reply with exactly REAL_HTTP_HITL_OK."
)


def _json_object(response: Response) -> dict[str, Any]:
    """读取 JSON object；非 object 响应视为 HTTP 协议失败."""
    payload: object = response.json()
    if not isinstance(payload, dict):
        raise TypeError("chat API response must be a JSON object")
    return cast(dict[str, Any], payload)


def _optional_object(value: object) -> dict[str, Any]:
    """把可选嵌套值收窄为 JSON object，类型不符时返回空对象."""
    if not isinstance(value, dict):
        return {}
    return cast(dict[str, Any], value)


def _is_non_empty_string(value: object) -> bool:
    """判断公开响应字段是否为非空字符串."""
    return isinstance(value, str) and bool(value.strip())


def _elapsed_ms(started_at: float) -> float:
    """返回从 smoke 开始到当前时刻的毫秒耗时."""
    return round((perf_counter() - started_at) * 1000, 2)


def _print_stage_failure(
    *,
    stage: str,
    started_at: float,
    status_code: int | None = None,
    response_status: object = None,
) -> None:
    """输出脱敏的阶段失败摘要，不输出响应正文."""
    print(
        json.dumps(
            {
                "ok": False,
                "stage": stage,
                "status_code": status_code,
                "response_status": response_status,
                "elapsed_ms": _elapsed_ms(started_at),
            }
        )
    )


async def run_chat_api_smoke() -> int:
    """经过真实 HTTP 应用边界执行 completed 与 HITL 闭环."""
    started_at = perf_counter()

    if not settings.OPENAI_API_KEY:
        print(json.dumps({"ok": False, "error_type": "MissingApiKey"}))
        return 1

    # 每次脚本运行都从干净的进程内 dependency 开始。随后四个 HTTP 请求
    # 必须复用同一个 ChatService，interrupt/resume 才能共享 InMemorySaver。
    get_chat_service.cache_clear()

    # 该 Lab 10 历史 smoke 只验证 Chat API/Agent 行为。9F 的独立 smoke 已覆盖
    # 401 与跨用户隔离，因此这里注入一个已认证身份，继续保持原实验职责单一。
    app.dependency_overrides[get_current_user] = lambda: SMOKE_USER

    # 两条业务路径使用不同 thread，防止普通聊天历史影响 HITL 工具决策。
    # uuid 只用于避免重复运行脚本时碰到旧 checkpoint，最终不会输出。
    completed_thread_id = f"smoke-http-completed-{uuid4().hex}"
    hitl_thread_id = f"smoke-http-hitl-{uuid4().hex}"

    # ASGITransport 不创建 TCP 监听端口，但请求仍会经过 FastAPI 的完整
    # ASGI 生命周期：middleware -> exception handler -> dependency -> route。
    # raise_app_exceptions=False 让未处理异常成为 HTTP 500，而不是越过 API
    # 边界直接从 client.post() 抛给脚本。
    transport = ASGITransport(
        app=app,
        raise_app_exceptions=False,
    )

    try:
        async with AsyncClient(
            transport=transport,
            base_url="http://testserver",
            timeout=HTTP_TIMEOUT_SECONDS,
        ) as client:
            # 断点观察点 1：此时 dependency 缓存为空，也尚未发送模型请求。
            # 先发送空白消息，验证请求在 Pydantic/API 层被拦截。
            validation_response = await client.post(
                "/api/v1/chat",
                json={
                    "thread_id": completed_thread_id,
                    "message": "   ",
                },
            )
            validation_body = _json_object(validation_response)
            validation_error = _optional_object(validation_body.get("error"))
            validation_request_id = validation_body.get("request_id")

            validation_matches = (
                validation_response.status_code == 422
                and validation_error.get("code") == "VALIDATION_ERROR"
                and _is_non_empty_string(validation_request_id)
            )

            # 断点观察点 2：validation_body 应只有统一 error 与 request_id。
            # 如果这一步失败，停止后续真实调用，因为 API 基础协议尚未成立。
            if not validation_matches:
                _print_stage_failure(
                    stage="validation",
                    started_at=started_at,
                    status_code=validation_response.status_code,
                )
                return 1

            # 第一条真实模型调用：route 注入缓存 ChatService，run_turn 把
            # message 转为 HumanMessage，图最终返回公开 ChatResponse。
            completed_response = await client.post(
                "/api/v1/chat",
                json={
                    "thread_id": completed_thread_id,
                    "message": COMPLETED_PROMPT,
                },
            )
            completed_body = _json_object(completed_response)
            completed_message = _optional_object(completed_body.get("message"))

            completed_thread_matches = completed_body.get("thread_id") == completed_thread_id
            completed_content_matches = (
                completed_message.get("role") == "assistant" and completed_message.get("content") == COMPLETED_EXPECTED
            )
            completed_matches = (
                completed_response.status_code == 200
                and completed_body.get("status") == "completed"
                and completed_thread_matches
                and completed_content_matches
                and "question" not in completed_body
            )

            # 断点观察点 3：completed_body 是公开 API JSON，不包含
            # AIMessage、tool_calls、provider metadata 或 checkpoint。
            if not completed_matches:
                _print_stage_failure(
                    stage="completed",
                    started_at=started_at,
                    status_code=completed_response.status_code,
                    response_status=completed_body.get("status"),
                )
                return 1

            # 第二条真实模型调用：模型必须生成 ask_human ToolCall；tools
            # 节点调用 interrupt 后，ChatService 把暂停状态转换为
            # ChatInterrupt，route 再转换为 ChatInterruptResponse。
            interrupt_response = await client.post(
                "/api/v1/chat",
                json={
                    "thread_id": hitl_thread_id,
                    "message": HITL_PROMPT,
                },
            )
            interrupt_body = _json_object(interrupt_response)

            interrupt_thread_matches = interrupt_body.get("thread_id") == hitl_thread_id
            interrupt_question_is_non_empty = _is_non_empty_string(interrupt_body.get("question"))
            interrupt_matches = (
                interrupt_response.status_code == 200
                and interrupt_body.get("status") == "interrupted"
                and interrupt_thread_matches
                and interrupt_question_is_non_empty
                and "message" not in interrupt_body
            )

            # 断点观察点 4：HTTP 响应只公开 question；真正的 interrupt、
            # ToolCall ID 和暂停任务保存在缓存 runtime 的 checkpointer 中。
            if not interrupt_matches:
                _print_stage_failure(
                    stage="interrupt",
                    started_at=started_at,
                    status_code=interrupt_response.status_code,
                    response_status=interrupt_body.get("status"),
                )
                return 1

            # 断点观察点 5：resume 不是新增 HumanMessage。该 HTTP body 会被
            # ChatService 转为 Command(resume=HUMAN_RESPONSE)，并使用相同
            # thread_id 找回刚才暂停在 tools 节点的 checkpoint。
            resume_response = await client.post(
                "/api/v1/chat/resume",
                json={
                    "thread_id": hitl_thread_id,
                    "response": HUMAN_RESPONSE,
                },
            )
            resume_body = _json_object(resume_response)
            resume_message = _optional_object(resume_body.get("message"))

            resume_thread_matches = resume_body.get("thread_id") == hitl_thread_id
            resume_content_matches = (
                resume_message.get("role") == "assistant" and resume_message.get("content") == RESUMED_EXPECTED
            )
            resume_matches = (
                resume_response.status_code == 200
                and resume_body.get("status") == "completed"
                and resume_thread_matches
                and resume_content_matches
                and "question" not in resume_body
            )

            # 断点观察点 6：成功时 resume_body 表示恢复后的最终公开结果；
            # 内部消息顺序应已由 LangGraph 形成 Human、AI、Tool、AI。
            if not resume_matches:
                _print_stage_failure(
                    stage="resume",
                    started_at=started_at,
                    status_code=resume_response.status_code,
                    response_status=resume_body.get("status"),
                )
                return 1
    except Exception as error:
        # 网络、provider、JSON 解析或客户端异常只输出类型和可选状态码。
        # 不打印 str(error)，避免第三方响应或输入内容进入控制台。
        print(
            json.dumps(
                {
                    "ok": False,
                    "stage": "exception",
                    "error_type": type(error).__name__,
                    "status_code": getattr(error, "status_code", None),
                    "elapsed_ms": _elapsed_ms(started_at),
                }
            )
        )
        return 1

    cache_info = get_chat_service.cache_info()

    # 422 请求是否解析 dependency 取决于 FastAPI 的依赖求解顺序，因此
    # hits 可能是 2 或 3；真正必须保证的是只有一次 miss、缓存中只有
    # 一个 service，且后续 interrupt/resume 至少发生两次命中。
    dependency_reused = cache_info.misses == 1 and cache_info.hits >= 2 and cache_info.currsize == 1

    ok = validation_matches and completed_matches and interrupt_matches and resume_matches and dependency_reused

    print(
        json.dumps(
            {
                "ok": ok,
                "model": settings.DEFAULT_LLM_MODEL,
                "validation_status_code": validation_response.status_code,
                "validation_contract_matches": validation_matches,
                "completed_status_code": completed_response.status_code,
                "completed_status": completed_body.get("status"),
                "completed_thread_matches": completed_thread_matches,
                "completed_content_matches": completed_content_matches,
                "interrupt_status_code": interrupt_response.status_code,
                "interrupt_status": interrupt_body.get("status"),
                "interrupt_thread_matches": interrupt_thread_matches,
                "interrupt_question_is_non_empty": interrupt_question_is_non_empty,
                "resume_status_code": resume_response.status_code,
                "resume_status": resume_body.get("status"),
                "resume_thread_matches": resume_thread_matches,
                "resume_content_matches": resume_content_matches,
                "dependency_reused": dependency_reused,
                "cache_hits": cache_info.hits,
                "cache_misses": cache_info.misses,
                "cache_size": cache_info.currsize,
                "elapsed_ms": _elapsed_ms(started_at),
            }
        )
    )

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(run_chat_api_smoke()))
