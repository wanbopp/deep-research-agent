"""ChatService.get_message_history 的映射与授权测试."""

from types import SimpleNamespace
from uuid import uuid4

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.schemas.chat import MAX_MESSAGE_LENGTH
from app.services.chat import ChatService
from app.services.chat_session_ownership import (
    ChatSessionNotFoundError,
    InProcessChatSessionOwnershipVerifier,
)


class FakeSnapshotGraph:
    """只支持 aget_state 的最小图替身，返回预置的状态值."""

    def __init__(self, values: dict[str, object]) -> None:
        """保存测试用例预置的 snapshot.values."""
        self._values = values
        self.received_configs: list[dict[str, object]] = []

    async def aget_state(self, config: dict[str, object]) -> SimpleNamespace:
        """记录收到的配置并返回固定 snapshot."""
        self.received_configs.append(config)
        return SimpleNamespace(values=self._values)


class UnusedGuard:
    """get_message_history 不进入执行锁；占位实现满足构造参数."""

    def hold(self, internal_thread_id: str) -> None:
        """历史读取路径不应调用 guard."""
        raise AssertionError("message history must not acquire the execution guard")


def build_service(
    values: dict[str, object],
    owned_sessions: list[tuple],
) -> ChatService:
    """构造使用假 snapshot 的 ChatService."""
    return ChatService(
        FakeSnapshotGraph(values),  # type: ignore[arg-type]
        execution_guard=UnusedGuard(),  # type: ignore[arg-type]
        ownership_verifier=InProcessChatSessionOwnershipVerifier(owned_sessions),
    )


@pytest.fixture
def anyio_backend() -> str:
    """与 conftest 保持一致，只使用 asyncio 后端."""
    return "asyncio"


@pytest.mark.anyio
async def test_history_maps_user_and_assistant_text(anyio_backend: str) -> None:
    """用户与助手的纯文本消息按顺序映射为公开消息."""
    user_id = uuid4()
    session_id = uuid4()
    service = build_service(
        {
            "messages": [
                HumanMessage(content="  你好  "),
                AIMessage(content="你好，有什么可以帮你？"),
                HumanMessage(content="继续"),
                AIMessage(content="好的。"),
            ]
        },
        [(user_id, session_id)],
    )

    result = await service.get_message_history(
        session_id=session_id,
        user_id=user_id,
    )

    assert [(message.role, message.content) for message in result.messages] == [
        ("user", "你好"),
        ("assistant", "你好，有什么可以帮你？"),
        ("user", "继续"),
        ("assistant", "好的。"),
    ]


@pytest.mark.anyio
async def test_history_skips_tool_and_non_public_messages(anyio_backend: str) -> None:
    """工具消息、带工具调用的 AI 消息、空内容与多模态内容都不公开."""
    user_id = uuid4()
    session_id = uuid4()
    service = build_service(
        {
            "messages": [
                HumanMessage(content="帮我查一下"),
                AIMessage(
                    content="",
                    tool_calls=[{"name": "search", "args": {}, "id": "call_1", "type": "tool_call"}],
                ),
                ToolMessage(content="搜索结果...", tool_call_id="call_1"),
                AIMessage(content=[{"type": "text", "text": "多模态块"}]),
                AIMessage(content="   "),
                AIMessage(content="最终回答"),
            ]
        },
        [(user_id, session_id)],
    )

    result = await service.get_message_history(
        session_id=session_id,
        user_id=user_id,
    )

    assert [(message.role, message.content) for message in result.messages] == [
        ("user", "帮我查一下"),
        ("assistant", "最终回答"),
    ]


@pytest.mark.anyio
async def test_history_truncates_oversized_content(anyio_backend: str) -> None:
    """超过公开上限的历史内容被截断而不是拒绝整个响应."""
    user_id = uuid4()
    session_id = uuid4()
    long_text = "长" * (MAX_MESSAGE_LENGTH + 500)
    service = build_service(
        {"messages": [HumanMessage(content="问题"), AIMessage(content=long_text)]},
        [(user_id, session_id)],
    )

    result = await service.get_message_history(
        session_id=session_id,
        user_id=user_id,
    )

    assert len(result.messages) == 2
    assert len(result.messages[1].content) == MAX_MESSAGE_LENGTH


@pytest.mark.anyio
async def test_history_is_empty_without_checkpoint(anyio_backend: str) -> None:
    """会话存在但没有任何 checkpoint 时返回空数组."""
    user_id = uuid4()
    session_id = uuid4()
    service = build_service({}, [(user_id, session_id)])

    result = await service.get_message_history(
        session_id=session_id,
        user_id=user_id,
    )

    assert result.messages == ()


@pytest.mark.anyio
async def test_history_rejects_unowned_session(anyio_backend: str) -> None:
    """未登记或跨用户的会话组合抛出统一的 not-found 错误."""
    owner_id = uuid4()
    other_user_id = uuid4()
    session_id = uuid4()
    service = build_service(
        {"messages": [HumanMessage(content="私密内容")]},
        [(owner_id, session_id)],
    )

    with pytest.raises(ChatSessionNotFoundError):
        await service.get_message_history(
            session_id=session_id,
            user_id=other_user_id,
        )


@pytest.mark.anyio
async def test_history_uses_user_scoped_checkpoint_key(anyio_backend: str) -> None:
    """Snapshot 查询必须使用包含可信 user_id 的内部 thread key."""
    user_id = uuid4()
    session_id = uuid4()
    graph = FakeSnapshotGraph({"messages": []})
    service = ChatService(
        graph,  # type: ignore[arg-type]
        execution_guard=UnusedGuard(),  # type: ignore[arg-type]
        ownership_verifier=InProcessChatSessionOwnershipVerifier([(user_id, session_id)]),
    )

    await service.get_message_history(session_id=session_id, user_id=user_id)

    config = graph.received_configs[0]
    assert config["configurable"]["thread_id"] == f"user:{user_id.hex}:thread:{session_id}"
