"""组合 production Chat Agent 的模型、工具、记忆节点与 checkpointer."""

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import SecretStr

from app.agents.chat.graph import ChatGraph, build_chat_graph
from app.agents.chat.nodes import (
    create_chat_node,
    create_memory_node,
    create_tool_node,
)
from app.agents.chat.tools.ask_human import ask_human
from app.agents.chat.tools.current_time import get_current_utc_time
from app.agents.chat.tools.registry import ToolRegistry
from app.core.config import settings
from app.schemas.llm import ModelSpec
from app.services.llm.factory import create_openai_chat_model
from app.services.llm.registry import LLMRegistry
from app.services.llm.service import LLMService
from app.services.memory_service import MemoryService


def create_chat_runtime(
    *,
    checkpointer: BaseCheckpointSaver[str] | None = None,
    memory_service: MemoryService | None = None,
) -> ChatGraph:
    """创建一个可跨请求复用、支持可信 runtime context 的聊天图.

    Args:
        checkpointer: 调用方注入的状态保存器。未提供时使用进程内
            InMemorySaver；保存器只拥有图状态，不拥有当前用户身份。
        memory_service: 可选的共享长期记忆应用服务。提供时启用
            ``START -> memory -> chat``；不提供时保留历史无长期记忆图。

    Returns:
        context 类型固定为 ChatRuntimeContext 的已编译 ChatGraph。具体用户实例
        仍由后续每一次 ChatService 调用提供。

    Raises:
        RuntimeError: 模型 API key 未配置，无法构造生产聊天 runtime。

    Notes:
        ``memory_service`` 是 Graph 编译时固定的共享依赖；可信 user_id 则属于
        每次 invocation，通过 ``ChatRuntimeContext`` 动态提供。两者不能合并成
        一个共享的“当前用户 service”。
    """
    # 配置缺失时尽早失败，但错误信息不能包含 key 本身。
    # Settings 使用空字符串表示未配置，因此不能只判断 None。
    if not settings.OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is required to create chat runtime")

    # 根据 settings 创建 ModelSpec。
    spec = ModelSpec(
        alias="primary",
        provider_model=settings.DEFAULT_LLM_MODEL,
        api_key=SecretStr(settings.OPENAI_API_KEY),
        base_url=settings.OPENAI_BASE_URL,
        temperature=settings.DEFAULT_LLM_TEMPERATURE,
        max_tokens=settings.MAX_TOKENS,
        request_timeout_seconds=max(
            1.0,
            settings.LLM_TOTAL_TIMEOUT * 0.75,
        ),
    )

    # 创建 LLMRegistry 和 LLMService。
    llm_registry = LLMRegistry(
        specs=[spec],
        factory=create_openai_chat_model,
    )

    llm_service = LLMService(
        llm_registry,
        max_attempts=settings.MAX_LLM_CALL_RETRIES,
        retry_wait_multiplier=0.2,
        total_timeout_seconds=settings.LLM_TOTAL_TIMEOUT,
    )

    # 注册 current_time 与 ask_human。
    tool_registry = ToolRegistry(
        (
            get_current_utc_time,
            ask_human,
        )
    )

    # 创建 chat node 和 tool node。
    chat_node = create_chat_node(
        llm_service=llm_service,
        aliases=("primary",),
        tool_registry=tool_registry,
    )

    tool_node = create_tool_node(
        registry=tool_registry,
        tool_timeout_seconds=20,
    )

    # 可选注入保留历史 smoke 的最小图，同时让 production lifespan 显式开启
    # 长期记忆。memory node 闭包只保存无状态 service，不保存当前用户身份。
    memory_node = create_memory_node(memory_service) if memory_service is not None else None

    # 调用方没有注入 checkpointer 时使用 InMemorySaver。
    runtime_checkpointer = checkpointer if checkpointer is not None else InMemorySaver()

    # 编译并返回 graph。
    return build_chat_graph(
        chat_node,
        memory_node=memory_node,
        tool_node=tool_node,
        checkpointer=runtime_checkpointer,
    )
