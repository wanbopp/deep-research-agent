"""确定性验证 LangGraph 的暂停与恢复机制.

本脚本故意不调用 LLM，而是手工构造一条标准 ToolCall，只验证运行时机制：

1. ask_human 内部的 interrupt(question) 让 tools 节点暂停。
2. InMemorySaver 保存暂停状态和待重新执行的节点。
3. thread_id 标识要恢复的那一条图执行线程。
4. Command(resume=answer) 为原 interrupt 提供返回值。
5. tools 节点重放后生成与原 ToolCall 配对的 ToolMessage。

模型是否会主动选择 ask_human 属于下一 checkpoint，必须使用真实 provider
验证，不能与这里的确定性运行时实验混在一起。
"""

import asyncio
import json
from uuid import UUID

from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.messages.tool import tool_call
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from app.agents.chat.context import ChatRuntimeContext
from app.agents.chat.nodes import create_tool_node
from app.agents.chat.state import ChatState
from app.agents.chat.tools.ask_human import ask_human
from app.agents.chat.tools.registry import ToolRegistry

THREAD_ID = "smoke-human-interrupt"
TOOL_CALL_ID = "call-human"
HUMAN_QUESTION = "Approve this bounded smoke action?"
HUMAN_REPLY = "approved"
SMOKE_CONTEXT = ChatRuntimeContext(user_id=UUID("00000000-0000-4000-8000-000000000001"))


async def smoke_human_interrupt() -> int:
    """执行一次暂停和恢复，并用退出码表示全部行为保证是否成立."""
    # Registry 同时是执行白名单：模型消息只能调用这里注册过的工具。
    # 当前只注册 ask_human，避免恢复时重放其他具有副作用的并发工具。
    registry = ToolRegistry((ask_human,))
    tool_node = create_tool_node(registry)

    # 这张图只有 tools 节点，因为本实验不测试模型决策：
    #
    # START -> tools -> END
    #
    # 初始 AIMessage 由脚本手工提供，作用等同于“模型已经决定调用
    # ask_human”。这样可以只观察 interrupt/checkpoint/resume。
    builder = StateGraph(
        state_schema=ChatState,
        context_schema=ChatRuntimeContext,
    )
    builder.add_node("tools", tool_node)
    builder.add_edge(START, "tools")
    builder.add_edge("tools", END)

    # interrupt 必须依赖 checkpointer。没有它，LangGraph 无法保存
    # 暂停时的状态、待执行节点和 resume 值应该返回给哪个 interrupt。
    graph = builder.compile(
        checkpointer=InMemorySaver(),
    )

    # thread_id 是 checkpoint 的逻辑索引。恢复时必须复用同一份 config；
    # 更换 thread_id 等于访问另一条执行线程，无法找到这里的暂停状态。
    config: RunnableConfig = {
        "configurable": {
            "thread_id": THREAD_ID,
        }
    }

    # 这里手工构造 ToolCall，不是在伪造模型行为，而是在隔离测试运行时。
    # TOOL_CALL_ID 会在恢复后写入 ToolMessage.tool_call_id，用于结果配对。
    model_message = AIMessage(
        content="",
        tool_calls=[
            tool_call(
                name=ask_human.name,
                args={
                    "question": HUMAN_QUESTION,
                },
                id=TOOL_CALL_ID,
            )
        ],
    )

    # 第一次执行到 interrupt(question) 时会抛出 GraphInterrupt。
    # LangGraph 捕获这个控制流信号、写入 checkpoint，然后让 ainvoke()
    # 正常返回；它不会阻塞当前 Python 进程等待用户输入。
    first_result = await graph.ainvoke(
        {
            "messages": [model_message],
        },
        config=config,
        context=SMOKE_CONTEXT,
    )

    # aget_state() 读取该 thread_id 最新的 checkpoint 快照。
    # 暂停时 next 应为 ("tools",)，表示 tools 节点尚未完成，
    # 恢复后需要从该节点开头重新执行。
    paused_state = await graph.aget_state(config)

    # 首次调用结果中的 __interrupt__ 是图向外暴露的暂停事件；
    # messages 则仍保存最初那条包含 ToolCall 的 AIMessage。
    first_result_dict = dict(first_result)
    first_has_interrupt = "__interrupt__" in first_result_dict

    # task.interrupts 保存当前暂停任务的 interrupt 元数据。
    # 最终摘要只记录数量，不打印问题正文或 checkpoint 内部对象。
    interrupts = paused_state.tasks[0].interrupts if paused_state.tasks else ()

    # Command(resume=...) 不是一条 HumanMessage，而是 LangGraph 控制输入。
    # 它通过相同 thread_id 找到 checkpoint，并把 HUMAN_REPLY 放入该任务
    # 的 resume scratchpad。
    #
    # 恢复不是从 interrupt 下一行直接继续：tools 节点会从开头重放。
    # 第二次执行到同一个 interrupt 位置时，它从 scratchpad 取出
    # HUMAN_REPLY 并正常返回，ask_human 才会继续生成工具结果。
    resumed_result = await graph.ainvoke(
        Command(resume=HUMAN_REPLY),
        config=config,
        context=SMOKE_CONTEXT,
    )

    # 恢复完成后再读取 checkpoint。next 为空元组表示没有待执行节点，
    # 即 tools 已完成并沿固定边到达 END。
    completed_state = await graph.aget_state(config)

    resumed_messages = resumed_result["messages"]
    last_message = resumed_messages[-1] if resumed_messages and isinstance(resumed_messages[-1], ToolMessage) else None

    # 恢复不会追加 HumanMessage。HUMAN_REPLY 成为 interrupt() 的返回值，
    # 随后由 ask_human 返回，并被 tool_node 包装成 ToolMessage。
    resumed_message_types = [type(message).__name__ for message in resumed_messages]

    # 三个字段共同证明 ToolMessage 是原调用的有效执行回执：
    # - tool_call_id：属于哪一条 ToolCall；
    # - status：工具是否成功完成；
    # - content：恢复值是否真正穿过 interrupt 和 ask_human。
    tool_call_id_matches = last_message is not None and last_message.tool_call_id == TOOL_CALL_ID
    tool_content_matches = last_message is not None and last_message.content == HUMAN_REPLY
    tool_status = last_message.status if last_message is not None else None

    # 自动 smoke 必须自行判断全部行为保证，不能依赖人眼阅读 print。
    # 任一条件失败都会令 ok=False，并最终返回非零进程退出码。
    ok = (
        first_has_interrupt
        and paused_state.next == ("tools",)
        and len(interrupts) == 1
        and resumed_message_types
        == [
            "AIMessage",
            "ToolMessage",
        ]
        and tool_call_id_matches
        and tool_status == "success"
        and tool_content_matches
        and completed_state.next == ()
    )

    # 只输出结构和布尔结果，不输出问题、人的回答、完整消息或 checkpoint。
    print(
        json.dumps(
            {
                "ok": ok,
                "first_has_interrupt": first_has_interrupt,
                "paused_next": list(paused_state.next),
                "interrupt_count": len(interrupts),
                "resumed_message_count": len(resumed_messages),
                "resumed_message_types": resumed_message_types,
                "tool_call_id_matches": tool_call_id_matches,
                "tool_status": tool_status,
                "tool_content_matches": tool_content_matches,
                "completed_next": list(completed_state.next),
            }
        )
    )

    # 进程退出码让本地命令或 CI 能自动识别 smoke 是否成功。
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(smoke_human_interrupt()))
