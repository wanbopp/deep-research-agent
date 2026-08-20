"""Human-in-the-loop tools for the chat Agent.

这个模块定义了 Agent 在运行过程中可以向人类提问的工具。
所谓 "Human-in-the-loop"（人机协作）是指：AI 不是完全自主运行，
而是在需要时暂停执行，把问题展示给真实用户，等用户回答后再继续。
"""

# tool 装饰器：把普通函数变成 LLM 可调用的工具。
# 加了 @tool 后，函数的名字、docstring、参数签名
# 都会被自动提取，发送给大模型，让模型知道"有这样一个工具可以用"。
from langchain_core.tools import tool

# interrupt 是 LangGraph 提供的"中断原语"。
# 调用它时，整个 LangGraph 图的执行会立刻暂停，
# 并把传入的值交给图的外部调用方（比如你的 API 接口）。
# 外部拿到值后，可以展示给用户，再通过 resume 机制把用户答案送回，图继续执行。
from langgraph.types import interrupt


# @tool 装饰器做了三件事：
#   1. 把函数注册为 LangChain 的 BaseTool 实例
#   2. 把函数的 docstring 作为工具描述发给 LLM
#   3. 把函数的参数签名（question: str）转为 JSON Schema 发给 LLM
@tool
def ask_human(question: str) -> str:
    """Pause the graph and ask the human for required input.

    上面这段 docstring 会被 @tool 提取，发给大模型。
    模型读到这段话后，就知道：当我需要向用户提问时，调用这个工具。

    参数:
        question: 要问用户的问题内容。

    返回:
        用户的回答（字符串）。
    """
    # ---- 执行到这里时，图会暂停 ----
    # interrupt() 做两件事：
    #   1. 把 question 作为中断值抛出，外部调用方可以读到它
    #   2. 阻塞当前执行，直到外部通过 Command(resume=用户答案) 恢复
    # 恢复后，interrupt() 的返回值就是用户输入的答案
    response = interrupt(question)

    # interrupt 的返回值类型不确定（取决于外部传入什么），
    # 这里统一转成字符串，确保与函数签名 -> str 一致，
    # 也让 LLM 拿到的工具结果始终是字符串格式。
    return str(response)
