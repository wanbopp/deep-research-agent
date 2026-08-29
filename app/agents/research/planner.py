"""使用结构化模型输出生成有边界的研究计划."""

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.runtime import Runtime

from app.agents.prompts.loader import render_prompt
from app.agents.research.context import ResearchRuntimeContext
from app.agents.research.state import ResearchState, ResearchStateUpdate
from app.core.logging import logger
from app.schemas.research import ResearchConfig, ResearchPlan, ResearchStatus
from app.services.llm.service import LLMService


class ResearchPlanner:
    """把用户问题转换为后续节点能够执行的步骤."""

    def __init__(self, llm_service: LLMService, *, aliases: tuple[str, ...]) -> None:
        """保存无请求状态的模型服务和候选模型名称.

        Args:
            llm_service: 统一处理真实模型超时、重试和结构化输出的服务。
            aliases: 按顺序尝试的模型别名；不能包含 API key 或用户数据。
        """
        if not aliases:
            raise ValueError("planner aliases must not be empty")
        self._llm_service = llm_service
        self._aliases = aliases

    async def __call__(
        self,
        state: ResearchState,
        *,
        runtime: Runtime[ResearchRuntimeContext],
    ) -> ResearchStateUpdate:
        """生成并校验计划，只返回本节点负责的状态变化.

        runtime.context.user_id 仅用于安全日志关联和下游授权，不进入 prompt。
        Planner 看到的是研究主题和服务端预算，不能看到连接对象或认证凭据。
        """
        # Checkpoint 保存普通字典；节点入口负责恢复和校验业务模型。这样即使任务
        # 在另一进程恢复，也不会依赖原进程中的 Python 对象身份。
        config = ResearchConfig.model_validate(state["config"])
        prompt = render_prompt(
            "research_plan",
            topic=state["topic"],
            max_steps=config.max_steps,
        )
        plan = await self._llm_service.call_structured(
            (
                SystemMessage(content=prompt),
                HumanMessage(
                    content=(
                        "Create an executable research plan. Every step must include a completion "
                        "criterion and one or more suitable retrieval strategies."
                    )
                ),
            ),
            response_model=ResearchPlan,
            aliases=self._aliases,
            overrides={"temperature": 0.0},
        )

        if len(plan.steps) > config.max_steps:
            # 不静默裁剪：后续步骤可能依赖被裁掉的前置步骤，裁剪会得到看似有效但
            # 语义不完整的计划。让本次任务明确失败比执行错误计划更安全。
            raise ValueError("research plan exceeds configured max_steps")

        next_status = ResearchStatus.NEEDS_CLARIFICATION if plan.clarification_question else ResearchStatus.RESEARCHING
        logger.info(
            "research_plan_created",
            research_id=str(runtime.context.research_id),
            step_count=len(plan.steps),
            needs_clarification=bool(plan.clarification_question),
        )
        return {
            "plan": plan.model_dump(mode="json"),
            "status": next_status.value,
        }


__all__ = ["ResearchPlanner"]
