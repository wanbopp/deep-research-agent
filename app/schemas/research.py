"""Structured research planning schemas."""

from pydantic import BaseModel, ConfigDict, Field


class ResearchStep(BaseModel):
    """描述研究计划中的一个可执行步骤."""

    model_config = ConfigDict(
        # 模型创建后不允许修改字段。
        frozen=True,
        # provider 多返回未知字段时立即失败。
        extra="forbid",
        # 自动清理字符串首尾空白。
        str_strip_whitespace=True,
    )

    # 编号必须从 1 开始。
    step_number: int = Field(ge=1)

    # 目标字符串不能为空。
    objective: str = Field(min_length=1)

    # 至少包含一个搜索词，并保持内部不可变。
    search_queries: tuple[str, ...] = Field(min_length=1)


class ResearchPlan(BaseModel):
    """描述一次研究任务的结构化执行计划."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        str_strip_whitespace=True,
    )

    # 研究主题不能为空。
    topic: str = Field(min_length=1)

    # 至少包含一个 ResearchStep，并保持内部不可变。
    steps: tuple[ResearchStep, ...] = Field(min_length=1)
