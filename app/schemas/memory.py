"""用户记忆 schema：类型定义、存储条目与查询接口.

记忆系统允许 Agent 在对话过程中提取并持久化用户级别的长期知识。
每条记忆都属于一个固定的 MemoryKind，用于区分用途和检索策略。
"""

from enum import StrEnum
from uuid import UUID

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)

# 记忆内容文本长度上限，防止单条记忆携带过长文本进入存储或检索链路。
MAX_MEMORY_CONTENT_LENGTH = 2000
# 查询文本长度上限，与记忆内容共享同一边界。
MAX_QUERY_LENGTH = 1000
# 单次查询最多返回的记忆条数，防止检索结果过大影响模型上下文窗口。
MAX_QUERY_LIMIT = 10


class MemoryKind(StrEnum):
    """用户记忆的固定分类.

    每种分类对应不同的提取策略和检索场景：

    - PREFERENCE：用户表达的偏好，如语言、风格、工具选择等。
    - FACT：用户陈述的客观事实，如职业、所在地、技术栈等。
    - CONSTRAINT：用户设定的约束或限制，如"不要使用 Java"、"回复不超过 200 字"等。
    """

    PREFERENCE = "preference"
    FACT = "fact"
    CONSTRAINT = "constraint"


class MemorySearchStatus(StrEnum):
    """长期记忆搜索的可观察结果状态."""

    # AVAILABLE 同时包含“查到若干条”和“正常查到零条”。调用方应查看 items，
    # 不能把空元组自动解释成基础设施故障。
    AVAILABLE = "available"
    # DEGRADED 表示真实 MemoryStore 当前不可用。Agent 后续可以不带长期记忆
    # 继续回答，但日志、SSE 或 tracing 仍能观察到这次降级。
    DEGRADED = "degraded"


class _StrictMemoryModel(BaseModel):
    """为记忆模型提供统一的严格输入边界.

    ``frozen=True`` 防止已经校验的记忆在 service、后台任务和未来 Agent node
    之间传递时被原地修改；``extra='forbid'`` 让字段拼写错误立即失败。记忆正文
    可能包含用户隐私，因此校验异常也不应回显完整输入。
    """

    model_config = ConfigDict(
        frozen=True,
        str_strip_whitespace=True,
        extra="forbid",
        hide_input_in_errors=True,
    )


class MemoryCreate(_StrictMemoryModel):
    """尚未绑定用户归属的长期记忆候选.

    该模型可以来自未来的记忆提取器，但不包含 ``user_id``、主键或审计时间。
    可信用户归属必须由认证链单独传给 MemoryStore，不能让模型输出或客户端正文
    决定记忆保存到哪个用户 namespace。
    """

    # 这里只保存提炼后的单一事实或偏好，不保存完整 Prompt、整轮消息、token、
    # 密码或工具凭据。后续 MemoryService 还要在写入前执行敏感内容策略。
    content: str = Field(
        min_length=1,
        max_length=MAX_MEMORY_CONTENT_LENGTH,
        description="提炼后的单条长期记忆正文",
    )

    # 使用枚举而非自由字符串，避免 adapter 和 Agent 分别理解出不同分类。
    kind: MemoryKind = Field(
        description="记忆分类",
    )

    # 第一版只接收从聊天中提取的记忆，因此要求记录来源会话。该 UUID 只负责
    # 溯源，不负责用户授权；写入 adapter 仍必须校验会话属于同一个 user_id。
    source_thread_id: UUID = Field(
        description="提取该记忆的业务聊天会话 UUID",
    )


class MemoryExtractionCandidate(_StrictMemoryModel):
    """模型可以提出、但尚未绑定任何可信归属的记忆候选.

    结构化提取模型只能决定记忆正文和分类，不能输出 ``user_id``、数据库主键、
    来源 thread 或审计时间。可信身份与来源会话必须由认证后的服务端调用链补充，
    防止 Prompt 内容把记忆写入其他用户的 namespace。
    """

    content: str = Field(
        min_length=1,
        max_length=MAX_MEMORY_CONTENT_LENGTH,
        description="从当前对话提炼出的单条稳定用户记忆",
    )
    kind: MemoryKind = Field(
        description="候选记忆的固定分类",
    )


class MemoryExtractionResult(_StrictMemoryModel):
    """一次结构化记忆提取的显式结果.

    ``candidate=None`` 是正常结果，表示当前对话没有值得跨会话保存的稳定偏好、
    事实或约束。显式包装对象比直接要求模型返回 nullable 根对象更容易被不同的
    OpenAI-compatible provider 稳定实现。
    """

    candidate: MemoryExtractionCandidate | None = Field(
        default=None,
        description="至多一条长期记忆候选；没有稳定信息时为 null",
    )


class MemoryItem(MemoryCreate):
    """存储层成功持久化后返回的不可变权威记忆条目.

    该模型属于应用层，不直接暴露数据库行结构或向量存储细节。
    它在 ``MemoryCreate`` 的候选内容上增加服务端绑定的用户归属、UUID 主键和
    带时区审计时间。
    """

    # 使用 UUID 与 User、ChatSession 等业务实体保持一致。对象在未来数据库
    # flush 前即可拥有稳定身份，也不会向客户端暴露可枚举的递增行号。
    id: UUID = Field(
        description="记忆条目的业务 UUID",
    )

    # user_id 是存储后形成的权威归属，不来自 MemoryCreate。即使返回对象带有
    # 该字段，后续 search/delete 仍必须把可信 user_id 作为独立过滤条件。
    user_id: UUID = Field(
        description="拥有该记忆的可信用户 UUID",
    )

    # AwareDatetime 拒绝无时区值，防止不同 worker 把本地时间误认为 UTC。
    created_at: AwareDatetime = Field(
        description="记忆创建时间",
    )
    updated_at: AwareDatetime = Field(
        description="记忆最后更新时间",
    )

    @model_validator(mode="after")
    def validate_audit_window(self) -> "MemoryItem":
        """要求更新时间不早于创建时间."""
        if self.updated_at < self.created_at:
            raise ValueError("updated_at must not be earlier than created_at")
        return self


class MemoryQuery(_StrictMemoryModel):
    """严格的记忆查询请求.

    限制查询文本长度和返回数量，防止过长查询进入向量检索或返回结果过大
    影响模型上下文窗口。extra="forbid" 确保查询参数拼写正确。
    """

    # 查询文本，用于向量检索或关键词匹配。
    text: str = Field(
        min_length=1,
        max_length=MAX_QUERY_LENGTH,
        description="记忆检索文本",
    )

    # 可选分类集合允许一次检索 preference + constraint。None 表示搜索全部分类；
    # 空集合没有明确语义，因此通过 min_length=1 拒绝。
    kinds: frozenset[MemoryKind] | None = Field(
        default=None,
        min_length=1,
        description="可选的记忆分类过滤集合",
    )

    # 返回数量上限，默认 5 条，最大不超过 MAX_QUERY_LIMIT。
    limit: int = Field(
        default=5,
        gt=0,
        le=MAX_QUERY_LIMIT,
        description="最多返回的记忆条数",
    )


class MemorySearchResult(_StrictMemoryModel):
    """MemoryService 返回的可观察搜索结果.

    普通空结果与后端故障使用不同 status。这样未来 Agent node 可以在故障时继续
    工作，同时不会把“降级为空”误记成用户确实没有长期记忆。
    """

    items: tuple[MemoryItem, ...] = Field(
        default=(),
        description="按相关性排序的用户长期记忆",
    )
    status: MemorySearchStatus = Field(
        default=MemorySearchStatus.AVAILABLE,
        description="搜索是否由真实存储正常完成",
    )
    error_code: str | None = Field(
        default=None,
        description="降级时使用的稳定错误代码",
    )

    @model_validator(mode="after")
    def validate_status_fields(self) -> "MemorySearchResult":
        """保持状态、错误码和结果集合语义一致."""
        if self.status is MemorySearchStatus.AVAILABLE:
            if self.error_code is not None:
                raise ValueError("available result must not contain error_code")
            return self

        if self.items:
            raise ValueError("degraded result must not contain memory items")
        if self.error_code != "MEMORY_UNAVAILABLE":
            raise ValueError("degraded result requires MEMORY_UNAVAILABLE")
        return self

    @property
    def is_degraded(self) -> bool:
        """返回本次搜索是否因后端故障而降级."""
        return self.status is MemorySearchStatus.DEGRADED


__all__ = [
    "MAX_MEMORY_CONTENT_LENGTH",
    "MAX_QUERY_LENGTH",
    "MAX_QUERY_LIMIT",
    "MemoryCreate",
    "MemoryExtractionCandidate",
    "MemoryExtractionResult",
    "MemoryItem",
    "MemoryKind",
    "MemoryQuery",
    "MemorySearchResult",
    "MemorySearchStatus",
]
