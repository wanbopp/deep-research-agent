"""带来源、信任级别、敏感度和截断事实的模型上下文片段."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import StrEnum

import tiktoken


class ContextKind(StrEnum):
    """片段在模型输入中的语义角色."""

    INSTRUCTION = "instruction"
    USER_INPUT = "user_input"
    EVIDENCE = "evidence"
    FEEDBACK = "feedback"
    TOOL_RESULT = "tool_result"


class ContextSource(StrEnum):
    """有限来源集合，后续策略不依赖自由格式标签."""

    SYSTEM = "system"
    USER_TOPIC = "user_topic"
    RESEARCH_PLAN = "research_plan"
    MEMORY = "memory"
    HYBRID_RAG = "hybrid_rag"
    GRAPH_RAG = "graph_rag"
    WEB = "web"
    VALIDATION_FEEDBACK = "validation_feedback"
    TOOL_OUTPUT = "tool_output"


class TrustLevel(StrEnum):
    """内容可被模型参考的可信度边界."""

    TRUSTED = "trusted"
    VALIDATED = "validated"
    UNTRUSTED = "untrusted"


class Sensitivity(StrEnum):
    """片段敏感级别；第一版只记录，不自动上传到 trace."""

    PUBLIC = "public"
    INTERNAL = "internal"
    USER_PRIVATE = "user_private"


@dataclass(frozen=True, slots=True)
class ContextFragment:
    """一个可独立预算和审计的上下文片段."""

    kind: ContextKind
    source: ContextSource
    trust_level: TrustLevel
    sensitivity: Sensitivity
    content: str
    estimated_tokens: int = 0
    original_tokens: int = 0
    truncated: bool = False

    def __post_init__(self) -> None:
        """片段正文不能为空，计数不能为负."""
        if not self.content:
            raise ValueError("context fragment content must not be empty")
        if self.estimated_tokens < 0 or self.original_tokens < 0:
            raise ValueError("context token counts must not be negative")


@dataclass(frozen=True, slots=True)
class AllocatedContext:
    """分配完成后的片段、文本和显式截断统计."""

    fragments: tuple[ContextFragment, ...]
    text: str
    token_count: int
    truncated_fragment_count: int


class ContextAllocator:
    """按输入顺序分配 token；最后一个超预算片段可被确定性截断."""

    def __init__(self, *, max_tokens: int, encoding_name: str = "cl100k_base") -> None:
        """创建固定硬预算的分配器."""
        if max_tokens <= 0:
            raise ValueError("max_tokens must be greater than zero")
        self._max_tokens = max_tokens
        self._encoding = tiktoken.get_encoding(encoding_name)

    def allocate(self, fragments: tuple[ContextFragment, ...]) -> AllocatedContext:
        """返回不超过预算的上下文，并保留原始/实际 token 与截断标记."""
        allocated: list[ContextFragment] = []
        used = 0
        truncated_count = 0
        for fragment in fragments:
            tokens = self._encoding.encode(fragment.content)
            original_count = len(tokens)
            remaining = self._max_tokens - used
            if remaining <= 0:
                truncated_count += 1
                continue
            selected = tokens[:remaining]
            was_truncated = len(selected) < original_count
            content = self._encoding.decode(selected)
            if not content:
                truncated_count += 1
                continue
            allocated.append(
                replace(
                    fragment,
                    content=content,
                    estimated_tokens=len(selected),
                    original_tokens=original_count,
                    truncated=was_truncated,
                )
            )
            used += len(selected)
            if was_truncated:
                truncated_count += 1
        return AllocatedContext(
            fragments=tuple(allocated),
            text="\n\n".join(fragment.content for fragment in allocated),
            token_count=used,
            truncated_fragment_count=truncated_count,
        )


__all__ = [
    "AllocatedContext",
    "ContextAllocator",
    "ContextFragment",
    "ContextKind",
    "ContextSource",
    "Sensitivity",
    "TrustLevel",
]
