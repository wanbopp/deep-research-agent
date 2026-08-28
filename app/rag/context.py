"""在 token 预算内组装可回查证据与稳定引用."""

from dataclasses import dataclass
from hashlib import sha256

import tiktoken

from app.rag.retrieval import RetrievedChunk


@dataclass(frozen=True, slots=True)
class Citation:
    """上下文中的稳定引用标识及其来源坐标."""

    citation_id: str
    chunk_id: str
    document_id: str
    source_locations: tuple[dict[str, object], ...]


@dataclass(frozen=True, slots=True)
class AssembledContext:
    """送给生成模型的正文与一一对应引用."""

    text: str
    citations: tuple[Citation, ...]
    token_count: int


class ContextAssembler:
    """按候选顺序装入预算，按 chunk_id 去重并保持引用一致."""

    def __init__(self, *, max_tokens: int, encoding_name: str = "cl100k_base") -> None:
        """保存硬 token 预算；不允许生成阶段再次无边界拼接文本."""
        if max_tokens <= 0:
            raise ValueError("max_tokens must be greater than zero")
        self._max_tokens = max_tokens
        self._encoding = tiktoken.get_encoding(encoding_name)

    def assemble(self, candidates: tuple[RetrievedChunk, ...]) -> AssembledContext:
        """只纳入完整 chunk；预算不足时停止，不产生无法回查的半截引用."""
        parts: list[str] = []
        citations: list[Citation] = []
        seen: set[object] = set()
        used = 0
        for candidate in candidates:
            if candidate.chunk_id in seen:
                continue
            citation_id = "C" + sha256(str(candidate.chunk_id).encode()).hexdigest()[:8].upper()
            rendered = f"[{citation_id}] {candidate.text}"
            token_count = len(self._encoding.encode(rendered))
            if used + token_count > self._max_tokens:
                continue
            seen.add(candidate.chunk_id)
            parts.append(rendered)
            citations.append(
                Citation(
                    citation_id=citation_id,
                    chunk_id=str(candidate.chunk_id),
                    document_id=str(candidate.document_id),
                    source_locations=tuple(candidate.source_locations),
                )
            )
            used += token_count
        return AssembledContext(text="\n\n".join(parts), citations=tuple(citations), token_count=used)


__all__ = ["AssembledContext", "Citation", "ContextAssembler"]
