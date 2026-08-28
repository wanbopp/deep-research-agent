"""把 Local/Global 图结果转换为带原始 chunk 引用的统一上下文."""

from hashlib import sha256

from app.graphrag.schemas import (
    GlobalGraphResult,
    GraphCitation,
    GraphContext,
    LocalGraphResult,
)


class GraphContextAssembler:
    """按硬字符预算组装路径和社区摘要，并维护稳定引用表."""

    def __init__(self, *, max_characters: int = 12000) -> None:
        """设置简单可预测的预模型预算；Phase 7 还会施加 token 总预算."""
        if max_characters <= 0:
            raise ValueError("max_characters must be greater than zero")
        self._max_characters = max_characters

    def assemble(self, *, local: LocalGraphResult, global_result: GlobalGraphResult) -> GraphContext:
        """生成图证据正文；社区摘要的引用仍指向原始 source chunk."""
        citations: dict[object, GraphCitation] = {}
        parts: list[str] = []

        def citation_for(chunk_id: object) -> str:
            citation_id = "G" + sha256(str(chunk_id).encode()).hexdigest()[:8].upper()
            if chunk_id not in citations:
                from uuid import UUID

                citations[chunk_id] = GraphCitation(citation_id=citation_id, source_chunk_id=UUID(str(chunk_id)))
            return citation_id

        for path in local.paths:
            labels = [citation_for(chunk_id) for chunk_id in path.source_chunk_ids]
            relation_text = " -> ".join(path.entity_names)
            part = f"[Path {' '.join(labels)}] {relation_text}: {' | '.join(path.evidence_texts)}"
            if sum(len(value) for value in parts) + len(part) <= self._max_characters:
                parts.append(part)

        for community in global_result.communities:
            labels = [citation_for(chunk_id) for chunk_id in community.source_chunk_ids]
            part = f"[Community {' '.join(labels)}] {community.title}: {community.summary}"
            if sum(len(value) for value in parts) + len(part) <= self._max_characters:
                parts.append(part)

        return GraphContext(text="\n\n".join(parts), citations=tuple(citations.values()))


__all__ = ["GraphContextAssembler"]
