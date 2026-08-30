"""把 Local/Global 图结果转换为带原始 chunk 引用的统一上下文."""

from hashlib import sha256

from app.graphrag.schemas import (
    GlobalGraphResult,
    GraphCitation,
    GraphContext,
    LocalGraphResult,
)
from app.runtime import (
    ContextAllocator,
    ContextFragment,
    ContextKind,
    ContextSource,
    Sensitivity,
    TrustLevel,
)


class GraphContextAssembler:
    """按硬字符预算组装路径和社区摘要，并维护稳定引用表."""

    def __init__(self, *, max_tokens: int = 3000) -> None:
        """使用与其他来源一致的 token 硬预算，不再混用字符估算."""
        self._allocator = ContextAllocator(max_tokens=max_tokens)

    def assemble(self, *, local: LocalGraphResult, global_result: GlobalGraphResult) -> GraphContext:
        """生成图证据正文；社区摘要的引用仍指向原始 source chunk."""
        citations: dict[object, GraphCitation] = {}
        fragments: list[ContextFragment] = []

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
            fragments.append(
                ContextFragment(
                    kind=ContextKind.EVIDENCE,
                    source=ContextSource.GRAPH_RAG,
                    trust_level=TrustLevel.UNTRUSTED,
                    sensitivity=Sensitivity.USER_PRIVATE,
                    content=part,
                )
            )

        for community in global_result.communities:
            labels = [citation_for(chunk_id) for chunk_id in community.source_chunk_ids]
            part = f"[Community {' '.join(labels)}] {community.title}: {community.summary}"
            fragments.append(
                ContextFragment(
                    kind=ContextKind.EVIDENCE,
                    source=ContextSource.GRAPH_RAG,
                    trust_level=TrustLevel.UNTRUSTED,
                    sensitivity=Sensitivity.USER_PRIVATE,
                    content=part,
                )
            )

        allocated = self._allocator.allocate(tuple(fragments))
        included = {
            citation_id
            for citation_id in (citation.citation_id for citation in citations.values())
            if f"[{citation_id}]" in allocated.text or f" {citation_id}]" in allocated.text
        }
        return GraphContext(
            text=allocated.text,
            citations=tuple(citation for citation in citations.values() if citation.citation_id in included),
            token_count=allocated.token_count,
            truncated_fragment_count=allocated.truncated_fragment_count,
        )


__all__ = ["GraphContextAssembler"]
