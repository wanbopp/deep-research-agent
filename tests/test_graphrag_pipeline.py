"""不触发模型或网络的 GraphRAG 规则、社区和上下文门禁."""

from hashlib import sha256
from uuid import uuid4

import pytest

from app.graphrag.community import ConnectedComponentsDetector
from app.graphrag.context import GraphContextAssembler
from app.graphrag.entity_resolution import ConservativeEntityResolver
from app.graphrag.schemas import (
    EntityCandidate,
    EntityMention,
    EntityType,
    GlobalGraphResult,
    GraphDocument,
    GraphEdge,
    GraphPath,
    LocalGraphResult,
    RelationCandidate,
    RelationType,
    SourceSpan,
)


class _EmptyEntityLookup:
    """只表达“当前用户图没有候选”的确定性仓储边界."""

    async def find_entities(self, *, user_id, normalized_names):  # noqa: ANN001
        """返回正常空结果；该测试不模拟任何模型或 provider 行为."""
        return ()


@pytest.mark.anyio
async def test_resolution_is_idempotent_and_context_keeps_chunk_citations() -> None:
    """相同来源重跑得到相同 ID，图上下文仍引用原始 chunk."""
    user_id, document_id, chunk_id = uuid4(), uuid4(), uuid4()
    content = "Alpha uses Beta."
    alpha = EntityCandidate(
        local_id="E1",
        canonical_name="Alpha",
        normalized_name="alpha",
        entity_type=EntityType.ORGANIZATION,
        mentions=(EntityMention(mention_id="E1:M1", text="Alpha", span=SourceSpan(start=0, end=5, text="Alpha")),),
        confidence=1.0,
    )
    beta = EntityCandidate(
        local_id="E2",
        canonical_name="Beta",
        normalized_name="beta",
        entity_type=EntityType.TECHNOLOGY,
        mentions=(EntityMention(mention_id="E2:M1", text="Beta", span=SourceSpan(start=11, end=15, text="Beta")),),
        confidence=1.0,
    )
    graph = GraphDocument(
        user_id=user_id,
        document_id=document_id,
        source_chunk_id=chunk_id,
        source_content_sha256=sha256(content.encode()).hexdigest(),
        taxonomy_version="v1",
        extraction_version="v1",
        prompt_version="v1",
        model_name="real-model-recorded-by-runtime",
        entities=(alpha, beta),
        relations=(
            RelationCandidate(
                local_id="R1",
                source_entity_id="E1",
                target_entity_id="E2",
                relation_type=RelationType.USES,
                evidence=SourceSpan(start=0, end=16, text=content),
                confidence=1.0,
            ),
        ),
    )
    resolver = ConservativeEntityResolver(_EmptyEntityLookup())
    first = await resolver.resolve(graph)
    second = await resolver.resolve(graph)
    assert first == second

    edge = GraphEdge(
        source_entity_id=first.relations[0].source_entity_id,
        source_name="Alpha",
        target_entity_id=first.relations[0].target_entity_id,
        target_name="Beta",
        relation_type=RelationType.USES,
        source_chunk_id=chunk_id,
        evidence_text=content,
    )
    drafts = ConnectedComponentsDetector().detect((edge,))
    assert len(drafts) == 1
    assert len(drafts[0].member_entity_ids) == 2

    local = LocalGraphResult(
        linked_entity_ids=(edge.source_entity_id,),
        paths=(
            GraphPath(
                entity_ids=(edge.source_entity_id, edge.target_entity_id),
                entity_names=(edge.source_name, edge.target_name),
                relation_types=(edge.relation_type,),
                source_chunk_ids=(chunk_id,),
                evidence_texts=(content,),
            ),
        ),
        fallback_required=False,
    )
    context = GraphContextAssembler().assemble(
        local=local,
        global_result=GlobalGraphResult(communities=(), scores=()),
    )
    assert "Alpha -> Beta" in context.text
    assert context.citations[0].source_chunk_id == chunk_id


@pytest.mark.anyio
async def test_same_normalized_name_with_different_types_is_not_merged() -> None:
    """同名组织和概念必须形成不同 canonical entity，不能只按字符串合并."""
    user_id, document_id, chunk_id = uuid4(), uuid4(), uuid4()
    content = "Apple announced a device. An apple is a fruit concept."
    graph = GraphDocument(
        user_id=user_id,
        document_id=document_id,
        source_chunk_id=chunk_id,
        source_content_sha256=sha256(content.encode()).hexdigest(),
        taxonomy_version="v1",
        extraction_version="v1",
        prompt_version="v1",
        model_name="real-model-recorded-by-runtime",
        entities=(
            EntityCandidate(
                local_id="E1",
                canonical_name="Apple",
                normalized_name="apple",
                entity_type=EntityType.ORGANIZATION,
                mentions=(
                    EntityMention(
                        mention_id="E1:M1",
                        text="Apple",
                        span=SourceSpan(start=0, end=5, text="Apple"),
                    ),
                ),
                confidence=1.0,
            ),
            EntityCandidate(
                local_id="E2",
                canonical_name="apple",
                normalized_name="apple",
                entity_type=EntityType.CONCEPT,
                mentions=(
                    EntityMention(
                        mention_id="E2:M1",
                        text="apple",
                        span=SourceSpan(start=29, end=34, text="apple"),
                    ),
                ),
                confidence=1.0,
            ),
        ),
        relations=(),
    )
    resolved = await ConservativeEntityResolver(_EmptyEntityLookup()).resolve(graph)
    assert len({entity.entity_id for entity in resolved.entities}) == 2
