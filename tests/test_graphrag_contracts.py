"""GraphRAG 候选图中最重要的跨对象不变量测试."""

import pytest
from pydantic import ValidationError

from app.graphrag.normalizer import locate_source_span, normalize_entity_name
from app.graphrag.schemas import (
    ExtractedEntityCandidate,
    ExtractedRelationCandidate,
    GraphExtractionPayload,
)


def test_graph_payload_rejects_dangling_relation_and_normalizes_unknown_types() -> None:
    """未知 taxonomy 可降级，但悬空关系绝不能越过结构校验."""
    # model_validate 模拟 provider 返回 JSON 后的运行时校验；普通构造器则由
    # Pyright 要求直接传入 EntityType，二者分别覆盖静态与动态输入边界。
    entity = ExtractedEntityCandidate.model_validate(
        {
            "local_id": "E1",
            "canonical_name": "Apple Inc.",
            "entity_type": "company-not-in-v1",
            "mentions": ["Apple"],
            "confidence": 0.9,
        }
    )
    assert entity.entity_type.value == "unknown"

    with pytest.raises(ValidationError, match="relation endpoints"):
        GraphExtractionPayload(
            entities=(entity,),
            relations=(
                ExtractedRelationCandidate.model_validate(
                    {
                        "local_id": "R1",
                        "source_entity_id": "E1",
                        "target_entity_id": "E2",
                        "relation_type": "invented-predicate",
                        "evidence_text": "Apple released a product",
                        "confidence": 0.8,
                    }
                ),
            ),
        )


def test_source_span_and_name_normalization_preserve_evidence_boundary() -> None:
    """名称可用于匹配，但原始证据必须保持逐字可回查."""
    source = "Ａｐｐｌｅ  Inc. released a product."
    span = locate_source_span(source, "Ａｐｐｌｅ  Inc.")
    assert source[span.start : span.end] == span.text
    assert normalize_entity_name(span.text) == "apple inc."
