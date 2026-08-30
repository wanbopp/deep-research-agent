"""GraphRAG 候选图、可信来源与检索结果的数据约定."""

from enum import StrEnum
from typing import Self
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

MAX_GRAPH_TEXT_LENGTH = 500
MAX_GRAPH_EVIDENCE_LENGTH = 1000


class _StrictGraphModel(BaseModel):
    """GraphRAG 值对象共享的严格输入边界.

    图对象会跨越 LLM、应用服务和 Neo4j 三层。禁止额外字段可以尽早发现
    prompt/schema 漂移；不可变对象则避免校验通过后被某个步骤原地篡改。
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        str_strip_whitespace=True,
        hide_input_in_errors=True,
    )


class EntityType(StrEnum):
    """首版受控实体分类；未知领域概念显式降级而非创建新 label."""

    PERSON = "person"
    ORGANIZATION = "organization"
    LOCATION = "location"
    PRODUCT = "product"
    TECHNOLOGY = "technology"
    EVENT = "event"
    CONCEPT = "concept"
    UNKNOWN = "unknown"

    @classmethod
    def _missing_(cls, value: object) -> "EntityType":
        """把模型产生的未注册类型收敛为 unknown."""
        return cls.UNKNOWN


class RelationType(StrEnum):
    """首版受控关系分类；Neo4j 中使用固定 FACT 边保存该值."""

    PART_OF = "part_of"
    LOCATED_IN = "located_in"
    CREATED_BY = "created_by"
    WORKS_FOR = "works_for"
    USES = "uses"
    PRODUCES = "produces"
    RELATED_TO = "related_to"
    UNKNOWN = "unknown"

    @classmethod
    def _missing_(cls, value: object) -> "RelationType":
        """把开放领域中的未注册谓词收敛为 unknown."""
        return cls.UNKNOWN


class SourceSpan(_StrictGraphModel):
    """证据在一个可信 chunk 中的左闭右开字符区间."""

    start: int = Field(ge=0, description="证据首字符在 chunk 中的下标")
    end: int = Field(gt=0, description="证据末字符后一位的下标")
    text: str = Field(min_length=1, max_length=MAX_GRAPH_EVIDENCE_LENGTH)

    @model_validator(mode="after")
    def validate_order(self) -> Self:
        """拒绝空区间和反向区间."""
        if self.end <= self.start:
            raise ValueError("source span end must be greater than start")
        return self


class ExtractedEntityCandidate(_StrictGraphModel):
    """LLM 提出的未可信实体候选，不包含任何用户或数据库身份."""

    local_id: str = Field(min_length=1, max_length=64)
    canonical_name: str = Field(min_length=1, max_length=MAX_GRAPH_TEXT_LENGTH)
    entity_type: EntityType
    mentions: tuple[str, ...] = Field(min_length=1, max_length=20)
    aliases: tuple[str, ...] = Field(default=(), max_length=20)
    # identity_hint 描述区分同名对象所需的短上下文，例如“美国科技公司”与
    # “英国唱片公司”。它不是最终 ID，实体消歧仍由服务端完成并记录审计。
    identity_hint: str | None = Field(default=None, max_length=MAX_GRAPH_TEXT_LENGTH)
    confidence: float = Field(ge=0.0, le=1.0)

    @field_validator("mentions", "aliases")
    @classmethod
    def reject_blank_items(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        """禁止空 mention/alias 进入规范化和实体消歧."""
        if any(not item.strip() for item in values):
            raise ValueError("mention and alias values must not be blank")
        return values


class ExtractedRelationCandidate(_StrictGraphModel):
    """LLM 提出的未可信关系候选，通过局部 ID 引用同批实体."""

    local_id: str = Field(min_length=1, max_length=64)
    source_entity_id: str = Field(min_length=1, max_length=64)
    target_entity_id: str = Field(min_length=1, max_length=64)
    relation_type: RelationType
    evidence_text: str = Field(min_length=1, max_length=MAX_GRAPH_EVIDENCE_LENGTH)
    confidence: float = Field(ge=0.0, le=1.0)


class GraphExtractionPayload(_StrictGraphModel):
    """一次 structured LLM 调用返回的完整候选集合."""

    entities: tuple[ExtractedEntityCandidate, ...] = Field(default=(), max_length=100)
    relations: tuple[ExtractedRelationCandidate, ...] = Field(default=(), max_length=200)

    @model_validator(mode="after")
    def validate_local_graph(self) -> Self:
        """拒绝重复局部 ID 和指向不存在实体的悬空关系."""
        entity_ids = [entity.local_id for entity in self.entities]
        relation_ids = [relation.local_id for relation in self.relations]
        if len(entity_ids) != len(set(entity_ids)):
            raise ValueError("entity local_id values must be unique")
        if len(relation_ids) != len(set(relation_ids)):
            raise ValueError("relation local_id values must be unique")
        known_entities = set(entity_ids)
        for relation in self.relations:
            if relation.source_entity_id not in known_entities or relation.target_entity_id not in known_entities:
                raise ValueError("relation endpoints must reference entities in the same payload")
            if relation.source_entity_id == relation.target_entity_id:
                raise ValueError("self-referential relation candidates are not accepted")
        return self


class EntityMention(_StrictGraphModel):
    """已由服务端定位到可信 chunk 原文的一次实体出现."""

    mention_id: str = Field(min_length=1, max_length=128)
    text: str = Field(min_length=1, max_length=MAX_GRAPH_TEXT_LENGTH)
    span: SourceSpan


class EntityCandidate(_StrictGraphModel):
    """带原文位置但尚未跨 chunk 消歧的实体候选."""

    local_id: str = Field(min_length=1, max_length=64)
    canonical_name: str = Field(min_length=1, max_length=MAX_GRAPH_TEXT_LENGTH)
    normalized_name: str = Field(min_length=1, max_length=MAX_GRAPH_TEXT_LENGTH)
    entity_type: EntityType
    mentions: tuple[EntityMention, ...] = Field(min_length=1, max_length=20)
    aliases: tuple[str, ...] = Field(default=(), max_length=20)
    identity_hint: str | None = Field(default=None, max_length=MAX_GRAPH_TEXT_LENGTH)
    confidence: float = Field(ge=0.0, le=1.0)


class RelationCandidate(_StrictGraphModel):
    """带可信原文证据但仍待验证和消歧的关系候选."""

    local_id: str = Field(min_length=1, max_length=64)
    source_entity_id: str = Field(min_length=1, max_length=64)
    target_entity_id: str = Field(min_length=1, max_length=64)
    relation_type: RelationType
    evidence: SourceSpan
    confidence: float = Field(ge=0.0, le=1.0)


class GraphDocument(_StrictGraphModel):
    """服务端绑定可信归属和版本后形成的 chunk 级候选图."""

    user_id: UUID
    document_id: UUID
    source_chunk_id: UUID
    source_content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    taxonomy_version: str = Field(min_length=1, max_length=64)
    extraction_version: str = Field(min_length=1, max_length=64)
    prompt_version: str = Field(min_length=1, max_length=64)
    model_name: str = Field(min_length=1, max_length=255)
    repair_count: int = Field(default=0, ge=0, le=2)
    rejection_code: str | None = Field(default=None, max_length=64)
    entities: tuple[EntityCandidate, ...] = Field(default=(), max_length=100)
    relations: tuple[RelationCandidate, ...] = Field(default=(), max_length=200)

    @model_validator(mode="after")
    def validate_bound_graph(self) -> Self:
        """在写图前再次验证唯一 ID、mention ID 与关系端点."""
        entity_ids = [entity.local_id for entity in self.entities]
        mention_ids = [mention.mention_id for entity in self.entities for mention in entity.mentions]
        relation_ids = [relation.local_id for relation in self.relations]
        if len(entity_ids) != len(set(entity_ids)):
            raise ValueError("bound entity local_id values must be unique")
        if len(mention_ids) != len(set(mention_ids)):
            raise ValueError("mention_id values must be unique")
        if len(relation_ids) != len(set(relation_ids)):
            raise ValueError("bound relation local_id values must be unique")
        known_entities = set(entity_ids)
        if any(
            relation.source_entity_id not in known_entities or relation.target_entity_id not in known_entities
            for relation in self.relations
        ):
            raise ValueError("bound relation endpoints must exist in GraphDocument")
        return self


class ResolutionStrategy(StrEnum):
    """实体候选形成 canonical entity 的可审计判定方式."""

    CREATED = "created"
    EXACT_NAME = "exact_name"
    ALIAS = "alias"
    JUDGED = "judged"


class StoredEntity(_StrictGraphModel):
    """实体消歧器从图仓储读取的现有 canonical entity 投影."""

    entity_id: UUID
    canonical_name: str = Field(min_length=1, max_length=MAX_GRAPH_TEXT_LENGTH)
    normalized_name: str = Field(min_length=1, max_length=MAX_GRAPH_TEXT_LENGTH)
    entity_type: EntityType
    aliases: tuple[str, ...] = Field(default=(), max_length=100)
    identity_hint: str | None = Field(default=None, max_length=MAX_GRAPH_TEXT_LENGTH)


class ResolutionDecision(_StrictGraphModel):
    """一次候选实体归并决策的最小审计记录."""

    local_entity_id: str = Field(min_length=1, max_length=64)
    canonical_entity_id: UUID
    strategy: ResolutionStrategy
    confidence: float = Field(ge=0.0, le=1.0)


class ResolvedEntity(_StrictGraphModel):
    """已经获得稳定 canonical ID、可安全写入图仓储的实体."""

    entity_id: UUID
    canonical_name: str = Field(min_length=1, max_length=MAX_GRAPH_TEXT_LENGTH)
    normalized_name: str = Field(min_length=1, max_length=MAX_GRAPH_TEXT_LENGTH)
    entity_type: EntityType
    aliases: tuple[str, ...] = Field(default=(), max_length=100)
    identity_hint: str | None = Field(default=None, max_length=MAX_GRAPH_TEXT_LENGTH)


class ResolvedRelation(_StrictGraphModel):
    """关系端点完成消歧后的来源绑定事实边."""

    fact_id: UUID
    source_entity_id: UUID
    target_entity_id: UUID
    relation_type: RelationType
    evidence: SourceSpan
    confidence: float = Field(ge=0.0, le=1.0)


class ResolvedMention(_StrictGraphModel):
    """一个原文 mention 到 canonical entity 的已解析连接."""

    entity_id: UUID
    mention: EntityMention


class ResolvedGraphDocument(_StrictGraphModel):
    """GraphRepository 唯一接受的已消歧 chunk 图."""

    user_id: UUID
    document_id: UUID
    source_chunk_id: UUID
    source_content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    taxonomy_version: str = Field(min_length=1, max_length=64)
    extraction_version: str = Field(min_length=1, max_length=64)
    prompt_version: str = Field(min_length=1, max_length=64)
    model_name: str = Field(min_length=1, max_length=255)
    repair_count: int = Field(default=0, ge=0, le=2)
    rejection_code: str | None = Field(default=None, max_length=64)
    entities: tuple[ResolvedEntity, ...]
    mentions: tuple[ResolvedMention, ...]
    relations: tuple[ResolvedRelation, ...]
    decisions: tuple[ResolutionDecision, ...]

    @model_validator(mode="after")
    def validate_resolved_graph(self) -> Self:
        """确保事实端点和审计决策都引用本图已解析实体."""
        entity_ids = {entity.entity_id for entity in self.entities}
        if len(entity_ids) != len(self.entities):
            raise ValueError("resolved entity_id values must be unique")
        if any(
            relation.source_entity_id not in entity_ids or relation.target_entity_id not in entity_ids
            for relation in self.relations
        ):
            raise ValueError("resolved relation endpoints must exist")
        if any(mention.entity_id not in entity_ids for mention in self.mentions):
            raise ValueError("resolved mentions must reference resolved entities")
        decided = {decision.canonical_entity_id for decision in self.decisions}
        if not decided.issubset(entity_ids):
            raise ValueError("resolution decisions must reference resolved entities")
        return self


class GraphPath(_StrictGraphModel):
    """Local GraphRAG 返回的一条可解释实体路径."""

    entity_ids: tuple[UUID, ...] = Field(min_length=2)
    entity_names: tuple[str, ...] = Field(min_length=2)
    relation_types: tuple[RelationType, ...] = Field(min_length=1)
    source_chunk_ids: tuple[UUID, ...] = Field(min_length=1)
    evidence_texts: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_path_shape(self) -> Self:
        """一条 n 条边的路径必须恰好连接 n+1 个实体."""
        if len(self.entity_ids) != len(self.entity_names):
            raise ValueError("entity ids and names must have the same length")
        if len(self.entity_ids) != len(self.relation_types) + 1:
            raise ValueError("graph path must contain one more entity than relations")
        return self


class CommunityRecord(_StrictGraphModel):
    """社区成员、摘要、版本与来源引用的持久化投影."""

    community_id: str = Field(min_length=1, max_length=128)
    member_entity_ids: tuple[UUID, ...] = Field(min_length=1)
    title: str = Field(min_length=1, max_length=200)
    summary: str = Field(min_length=1, max_length=4000)
    source_chunk_ids: tuple[UUID, ...] = Field(min_length=1)
    algorithm_version: str = Field(min_length=1, max_length=64)
    summary_version: str = Field(min_length=1, max_length=64)
    model_name: str = Field(min_length=1, max_length=255)


class GraphEdge(_StrictGraphModel):
    """社区检测使用的最小、来源绑定图边投影."""

    source_entity_id: UUID
    source_name: str = Field(min_length=1, max_length=MAX_GRAPH_TEXT_LENGTH)
    target_entity_id: UUID
    target_name: str = Field(min_length=1, max_length=MAX_GRAPH_TEXT_LENGTH)
    relation_type: RelationType
    source_chunk_id: UUID
    evidence_text: str = Field(min_length=1, max_length=MAX_GRAPH_EVIDENCE_LENGTH)


class QueryEntityPayload(_StrictGraphModel):
    """真实 LLM 从查询中提取的待链接实体名称."""

    names: tuple[str, ...] = Field(default=(), max_length=10)


class LocalGraphResult(_StrictGraphModel):
    """Local GraphRAG 的实体链接和邻域路径结果."""

    linked_entity_ids: tuple[UUID, ...]
    paths: tuple[GraphPath, ...]
    fallback_required: bool


class GlobalGraphResult(_StrictGraphModel):
    """Global GraphRAG 按语义相关性选出的社区摘要."""

    communities: tuple[CommunityRecord, ...]
    scores: tuple[float, ...]

    @model_validator(mode="after")
    def validate_scores(self) -> Self:
        """每个社区必须有一个同位置的相似度分数."""
        if len(self.communities) != len(self.scores):
            raise ValueError("communities and scores must have the same length")
        return self


class GraphCitation(_StrictGraphModel):
    """图上下文引用到的原始 chunk，不把社区摘要伪装成原始来源."""

    citation_id: str = Field(pattern=r"^G[0-9A-F]{8}$")
    source_chunk_id: UUID


class GraphContext(_StrictGraphModel):
    """送给后续 DeepResearch Agent 的图证据文本和引用表."""

    text: str
    citations: tuple[GraphCitation, ...]
    token_count: int = Field(default=0, ge=0)
    truncated_fragment_count: int = Field(default=0, ge=0)


class CommunityMapResult(_StrictGraphModel):
    """Global map 阶段的局部结论与服务端绑定来源."""

    community_id: str
    claim: str = Field(min_length=1, max_length=3000)
    source_chunk_ids: tuple[UUID, ...] = Field(min_length=1)


class GlobalGraphAnswer(_StrictGraphModel):
    """Global reduce 阶段的最终回答与不可伪造引用集合."""

    answer: str = Field(min_length=1, max_length=8000)
    source_chunk_ids: tuple[UUID, ...] = Field(min_length=1)
    map_results: tuple[CommunityMapResult, ...] = Field(min_length=1)


__all__ = [
    "EntityCandidate",
    "EntityMention",
    "EntityType",
    "ExtractedEntityCandidate",
    "ExtractedRelationCandidate",
    "GraphDocument",
    "GraphExtractionPayload",
    "RelationCandidate",
    "RelationType",
    "ResolutionDecision",
    "ResolutionStrategy",
    "ResolvedEntity",
    "ResolvedGraphDocument",
    "ResolvedMention",
    "ResolvedRelation",
    "SourceSpan",
    "StoredEntity",
    "CommunityRecord",
    "GlobalGraphResult",
    "GlobalGraphAnswer",
    "GraphCitation",
    "GraphContext",
    "GraphEdge",
    "GraphPath",
    "LocalGraphResult",
    "CommunityMapResult",
    "QueryEntityPayload",
]
