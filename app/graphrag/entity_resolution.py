"""保守的实体消歧：规则优先，歧义时创建新实体并保留审计."""

from collections.abc import Sequence
from typing import Protocol
from uuid import NAMESPACE_URL, UUID, uuid5

from app.graphrag.normalizer import normalize_entity_name
from app.graphrag.schemas import (
    EntityCandidate,
    GraphDocument,
    ResolutionDecision,
    ResolutionStrategy,
    ResolvedEntity,
    ResolvedGraphDocument,
    ResolvedMention,
    ResolvedRelation,
    StoredEntity,
)


class EntityLookup(Protocol):
    """实体消歧依赖的只读仓储能力."""

    async def find_entities(self, *, user_id: UUID, normalized_names: Sequence[str]) -> tuple[StoredEntity, ...]:
        """在一个用户图中按 canonical name 或 alias 返回候选实体."""
        ...


class ConservativeEntityResolver:
    """避免同名盲合并的确定性实体消歧器.

    首版只在类型一致且名称/别名命中，并且 identity_hint 不冲突时合并。遇到
    多候选或身份提示冲突时宁可创建新实体，也不把两个人或两家公司错误合并。
    后续可在同一 Protocol 后加入 embedding 候选与真实 LLM judge。
    """

    def __init__(self, lookup: EntityLookup) -> None:
        """保存无请求状态的实体只读仓储."""
        self._lookup = lookup

    async def resolve(self, graph: GraphDocument) -> ResolvedGraphDocument:
        """把 chunk 局部实体转换为用户范围内稳定 canonical entity."""
        lookup_names = tuple(
            dict.fromkeys(
                name
                for entity in graph.entities
                for name in (
                    entity.normalized_name,
                    *(normalize_entity_name(alias) for alias in entity.aliases),
                )
            )
        )
        existing = list(await self._lookup.find_entities(user_id=graph.user_id, normalized_names=lookup_names))
        resolved_by_local: dict[str, ResolvedEntity] = {}
        decisions: list[ResolutionDecision] = []

        for candidate in graph.entities:
            matches = self._matching_entities(candidate, existing)
            if len(matches) == 1:
                stored, strategy = matches[0]
                resolved = ResolvedEntity(
                    entity_id=stored.entity_id,
                    canonical_name=stored.canonical_name,
                    normalized_name=stored.normalized_name,
                    entity_type=stored.entity_type,
                    aliases=tuple(dict.fromkeys((*stored.aliases, *candidate.aliases, candidate.canonical_name))),
                    identity_hint=stored.identity_hint or candidate.identity_hint,
                )
                confidence = candidate.confidence
            else:
                # UUID5 的输入包含可信 user_id、类型、规范名和身份提示。同一用户
                # 对同一候选重跑会得到相同 ID；不同用户永远不会共享图节点。
                identity_key = normalize_entity_name(candidate.identity_hint or "")
                entity_id = uuid5(
                    NAMESPACE_URL,
                    f"graphrag:{graph.user_id}:{candidate.entity_type.value}:"
                    f"{candidate.normalized_name}:{identity_key}",
                )
                resolved = ResolvedEntity(
                    entity_id=entity_id,
                    canonical_name=candidate.canonical_name,
                    normalized_name=candidate.normalized_name,
                    entity_type=candidate.entity_type,
                    aliases=tuple(dict.fromkeys((*candidate.aliases, candidate.canonical_name))),
                    identity_hint=candidate.identity_hint,
                )
                strategy = ResolutionStrategy.CREATED
                confidence = candidate.confidence
                existing.append(
                    StoredEntity(
                        entity_id=resolved.entity_id,
                        canonical_name=resolved.canonical_name,
                        normalized_name=resolved.normalized_name,
                        entity_type=resolved.entity_type,
                        aliases=resolved.aliases,
                        identity_hint=resolved.identity_hint,
                    )
                )

            resolved_by_local[candidate.local_id] = resolved
            decisions.append(
                ResolutionDecision(
                    local_entity_id=candidate.local_id,
                    canonical_entity_id=resolved.entity_id,
                    strategy=strategy,
                    confidence=confidence,
                )
            )

        relations = tuple(
            ResolvedRelation(
                fact_id=uuid5(
                    NAMESPACE_URL,
                    f"graphrag-fact:{graph.user_id}:{graph.source_chunk_id}:"
                    f"{graph.extraction_version}:{relation.local_id}",
                ),
                source_entity_id=resolved_by_local[relation.source_entity_id].entity_id,
                target_entity_id=resolved_by_local[relation.target_entity_id].entity_id,
                relation_type=relation.relation_type,
                evidence=relation.evidence,
                confidence=relation.confidence,
            )
            for relation in graph.relations
        )
        mentions = tuple(
            ResolvedMention(
                entity_id=resolved_by_local[candidate.local_id].entity_id,
                mention=mention,
            )
            for candidate in graph.entities
            for mention in candidate.mentions
        )
        unique_entities = {entity.entity_id: entity for entity in resolved_by_local.values()}
        return ResolvedGraphDocument(
            user_id=graph.user_id,
            document_id=graph.document_id,
            source_chunk_id=graph.source_chunk_id,
            source_content_sha256=graph.source_content_sha256,
            taxonomy_version=graph.taxonomy_version,
            extraction_version=graph.extraction_version,
            prompt_version=graph.prompt_version,
            model_name=graph.model_name,
            repair_count=graph.repair_count,
            rejection_code=graph.rejection_code,
            entities=tuple(unique_entities.values()),
            mentions=mentions,
            relations=relations,
            decisions=tuple(decisions),
        )

    @staticmethod
    def _matching_entities(
        candidate: EntityCandidate,
        existing: Sequence[StoredEntity],
    ) -> list[tuple[StoredEntity, ResolutionStrategy]]:
        """返回没有身份冲突的精确名称/别名候选."""
        candidate_hint = normalize_entity_name(candidate.identity_hint or "")
        matches: list[tuple[StoredEntity, ResolutionStrategy]] = []
        for stored in existing:
            if stored.entity_type != candidate.entity_type:
                continue
            stored_hint = normalize_entity_name(stored.identity_hint or "")
            if candidate_hint and stored_hint and candidate_hint != stored_hint:
                continue
            aliases = {normalize_entity_name(alias) for alias in stored.aliases}
            if candidate.normalized_name == stored.normalized_name:
                matches.append((stored, ResolutionStrategy.EXACT_NAME))
            elif candidate.normalized_name in aliases or stored.normalized_name in {
                normalize_entity_name(alias) for alias in candidate.aliases
            }:
                matches.append((stored, ResolutionStrategy.ALIAS))
        return matches


__all__ = ["ConservativeEntityResolver", "EntityLookup"]
