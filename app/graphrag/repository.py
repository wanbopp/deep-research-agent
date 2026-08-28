"""参数化 Neo4j GraphRepository 与用户隔离图查询."""

from collections.abc import Sequence
from typing import Any, Final
from uuid import UUID

from neo4j import AsyncDriver, RoutingControl
from neo4j.exceptions import Neo4jError

from app.graphrag.errors import GraphRepositoryError
from app.graphrag.normalizer import normalize_entity_name
from app.graphrag.schemas import (
    CommunityRecord,
    EntityType,
    GraphEdge,
    GraphPath,
    RelationType,
    ResolvedGraphDocument,
    StoredEntity,
)

_SCHEMA_QUERIES: Final = (
    "CREATE CONSTRAINT graphrag_document_key IF NOT EXISTS FOR (n:GraphDocument) REQUIRE n.key IS UNIQUE",
    "CREATE CONSTRAINT graphrag_chunk_key IF NOT EXISTS FOR (n:GraphChunk) REQUIRE n.key IS UNIQUE",
    "CREATE CONSTRAINT graphrag_entity_key IF NOT EXISTS FOR (n:Entity) REQUIRE n.key IS UNIQUE",
    "CREATE CONSTRAINT graphrag_mention_key IF NOT EXISTS FOR (n:EntityMention) REQUIRE n.key IS UNIQUE",
    "CREATE CONSTRAINT graphrag_resolution_audit_key IF NOT EXISTS FOR (n:ResolutionAudit) REQUIRE n.key IS UNIQUE",
    "CREATE CONSTRAINT graphrag_community_key IF NOT EXISTS FOR (n:Community) REQUIRE n.key IS UNIQUE",
    "CREATE INDEX graphrag_entity_lookup IF NOT EXISTS FOR (n:Entity) ON (n.user_id, n.normalized_name)",
    "CREATE INDEX graphrag_fact_owner IF NOT EXISTS FOR ()-[r:FACT]-() ON (r.user_id)",
)

_UPSERT_DOCUMENT_CHUNK: Final = """
MERGE (d:GraphDocument {key: $document_key})
SET d.id = $document_id, d.user_id = $user_id
MERGE (c:GraphChunk {key: $chunk_key})
SET c.id = $chunk_id, c.user_id = $user_id, c.document_id = $document_id,
    c.content_sha256 = $content_sha256, c.extraction_version = $extraction_version,
    c.taxonomy_version = $taxonomy_version, c.prompt_version = $prompt_version,
    c.model_name = $model_name, c.repair_count = $repair_count,
    c.rejection_code = $rejection_code
MERGE (d)-[:HAS_CHUNK]->(c)
"""
_DELETE_CHUNK_DERIVED: Final = """
MATCH (c:GraphChunk {key: $chunk_key})
OPTIONAL MATCH (m:EntityMention)-[:IN_CHUNK]->(c)
OPTIONAL MATCH (a:ResolutionAudit)-[:IN_CHUNK]->(c)
DETACH DELETE m, a
"""
_DELETE_CHUNK_FACTS: Final = """
MATCH ()-[f:FACT {user_id: $user_id, chunk_id: $chunk_id}]->()
DELETE f
"""
_UPSERT_ENTITIES: Final = """
UNWIND $rows AS row
MERGE (e:Entity {key: row.key})
SET e.id = row.id, e.user_id = $user_id, e.canonical_name = row.canonical_name,
    e.normalized_name = row.normalized_name, e.entity_type = row.entity_type,
    e.aliases = row.aliases, e.identity_hint = row.identity_hint
"""
_UPSERT_MENTIONS: Final = """
UNWIND $rows AS row
MATCH (e:Entity {key: row.entity_key})
MATCH (c:GraphChunk {key: $chunk_key})
MERGE (m:EntityMention {key: row.key})
SET m.id = row.id, m.user_id = $user_id, m.chunk_id = $chunk_id,
    m.text = row.text, m.start = row.start, m.end = row.end
MERGE (m)-[:REFERS_TO]->(e)
MERGE (m)-[:IN_CHUNK]->(c)
"""
_UPSERT_FACTS: Final = """
UNWIND $rows AS row
MATCH (source:Entity {key: row.source_key})
MATCH (target:Entity {key: row.target_key})
MERGE (source)-[f:FACT {key: row.key}]->(target)
SET f.id = row.id, f.user_id = $user_id, f.chunk_id = $chunk_id,
    f.relation_type = row.relation_type, f.evidence_text = row.evidence_text,
    f.evidence_start = row.evidence_start, f.evidence_end = row.evidence_end,
    f.confidence = row.confidence, f.extraction_version = $extraction_version
"""
_UPSERT_RESOLUTION_AUDITS: Final = """
UNWIND $rows AS row
MATCH (e:Entity {key: row.entity_key})
MATCH (c:GraphChunk {key: $chunk_key})
MERGE (a:ResolutionAudit {key: row.key})
SET a.user_id = $user_id, a.chunk_id = $chunk_id,
    a.local_entity_id = row.local_entity_id, a.strategy = row.strategy,
    a.confidence = row.confidence, a.extraction_version = $extraction_version
MERGE (a)-[:RESOLVED_TO]->(e)
MERGE (a)-[:IN_CHUNK]->(c)
"""
_FIND_ENTITIES: Final = """
MATCH (e:Entity {user_id: $user_id})
WHERE e.normalized_name IN $names
   OR any(alias IN coalesce(e.aliases, []) WHERE alias IN $names)
RETURN e.id AS id, e.canonical_name AS canonical_name,
       e.normalized_name AS normalized_name, e.entity_type AS entity_type,
       e.aliases AS aliases, e.identity_hint AS identity_hint
"""
_LOCAL_PATHS_DEPTH_1: Final = """
MATCH p=(start:Entity {user_id: $user_id})-[rels:FACT]-(other:Entity {user_id: $user_id})
WHERE start.id IN $entity_ids AND all(r IN rels WHERE r.user_id = $user_id)
RETURN nodes(p) AS nodes, relationships(p) AS rels LIMIT $limit
"""
_LOCAL_PATHS_DEPTH_2: Final = """
MATCH p=(start:Entity {user_id: $user_id})-[rels:FACT*1..2]-(other:Entity {user_id: $user_id})
WHERE start.id IN $entity_ids AND all(r IN rels WHERE r.user_id = $user_id)
RETURN nodes(p) AS nodes, relationships(p) AS rels LIMIT $limit
"""
_FETCH_GRAPH: Final = """
MATCH (source:Entity {user_id: $user_id})-[fact:FACT {user_id: $user_id}]->(target:Entity {user_id: $user_id})
RETURN source.id AS source_id, source.canonical_name AS source_name,
       target.id AS target_id, target.canonical_name AS target_name,
       fact.relation_type AS relation_type, fact.chunk_id AS chunk_id,
       fact.evidence_text AS evidence_text
"""
_DELETE_USER_GRAPH: Final = """
MATCH (n {user_id: $user_id})
WHERE n:GraphDocument OR n:GraphChunk OR n:Entity OR n:EntityMention
   OR n:ResolutionAudit OR n:Community
DETACH DELETE n
"""
_DELETE_DOCUMENT_GRAPH: Final = """
MATCH (d:GraphDocument {key: $document_key})
OPTIONAL MATCH (d)-[:HAS_CHUNK]->(c:GraphChunk)
OPTIONAL MATCH (m:EntityMention)-[:IN_CHUNK]->(c)
OPTIONAL MATCH (a:ResolutionAudit)-[:IN_CHUNK]->(c)
DETACH DELETE m, a, c, d
WITH 1 AS ignored
MATCH ()-[f:FACT {user_id: $user_id}]->()
WHERE f.chunk_id IN $chunk_ids
DELETE f
"""
_DELETE_ORPHAN_ENTITIES: Final = """
MATCH (e:Entity {user_id: $user_id})
WHERE NOT (e)<-[:REFERS_TO]-(:EntityMention)
  AND NOT (e)-[:FACT]-()
DETACH DELETE e
"""
_INVALIDATE_COMMUNITIES: Final = """
MATCH (c:Community {user_id: $user_id})
DETACH DELETE c
"""


class Neo4jGraphRepository:
    """借用 lifespan AsyncDriver 的用户隔离图仓储."""

    def __init__(self, driver: AsyncDriver, *, database: str | None = None) -> None:
        """保存共享 driver；仓储不拥有也不关闭它."""
        self._driver = driver
        self._database = database

    async def setup_schema(self) -> None:
        """幂等创建 Community Edition 可用的 constraint/index."""
        try:
            for query in _SCHEMA_QUERIES:
                await self._driver.execute_query(query, database_=self._database)
        except Neo4jError as error:
            raise GraphRepositoryError("Neo4j GraphRAG schema setup failed") from error

    async def find_entities(self, *, user_id: UUID, normalized_names: Sequence[str]) -> tuple[StoredEntity, ...]:
        """只在可信 user_id 的图命名空间中寻找名称或别名候选."""
        if not normalized_names:
            return ()
        names = [normalize_entity_name(name) for name in normalized_names]
        try:
            records, _, _ = await self._driver.execute_query(
                _FIND_ENTITIES,
                {"user_id": str(user_id), "names": names},
                routing_=RoutingControl.READ,
                database_=self._database,
            )
        except Neo4jError as error:
            raise GraphRepositoryError("Neo4j entity lookup failed") from error
        return tuple(
            StoredEntity(
                entity_id=UUID(record["id"]),
                canonical_name=record["canonical_name"],
                normalized_name=record["normalized_name"],
                entity_type=EntityType(record["entity_type"]),
                aliases=tuple(record["aliases"] or ()),
                identity_hint=record["identity_hint"],
            )
            for record in records
        )

    async def replace_graph_document(self, graph: ResolvedGraphDocument) -> None:
        """在一个托管事务中幂等替换指定 chunk 派生的 mention 与事实.

        Entity 使用稳定 canonical ID 执行 MERGE；mention 和 FACT 则先按可信
        chunk 删除后重建。这样同一 extraction version 重跑不会增长计数，新版本
        也不会把旧事实残留在图中。事务失败时整批回滚，不留下半张图。
        """
        user_id = str(graph.user_id)
        document_key = f"{user_id}:{graph.document_id}"
        chunk_key = f"{user_id}:{graph.source_chunk_id}"
        entity_rows = [
            {
                "key": f"{user_id}:{entity.entity_id}",
                "id": str(entity.entity_id),
                "canonical_name": entity.canonical_name,
                "normalized_name": entity.normalized_name,
                "entity_type": entity.entity_type.value,
                "aliases": [normalize_entity_name(alias) for alias in entity.aliases],
                "identity_hint": entity.identity_hint,
            }
            for entity in graph.entities
        ]
        mention_rows = [
            {
                "key": f"{user_id}:{graph.source_chunk_id}:{item.mention.mention_id}",
                "id": item.mention.mention_id,
                "entity_key": f"{user_id}:{item.entity_id}",
                "text": item.mention.text,
                "start": item.mention.span.start,
                "end": item.mention.span.end,
            }
            for item in graph.mentions
        ]
        fact_rows = [
            {
                "key": f"{user_id}:{relation.fact_id}",
                "id": str(relation.fact_id),
                "source_key": f"{user_id}:{relation.source_entity_id}",
                "target_key": f"{user_id}:{relation.target_entity_id}",
                "relation_type": relation.relation_type.value,
                "evidence_text": relation.evidence.text,
                "evidence_start": relation.evidence.start,
                "evidence_end": relation.evidence.end,
                "confidence": relation.confidence,
            }
            for relation in graph.relations
        ]
        decision_rows = [
            {
                "key": f"{user_id}:{graph.source_chunk_id}:{decision.local_entity_id}",
                "entity_key": f"{user_id}:{decision.canonical_entity_id}",
                "local_entity_id": decision.local_entity_id,
                "strategy": decision.strategy.value,
                "confidence": decision.confidence,
            }
            for decision in graph.decisions
        ]

        async def write(tx: Any) -> None:
            """在同一事务中执行固定 Cypher；所有动态值都走参数."""
            await (await tx.run(_DELETE_CHUNK_FACTS, user_id=user_id, chunk_id=str(graph.source_chunk_id))).consume()
            await (await tx.run(_DELETE_CHUNK_DERIVED, chunk_key=chunk_key)).consume()
            await (
                await tx.run(
                    _UPSERT_DOCUMENT_CHUNK,
                    document_key=document_key,
                    chunk_key=chunk_key,
                    user_id=user_id,
                    document_id=str(graph.document_id),
                    chunk_id=str(graph.source_chunk_id),
                    content_sha256=graph.source_content_sha256,
                    extraction_version=graph.extraction_version,
                    taxonomy_version=graph.taxonomy_version,
                    prompt_version=graph.prompt_version,
                    model_name=graph.model_name,
                    repair_count=graph.repair_count,
                    rejection_code=graph.rejection_code,
                )
            ).consume()
            await (await tx.run(_UPSERT_ENTITIES, rows=entity_rows, user_id=user_id)).consume()
            await (
                await tx.run(
                    _UPSERT_MENTIONS,
                    rows=mention_rows,
                    user_id=user_id,
                    chunk_id=str(graph.source_chunk_id),
                    chunk_key=chunk_key,
                )
            ).consume()
            await (
                await tx.run(
                    _UPSERT_FACTS,
                    rows=fact_rows,
                    user_id=user_id,
                    chunk_id=str(graph.source_chunk_id),
                    extraction_version=graph.extraction_version,
                )
            ).consume()
            await (
                await tx.run(
                    _UPSERT_RESOLUTION_AUDITS,
                    rows=decision_rows,
                    user_id=user_id,
                    chunk_id=str(graph.source_chunk_id),
                    chunk_key=chunk_key,
                    extraction_version=graph.extraction_version,
                )
            ).consume()

        try:
            async with self._driver.session(database=self._database) as session:
                await session.execute_write(write)
        except Neo4jError as error:
            raise GraphRepositoryError("Neo4j graph document write failed") from error

    async def local_paths(
        self,
        *,
        user_id: UUID,
        entity_ids: Sequence[UUID],
        depth: int = 2,
        limit: int = 20,
    ) -> tuple[GraphPath, ...]:
        """从已链接实体向外扩展最多两跳，并保留每条边的 chunk 证据."""
        if depth not in (1, 2):
            raise ValueError("depth must be 1 or 2")
        if limit <= 0 or limit > 100:
            raise ValueError("limit must be between 1 and 100")
        query = _LOCAL_PATHS_DEPTH_1 if depth == 1 else _LOCAL_PATHS_DEPTH_2
        try:
            records, _, _ = await self._driver.execute_query(
                query,
                {"user_id": str(user_id), "entity_ids": [str(value) for value in entity_ids], "limit": limit},
                routing_=RoutingControl.READ,
                database_=self._database,
            )
        except Neo4jError as error:
            raise GraphRepositoryError("Neo4j local graph query failed") from error
        return tuple(self._record_to_path(record) for record in records)

    async def fetch_edges(self, *, user_id: UUID) -> tuple[GraphEdge, ...]:
        """返回社区算法所需的最小用户图投影，不泄漏其他用户节点."""
        try:
            records, _, _ = await self._driver.execute_query(
                _FETCH_GRAPH,
                {"user_id": str(user_id)},
                routing_=RoutingControl.READ,
                database_=self._database,
            )
        except Neo4jError as error:
            raise GraphRepositoryError("Neo4j graph projection failed") from error
        return tuple(
            GraphEdge(
                source_entity_id=UUID(record["source_id"]),
                source_name=record["source_name"],
                target_entity_id=UUID(record["target_id"]),
                target_name=record["target_name"],
                relation_type=RelationType(record["relation_type"]),
                source_chunk_id=UUID(record["chunk_id"]),
                evidence_text=record["evidence_text"],
            )
            for record in records
        )

    async def replace_communities(self, *, user_id: UUID, communities: Sequence[CommunityRecord]) -> None:
        """替换一个用户的社区快照；成员关系与摘要版本同时更新."""
        delete_query = "MATCH (c:Community {user_id: $user_id}) DETACH DELETE c"
        upsert_query = """
UNWIND $rows AS row
CREATE (c:Community {key: row.key, id: row.id, user_id: $user_id,
 title: row.title, summary: row.summary, source_chunk_ids: row.source_chunk_ids,
 algorithm_version: row.algorithm_version, summary_version: row.summary_version,
 model_name: row.model_name})
WITH c, row
UNWIND row.member_entity_ids AS member_id
MATCH (e:Entity {user_id: $user_id, id: member_id})
MERGE (e)-[:MEMBER_OF]->(c)
"""
        rows = [
            {
                "key": f"{user_id}:{item.community_id}",
                "id": item.community_id,
                "title": item.title,
                "summary": item.summary,
                "source_chunk_ids": [str(value) for value in item.source_chunk_ids],
                "algorithm_version": item.algorithm_version,
                "summary_version": item.summary_version,
                "model_name": item.model_name,
                "member_entity_ids": [str(value) for value in item.member_entity_ids],
            }
            for item in communities
        ]
        try:
            async with self._driver.session(database=self._database) as session:

                async def write(tx: Any) -> None:
                    await (await tx.run(delete_query, user_id=str(user_id))).consume()
                    await (await tx.run(upsert_query, rows=rows, user_id=str(user_id))).consume()

                await session.execute_write(write)
        except Neo4jError as error:
            raise GraphRepositoryError("Neo4j community write failed") from error

    async def list_communities(self, *, user_id: UUID) -> tuple[CommunityRecord, ...]:
        """读取用户自己的社区摘要和成员版本."""
        query = """
MATCH (e:Entity {user_id: $user_id})-[:MEMBER_OF]->(c:Community {user_id: $user_id})
RETURN c.id AS id, c.title AS title, c.summary AS summary,
       c.source_chunk_ids AS source_chunk_ids, c.algorithm_version AS algorithm_version,
       c.summary_version AS summary_version, c.model_name AS model_name,
       collect(e.id) AS member_entity_ids ORDER BY c.id
"""
        try:
            records, _, _ = await self._driver.execute_query(
                query,
                {"user_id": str(user_id)},
                routing_=RoutingControl.READ,
                database_=self._database,
            )
        except Neo4jError as error:
            raise GraphRepositoryError("Neo4j community read failed") from error
        return tuple(
            CommunityRecord(
                community_id=record["id"],
                member_entity_ids=tuple(UUID(value) for value in record["member_entity_ids"]),
                title=record["title"],
                summary=record["summary"],
                source_chunk_ids=tuple(UUID(value) for value in record["source_chunk_ids"]),
                algorithm_version=record["algorithm_version"],
                summary_version=record["summary_version"],
                model_name=record["model_name"],
            )
            for record in records
        )

    async def delete_user_graph(self, *, user_id: UUID) -> None:
        """删除 smoke 或用户删除流程拥有的全部图数据."""
        try:
            await self._driver.execute_query(
                _DELETE_USER_GRAPH,
                {"user_id": str(user_id)},
                database_=self._database,
            )
        except Neo4jError as error:
            raise GraphRepositoryError("Neo4j user graph cleanup failed") from error

    async def delete_document_graph(self, *, user_id: UUID, document_id: UUID) -> None:
        """删除失败重试或文档删除流程派生的全部 chunk 图.

        先读取可信 document 下的 chunk IDs，再在同一事务删除 mention、chunk、
        FACT 和失去全部证据的孤立实体。其他文档仍引用的 canonical entity 会保留。
        """
        user_text = str(user_id)
        document_key = f"{user_text}:{document_id}"
        query_chunk_ids = """
MATCH (:GraphDocument {key: $document_key})-[:HAS_CHUNK]->(c:GraphChunk)
RETURN collect(c.id) AS chunk_ids
"""
        try:
            async with self._driver.session(database=self._database) as session:

                async def write(tx: Any) -> None:
                    result = await tx.run(query_chunk_ids, document_key=document_key)
                    record = await result.single()
                    chunk_ids = list(record["chunk_ids"] if record is not None else ())
                    await (
                        await tx.run(
                            _DELETE_DOCUMENT_GRAPH,
                            document_key=document_key,
                            user_id=user_text,
                            chunk_ids=chunk_ids,
                        )
                    ).consume()
                    await (await tx.run(_DELETE_ORPHAN_ENTITIES, user_id=user_text)).consume()
                    # 任一成员或事实变化都会让社区成员和摘要版本过期。首版选择
                    # 全用户失效，下一次 rebuild 原子生成新快照，避免返回陈旧引用。
                    await (await tx.run(_INVALIDATE_COMMUNITIES, user_id=user_text)).consume()

                await session.execute_write(write)
        except Neo4jError as error:
            raise GraphRepositoryError("Neo4j document graph cleanup failed") from error

    @staticmethod
    def _record_to_path(record: Any) -> GraphPath:
        """把 Neo4j Node/Relationship 投影转换为稳定应用 schema."""
        nodes = record["nodes"]
        relations = record["rels"]
        return GraphPath(
            entity_ids=tuple(UUID(node["id"]) for node in nodes),
            entity_names=tuple(str(node["canonical_name"]) for node in nodes),
            relation_types=tuple(RelationType(relation["relation_type"]) for relation in relations),
            source_chunk_ids=tuple(UUID(relation["chunk_id"]) for relation in relations),
            evidence_texts=tuple(str(relation["evidence_text"]) for relation in relations),
        )


__all__ = ["Neo4jGraphRepository"]
