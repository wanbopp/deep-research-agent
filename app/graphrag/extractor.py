"""使用真实 structured LLM 抽取候选图，并绑定可信 chunk 来源."""

from collections.abc import Sequence
from hashlib import sha256
from typing import Protocol
from uuid import UUID

from langchain_core.messages import HumanMessage, SystemMessage

from app.agents.prompts.loader import PromptArtifact, get_prompt_spec, load_prompt_artifact, render_prompt_input
from app.graphrag.normalizer import locate_source_span, normalize_entity_name
from app.graphrag.errors import GraphExtractionRejectedError
from app.core.logging import logger
from app.graphrag.schemas import (
    EntityCandidate,
    EntityMention,
    GraphDocument,
    GraphExtractionPayload,
    RelationCandidate,
)
from app.services.llm.service import LLMService
from app.services.llm.errors import StructuredOutputError

DEFAULT_TAXONOMY_VERSION = "graphrag-taxonomy-v1"
DEFAULT_EXTRACTION_VERSION = "graphrag-extraction-v1"
_EXTRACTION_PROMPT_SPEC = get_prompt_spec("graphrag_extract")
DEFAULT_PROMPT_VERSION = f"{_EXTRACTION_PROMPT_SPEC.name}:{_EXTRACTION_PROMPT_SPEC.version}"


class GraphExtractor(Protocol):
    """把一个可信 chunk 转换为来源绑定 GraphDocument 的应用边界."""

    async def extract(
        self,
        *,
        user_id: UUID,
        document_id: UUID,
        chunk_id: UUID,
        content: str,
        content_sha256: str,
    ) -> GraphDocument:
        """抽取候选图；可信身份参数必须来自服务端调用链."""
        ...


class LLMGraphExtractor:
    """通过 LLMService 的真实 structured output 提取 chunk 级候选图."""

    def __init__(
        self,
        llm_service: LLMService,
        *,
        aliases: Sequence[str],
        model_name: str,
        taxonomy_version: str = DEFAULT_TAXONOMY_VERSION,
        extraction_version: str = DEFAULT_EXTRACTION_VERSION,
        prompt_version: str = DEFAULT_PROMPT_VERSION,
    ) -> None:
        """保存无请求状态的模型边界和可审计版本信息."""
        if isinstance(aliases, str) or not aliases:
            raise ValueError("aliases must contain at least one model alias")
        self._llm_service = llm_service
        self._aliases = tuple(aliases)
        self._model_name = model_name
        self._taxonomy_version = taxonomy_version
        self._extraction_version = extraction_version
        self._prompt_version = prompt_version

    async def extract(
        self,
        *,
        user_id: UUID,
        document_id: UUID,
        chunk_id: UUID,
        content: str,
        content_sha256: str,
    ) -> GraphDocument:
        """执行真实抽取，并在服务端绑定身份、来源和 span.

        模型只看到正文，不看到可信 user/document/chunk ID。调用结束后，本方法
        用原文逐字定位 evidence；任何无法定位的摘录都会让整次结果被拒绝，确保
        后续 Neo4j 中不存在无法回查的边。
        """
        if not content.strip():
            raise ValueError("content must not be blank")
        actual_sha256 = sha256(content.encode("utf-8")).hexdigest()
        if actual_sha256 != content_sha256:
            raise ValueError("content_sha256 does not match content")

        extraction_prompt = load_prompt_artifact("graphrag_extract")
        repair_prompt = load_prompt_artifact("graphrag_extract_repair")
        prompts = (extraction_prompt, repair_prompt, repair_prompt)
        for repair_count, prompt in enumerate(prompts):
            try:
                payload = await self._request_payload(content=content, prompt=prompt)
                return self._bind_payload(
                    payload=payload,
                    user_id=user_id,
                    document_id=document_id,
                    chunk_id=chunk_id,
                    content=content,
                    content_sha256=content_sha256,
                    repair_count=repair_count,
                    allow_partial=repair_count > 0,
                )
            except (GraphExtractionRejectedError, StructuredOutputError) as error:
                if repair_count < len(prompts) - 1:
                    continue
                rejection_code = (
                    "STRUCTURED_OUTPUT_REJECTED" if isinstance(error, StructuredOutputError) else error.reason_code
                )
                logger.warning(
                    "graphrag_chunk_rejected",
                    rejection_code=rejection_code,
                    repair_count=repair_count,
                )
                # rejected GraphDocument 仍记录可信来源和版本，但没有任何候选。
                # Repository 会保存 GraphChunk.rejection_code，方便后续定向重放；
                # 坏模型输出本身不会被持久化或进入实体图。
                return GraphDocument(
                    user_id=user_id,
                    document_id=document_id,
                    source_chunk_id=chunk_id,
                    source_content_sha256=content_sha256,
                    taxonomy_version=self._taxonomy_version,
                    extraction_version=self._extraction_version,
                    prompt_version=self._prompt_version,
                    model_name=self._model_name,
                    repair_count=repair_count,
                    rejection_code=rejection_code,
                    entities=(),
                    relations=(),
                )
        raise RuntimeError("Graph extraction retry loop exited without a result")

    async def _request_payload(self, *, content: str, prompt: PromptArtifact) -> GraphExtractionPayload:
        """执行一次真实 structured 调用；修复仍使用同一 provider 弹性边界."""
        return await self._llm_service.call_structured(
            (
                SystemMessage(content=prompt.content),
                HumanMessage(content=render_prompt_input(prompt.name, content=content)),
            ),
            response_model=GraphExtractionPayload,
            aliases=self._aliases,
            overrides={"temperature": 0.0},
            prompt=prompt,
        )

    def _bind_payload(
        self,
        *,
        payload: GraphExtractionPayload,
        user_id: UUID,
        document_id: UUID,
        chunk_id: UUID,
        content: str,
        content_sha256: str,
        repair_count: int,
        allow_partial: bool,
    ) -> GraphDocument:
        """把一次模型候选绑定到可信原文；本方法不执行网络 I/O."""
        entities: list[EntityCandidate] = []
        dropped_entities = 0
        for entity in payload.entities:
            mentions: list[EntityMention] = []
            for index, mention in enumerate(entity.mentions, start=1):
                try:
                    span = locate_source_span(content, mention)
                except GraphExtractionRejectedError:
                    if not allow_partial:
                        raise
                    continue
                mentions.append(
                    EntityMention(
                        mention_id=f"{entity.local_id}:M{index}",
                        text=span.text,
                        span=span,
                    )
                )
            if not mentions:
                dropped_entities += 1
                continue
            entities.append(
                EntityCandidate(
                    local_id=entity.local_id,
                    canonical_name=entity.canonical_name,
                    normalized_name=normalize_entity_name(entity.canonical_name),
                    entity_type=entity.entity_type,
                    mentions=tuple(mentions),
                    aliases=tuple(dict.fromkeys(entity.aliases)),
                    identity_hint=entity.identity_hint,
                    confidence=entity.confidence,
                )
            )

        retained_entity_ids = {entity.local_id for entity in entities}
        relations: list[RelationCandidate] = []
        dropped_relations = 0
        for relation in payload.relations:
            if (
                relation.source_entity_id not in retained_entity_ids
                or relation.target_entity_id not in retained_entity_ids
            ):
                dropped_relations += 1
                continue
            try:
                evidence = locate_source_span(content, relation.evidence_text)
            except GraphExtractionRejectedError:
                if not allow_partial:
                    raise
                dropped_relations += 1
                continue
            relations.append(
                RelationCandidate(
                    local_id=relation.local_id,
                    source_entity_id=relation.source_entity_id,
                    target_entity_id=relation.target_entity_id,
                    relation_type=relation.relation_type,
                    evidence=evidence,
                    confidence=relation.confidence,
                )
            )
        if allow_partial and (dropped_entities or dropped_relations):
            logger.info(
                "graphrag_extraction_candidates_dropped",
                dropped_entity_count=dropped_entities,
                dropped_relation_count=dropped_relations,
            )
        return GraphDocument(
            user_id=user_id,
            document_id=document_id,
            source_chunk_id=chunk_id,
            source_content_sha256=content_sha256,
            taxonomy_version=self._taxonomy_version,
            extraction_version=self._extraction_version,
            prompt_version=self._prompt_version,
            model_name=self._model_name,
            repair_count=repair_count,
            entities=tuple(entities),
            relations=tuple(relations),
        )


__all__ = [
    "DEFAULT_EXTRACTION_VERSION",
    "DEFAULT_PROMPT_VERSION",
    "DEFAULT_TAXONOMY_VERSION",
    "GraphExtractor",
    "LLMGraphExtractor",
]
