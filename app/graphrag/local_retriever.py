"""Local GraphRAG：查询实体链接与受限邻域扩展."""

from collections.abc import Sequence
from typing import Protocol
from uuid import UUID

from langchain_core.messages import HumanMessage, SystemMessage

from app.graphrag.normalizer import normalize_entity_name
from app.graphrag.schemas import GraphPath, LocalGraphResult, QueryEntityPayload, StoredEntity
from app.services.llm.service import LLMService


class QueryEntityLinker(Protocol):
    """从用户问题中提取实体名称的可替换边界."""

    async def extract_names(self, query: str) -> tuple[str, ...]:
        """返回待链接名称；无法识别实体时返回空元组."""
        ...


class LLMQueryEntityLinker:
    """使用真实 structured LLM 提取查询实体，不决定用户归属或图 ID."""

    def __init__(self, llm_service: LLMService, *, aliases: Sequence[str]) -> None:
        """保存共享 LLMService 和不可变 alias 顺序."""
        if isinstance(aliases, str) or not aliases:
            raise ValueError("aliases must contain at least one model alias")
        self._llm_service = llm_service
        self._aliases = tuple(aliases)

    async def extract_names(self, query: str) -> tuple[str, ...]:
        """提取问题中逐字出现或明确指代的实体名称."""
        if not query.strip():
            raise ValueError("query must not be blank")
        payload = await self._llm_service.call_structured(
            (
                SystemMessage(
                    content=(
                        "Extract up to 10 entity names explicitly mentioned in the query. "
                        "Do not answer the query and do not invent entities."
                    )
                ),
                HumanMessage(content=query),
            ),
            response_model=QueryEntityPayload,
            aliases=self._aliases,
            overrides={"temperature": 0.0},
        )
        return tuple(dict.fromkeys(payload.names))


class LocalGraphRepository(Protocol):
    """LocalGraphRetriever 所需的用户隔离 Neo4j 查询能力."""

    async def find_entities(self, *, user_id: UUID, normalized_names: Sequence[str]) -> tuple[StoredEntity, ...]:
        """在一个用户图中链接名称候选."""
        ...

    async def local_paths(
        self,
        *,
        user_id: UUID,
        entity_ids: Sequence[UUID],
        depth: int,
        limit: int,
    ) -> tuple[GraphPath, ...]:
        """返回固定深度和数量预算内的图路径."""
        ...


class LocalGraphRetriever:
    """执行 query entity extraction、owner-scoped linking 与邻域检索."""

    def __init__(self, *, repository: LocalGraphRepository, linker: QueryEntityLinker) -> None:
        """注入存储和模型边界；对象自身不保存当前用户或查询."""
        self._repository = repository
        self._linker = linker

    async def search(self, *, user_id: UUID, query: str, depth: int = 2, limit: int = 20) -> LocalGraphResult:
        """返回可解释路径；未链接实体时显式要求回退 Hybrid RAG."""
        names = await self._linker.extract_names(query)
        entities = await self._repository.find_entities(
            user_id=user_id,
            normalized_names=tuple(normalize_entity_name(name) for name in names),
        )
        entity_ids = tuple(entity.entity_id for entity in entities)
        if not entity_ids:
            return LocalGraphResult(linked_entity_ids=(), paths=(), fallback_required=True)
        paths = await self._repository.local_paths(
            user_id=user_id,
            entity_ids=entity_ids,
            depth=depth,
            limit=limit,
        )
        return LocalGraphResult(
            linked_entity_ids=entity_ids,
            paths=paths,
            fallback_required=not paths,
        )


__all__ = ["LLMQueryEntityLinker", "LocalGraphRetriever", "QueryEntityLinker"]
