"""组合 GraphRAG 真实 provider、Neo4j adapter 与检索服务."""

from dataclasses import dataclass
from uuid import UUID

from neo4j import AsyncDriver
from pydantic import SecretStr

from app.core.config import Settings
from app.graphrag.community import (
    CommunityService,
    ConnectedComponentsDetector,
    LLMCommunitySummarizer,
)
from app.graphrag.answering import GlobalGraphAnswerer
from app.graphrag.context import GraphContextAssembler
from app.graphrag.entity_resolution import ConservativeEntityResolver
from app.graphrag.extractor import LLMGraphExtractor
from app.graphrag.global_retriever import GlobalGraphRetriever
from app.graphrag.local_retriever import LLMQueryEntityLinker, LocalGraphRetriever
from app.graphrag.repository import Neo4jGraphRepository
from app.graphrag.schemas import GraphDocument, ResolvedGraphDocument
from app.infrastructure.embeddings import OpenAITextEmbedder
from app.schemas.llm import ModelSpec
from app.services.llm.factory import create_openai_chat_model
from app.services.llm.registry import LLMRegistry
from app.services.llm.service import LLMService

GRAPHRAG_MODEL_ALIAS = "graphrag"


@dataclass(frozen=True, slots=True)
class GraphRAGRuntime:
    """Phase 6 各无请求状态组件的应用级共享快照."""

    repository: Neo4jGraphRepository
    extractor: LLMGraphExtractor
    resolver: ConservativeEntityResolver
    communities: CommunityService
    local: LocalGraphRetriever
    global_retriever: GlobalGraphRetriever
    global_answerer: GlobalGraphAnswerer
    context: GraphContextAssembler

    async def index_chunk(
        self,
        *,
        user_id: UUID,
        document_id: UUID,
        chunk_id: UUID,
        content: str,
        content_sha256: str,
    ) -> tuple[GraphDocument, ResolvedGraphDocument]:
        """执行抽取、消歧和幂等 Neo4j 写入的完整 chunk 调用链.

        返回候选图与已解析图，便于 worker 记录审计和断点观察。写入只接收
        ResolvedGraphDocument，模型原始候选永远不能绕过 resolver 直达 Neo4j。
        """
        candidate = await self.extractor.extract(
            user_id=user_id,
            document_id=document_id,
            chunk_id=chunk_id,
            content=content,
            content_sha256=content_sha256,
        )
        resolved = await self.resolver.resolve(candidate)
        await self.repository.replace_graph_document(resolved)
        return candidate, resolved


def create_graphrag_runtime(
    *,
    config: Settings,
    neo4j_driver: AsyncDriver,
) -> GraphRAGRuntime:
    """从当前环境配置构造惰性的真实 GraphRAG runtime.

    Args:
        config: 当前进程已经加载的配置快照。
        neo4j_driver: lifespan 拥有的共享异步 Neo4j driver。

    Returns:
        可跨请求共享、但不拥有 driver 生命周期的 GraphRAGRuntime。

    Raises:
        RuntimeError: 未配置真实模型凭据。
    """
    if not config.OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is required to create GraphRAG runtime")
    spec = ModelSpec(
        alias=GRAPHRAG_MODEL_ALIAS,
        provider_model=config.DEFAULT_LLM_MODEL,
        api_key=SecretStr(config.OPENAI_API_KEY),
        base_url=config.OPENAI_BASE_URL,
        temperature=0.0,
        max_tokens=min(config.MAX_TOKENS, 2000),
        request_timeout_seconds=max(1.0, config.LLM_TOTAL_TIMEOUT * 0.75),
    )
    registry = LLMRegistry(specs=(spec,), factory=create_openai_chat_model)
    llm_service = LLMService(
        registry,
        max_attempts=config.MAX_LLM_CALL_RETRIES,
        retry_wait_multiplier=0.2,
        total_timeout_seconds=config.LLM_TOTAL_TIMEOUT,
    )
    repository = Neo4jGraphRepository(neo4j_driver)
    extractor = LLMGraphExtractor(
        llm_service,
        aliases=(GRAPHRAG_MODEL_ALIAS,),
        model_name=config.DEFAULT_LLM_MODEL,
    )
    summarizer = LLMCommunitySummarizer(
        llm_service,
        aliases=(GRAPHRAG_MODEL_ALIAS,),
        model_name=config.DEFAULT_LLM_MODEL,
    )
    linker = LLMQueryEntityLinker(llm_service, aliases=(GRAPHRAG_MODEL_ALIAS,))
    embedder = OpenAITextEmbedder.from_settings(config)
    resolver = ConservativeEntityResolver(repository)
    return GraphRAGRuntime(
        repository=repository,
        extractor=extractor,
        resolver=resolver,
        communities=CommunityService(
            repository=repository,
            detector=ConnectedComponentsDetector(),
            summarizer=summarizer,
        ),
        local=LocalGraphRetriever(repository=repository, linker=linker),
        global_retriever=GlobalGraphRetriever(repository=repository, embedder=embedder),
        global_answerer=GlobalGraphAnswerer(llm_service, aliases=(GRAPHRAG_MODEL_ALIAS,)),
        context=GraphContextAssembler(),
    )


__all__ = ["GRAPHRAG_MODEL_ALIAS", "GraphRAGRuntime", "create_graphrag_runtime"]
