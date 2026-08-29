"""把 Phase 5/6 能力装配成共享 DeepResearch 运行图."""

from langgraph.checkpoint.base import BaseCheckpointSaver
from pydantic import SecretStr

from app.agents.research.graph import ResearchGraph, build_research_graph
from app.agents.research.planner import ResearchPlanner
from app.agents.research.retrieval import (
    GlobalGraphEvidenceRetriever,
    HybridEvidenceRetriever,
    LocalGraphEvidenceRetriever,
    ParallelResearchRetriever,
    WebEvidenceRetriever,
)
from app.agents.research.router import ResearchRouter
from app.agents.research.validator import ResearchValidator
from app.agents.research.writer import ResearchWriter
from app.core.config import Settings
from app.graphrag.runtime import GraphRAGRuntime
from app.rag.hybrid import HybridRetriever
from app.schemas.llm import ModelSpec
from app.schemas.research import RetrievalStrategy
from app.services.llm.factory import create_openai_chat_model
from app.services.llm.registry import LLMRegistry
from app.services.llm.service import LLMService

RESEARCH_MODEL_ALIAS = "research"


def create_research_runtime(
    *,
    config: Settings,
    hybrid_retriever: HybridRetriever,
    graphrag_runtime: GraphRAGRuntime,
    checkpointer: BaseCheckpointSaver[str] | None = None,
) -> ResearchGraph:
    """创建一次、跨请求复用的研究图.

    Args:
        config: 当前环境的模型和超时配置快照。
        hybrid_retriever: Phase 5 的用户文档文字检索能力。
        graphrag_runtime: Phase 6 的局部关系和全局主题检索能力。
        checkpointer: 可选持久状态保存器；生产传 PostgreSQL saver。

    Returns:
        不保存当前用户数据的已编译 ResearchGraph。user_id/research_id 在每次
        ainvoke 时通过 ResearchRuntimeContext 注入。
    """
    if not config.OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is required to create research runtime")

    registry = LLMRegistry(
        specs=(
            ModelSpec(
                alias=RESEARCH_MODEL_ALIAS,
                provider_model=config.DEFAULT_LLM_MODEL,
                api_key=SecretStr(config.OPENAI_API_KEY),
                base_url=config.OPENAI_BASE_URL,
                temperature=0.0,
                max_tokens=config.MAX_TOKENS,
                request_timeout_seconds=max(config.LLM_TOTAL_TIMEOUT, 30.0),
            ),
        ),
        factory=create_openai_chat_model,
    )
    llm_service = LLMService(
        registry,
        max_attempts=config.MAX_LLM_CALL_RETRIES,
        retry_wait_multiplier=0.2,
        total_timeout_seconds=config.LLM_TOTAL_TIMEOUT,
    )
    aliases = (RESEARCH_MODEL_ALIAS,)
    retriever = ParallelResearchRetriever(
        router=ResearchRouter(),
        retrievers={
            RetrievalStrategy.HYBRID: HybridEvidenceRetriever(hybrid_retriever),
            RetrievalStrategy.GRAPH_LOCAL: LocalGraphEvidenceRetriever(graphrag_runtime.local),
            RetrievalStrategy.GRAPH_GLOBAL: GlobalGraphEvidenceRetriever(graphrag_runtime.global_retriever),
            RetrievalStrategy.WEB: WebEvidenceRetriever(),
        },
    )
    return build_research_graph(
        planner=ResearchPlanner(llm_service, aliases=aliases),
        retriever=retriever,
        validator=ResearchValidator(llm_service, aliases=aliases),
        writer=ResearchWriter(llm_service, aliases=aliases),
        checkpointer=checkpointer,
    )


__all__ = ["RESEARCH_MODEL_ALIAS", "create_research_runtime"]
