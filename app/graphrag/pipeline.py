"""把 Phase 6 GraphRAG 接入现有文档索引 worker."""

import asyncio

from app.core.logging import logger
from app.graphrag.errors import GraphExtractionRejectedError
from app.graphrag.runtime import GraphRAGRuntime
from app.rag.chunker import TextChunk
from app.services.index_worker import IndexProcessingError, IndexSource


class GraphIndexSink:
    """在有限并发下抽取 chunks，再按顺序消歧并写入 Neo4j."""

    def __init__(self, runtime: GraphRAGRuntime, *, extraction_concurrency: int = 3) -> None:
        """保存共享 runtime 与真实模型调用并发预算."""
        if extraction_concurrency <= 0:
            raise ValueError("extraction_concurrency must be greater than zero")
        self._runtime = runtime
        self._extraction_concurrency = extraction_concurrency

    async def replace(self, *, source: IndexSource, chunks: tuple[TextChunk, ...]) -> None:
        """替换一份文档的图派生数据，失败时清理半成品.

        抽取阶段可以并发，因为各 chunk 没有共享状态；消歧和写入必须按 ordinal
        顺序执行，让后续 chunk 看见先前建立的 canonical entity。任何失败都转换
        为稳定任务错误码，日志和数据库不会保存文档正文或 provider body。
        """
        semaphore = asyncio.Semaphore(self._extraction_concurrency)

        async def extract(chunk: TextChunk):
            """只在预算内调用真实模型，不执行共享图写入."""
            async with semaphore:
                return await self._runtime.extractor.extract(
                    user_id=source.user_id,
                    document_id=source.document_id,
                    chunk_id=chunk.id,
                    content=chunk.text,
                    content_sha256=chunk.content_sha256,
                )

        try:
            candidates = await asyncio.gather(*(extract(chunk) for chunk in chunks))
            for candidate in candidates:
                resolved = await self._runtime.resolver.resolve(candidate)
                await self._runtime.repository.replace_graph_document(resolved)
        except Exception as error:
            # 只记录阶段和异常类型，不记录异常文本、chunk 正文或 provider body。
            # 这足以区分 structured output、证据绑定与 Neo4j 写入故障。
            logger.warning(
                "graphrag_document_index_failed",
                error_type=type(error).__name__,
                reason_code=(error.reason_code if isinstance(error, GraphExtractionRejectedError) else None),
            )
            # Neo4j 不是 PostgreSQL 事务的一部分，无法获得真正的跨数据库原子性。
            # 补偿删除把可观察状态恢复到“该文档尚未建图”，随后原 IndexJob 可重试。
            try:
                await self._runtime.repository.delete_document_graph(
                    user_id=source.user_id,
                    document_id=source.document_id,
                )
            except Exception:
                # 清理失败不能掩盖主失败；稳定错误码要求运维先修复 Neo4j，再重试。
                raise IndexProcessingError("GRAPH_CLEANUP_FAILED") from None
            raise IndexProcessingError("GRAPH_EXTRACTION_FAILED") from None


__all__ = ["GraphIndexSink"]
