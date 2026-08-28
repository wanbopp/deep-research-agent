"""从应用配置组合可运行的 RAG 索引与检索对象图."""

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.config import Settings
from app.infrastructure.embeddings import OpenAITextEmbedder
from app.rag.bm25_store import BM25Retriever
from app.rag.chunker import TokenAwareChunker
from app.rag.hybrid import HybridRetriever
from app.rag.parsers import DocxParser, MarkdownParser, ParserRegistry, PdfParser, PlainTextParser
from app.rag.pipeline import DocumentIndexProcessor
from app.rag.reranker import TokenOverlapReranker
from app.rag.vector_store import PostgresDocumentChunkStore
from app.services.file_storage import FileStorage
from app.services.index_worker import IndexWorker


def create_rag_runtime(
    *,
    config: Settings,
    session_factory: async_sessionmaker[AsyncSession],
    storage: FileStorage,
    worker_id: str,
) -> tuple[IndexWorker, HybridRetriever]:
    """创建共享 chunk store、索引 worker 与 Hybrid Retriever.

    Args:
        config: 当前环境配置；所有 parser/chunk/embedding 版本从这里集中确定。
        session_factory: lifespan 或 worker 进程拥有的 ORM Session 工厂。
        storage: 原始文档 FileStorage adapter。
        worker_id: 当前索引 worker 实例的非空内部标识。

    Returns:
        可轮询 pending job 的 worker，以及供 API/Agent 后续注入的 retriever。

    Notes:
        工厂只构造惰性对象，不发送模型或数据库请求。Embedding I/O 发生在 worker
        处理文档或 retriever 执行 dense search 时。
    """
    embedder = OpenAITextEmbedder.from_settings(config)
    chunk_store = PostgresDocumentChunkStore(
        session_factory=session_factory,
        embedder=embedder,
        embedding_model=config.EMBEDDING_MODEL,
        embedding_version=config.RAG_EMBEDDING_VERSION,
    )
    processor = DocumentIndexProcessor(
        registry=ParserRegistry((PlainTextParser(), MarkdownParser(), PdfParser(), DocxParser())),
        chunker=TokenAwareChunker(
            chunk_size=config.RAG_CHUNK_SIZE,
            chunk_overlap=config.RAG_CHUNK_OVERLAP,
        ),
        sink=chunk_store,
    )
    worker = IndexWorker(
        session_factory=session_factory,
        storage=storage,
        processor=processor,
        worker_id=worker_id,
        lease_seconds=config.KNOWLEDGE_INDEX_LEASE_SECONDS,
    )
    retriever = HybridRetriever(
        dense=chunk_store,
        sparse=BM25Retriever(chunk_store),
        reranker=TokenOverlapReranker(),
    )
    return worker, retriever


__all__ = ["create_rag_runtime"]
