"""使用真实 PostgreSQL、pgvector 和 Embedding 验收 Phase 5 检索闭环.

脚本只把固定无敏感语料发送给已配置的真实 embedding provider。最终摘要不输出
凭据、连接串、用户 UUID、正文或向量；临时数据库在成功和失败后都会被删除。
"""

import asyncio
from hashlib import sha256
import json
import os
from pathlib import Path
import selectors
from time import perf_counter
from uuid import uuid4

from alembic import command
from alembic.config import Config
from langchain_core.messages import HumanMessage
import psycopg
from pydantic import SecretStr
from psycopg import sql
from psycopg.conninfo import make_conninfo
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.core.config import settings
from app.infrastructure.database import build_orm_database_url
from app.infrastructure.embeddings import OpenAITextEmbedder
from app.models import Document, DocumentStatus, User
from app.rag.bm25_store import BM25Retriever
from app.rag.chunker import TokenAwareChunker
from app.rag.context import ContextAssembler
from app.rag.hybrid import HybridRetriever
from app.rag.parsers import MarkdownParser, ParserRegistry
from app.rag.pipeline import DocumentIndexProcessor
from app.rag.reranker import TokenOverlapReranker
from app.rag.vector_store import PostgresDocumentChunkStore
from app.schemas.llm import ModelSpec
from app.services.index_worker import IndexSource
from app.services.llm.factory import create_openai_chat_model
from app.services.llm.registry import LLMRegistry
from app.services.llm.service import LLMService

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ADMIN_DATABASE = "postgres"


def _conninfo(database: str) -> str:
    """构造仅存在内存中的 psycopg 连接信息，调用方不得打印返回值."""
    return make_conninfo(
        host=settings.POSTGRES_HOST,
        port=settings.POSTGRES_PORT,
        dbname=database,
        user=settings.POSTGRES_USER,
        password=settings.POSTGRES_PASSWORD,
        connect_timeout=10,
    )


def _create_database(name: str) -> None:
    """通过安全 Identifier 创建本次 smoke 独占的随机数据库."""
    with psycopg.connect(_conninfo(ADMIN_DATABASE), autocommit=True) as connection:
        connection.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(name)))


def _drop_database(name: str) -> None:
    """终止临时连接并只删除随机命名的本次数据库."""
    with psycopg.connect(_conninfo(ADMIN_DATABASE), autocommit=True) as connection:
        connection.execute(
            "SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname = %s AND pid <> pg_backend_pid()",
            (name,),
        )
        connection.execute(sql.SQL("DROP DATABASE IF EXISTS {}").format(sql.Identifier(name)))


def _migrate(database: str) -> None:
    """对临时库执行项目正式 Alembic 链，不用 ORM create_all 绕过迁移."""
    previous = os.environ.get("ALEMBIC_DATABASE_URL")
    os.environ["ALEMBIC_DATABASE_URL"] = (
        build_orm_database_url(settings).set(database=database).render_as_string(hide_password=False)
    )
    try:
        config = Config(str(PROJECT_ROOT / "alembic.ini"))
        config.set_main_option("script_location", str(PROJECT_ROOT / "migrations"))
        command.upgrade(config, "head")
    finally:
        if previous is None:
            os.environ.pop("ALEMBIC_DATABASE_URL", None)
        else:
            os.environ["ALEMBIC_DATABASE_URL"] = previous


async def _seed(
    session_factory: async_sessionmaker[AsyncSession],
    *,
    first_content: bytes,
    second_content: bytes,
) -> tuple[User, Document, User, Document]:
    """创建两个 owner 和相互隔离的文档元数据，不写入任何真实用户资料."""
    async with session_factory() as session:
        async with session.begin():
            first_user = User(email=f"rag-a-{uuid4().hex[:8]}@example.com", password_hash="smoke-only")
            second_user = User(email=f"rag-b-{uuid4().hex[:8]}@example.com", password_hash="smoke-only")
            session.add_all((first_user, second_user))
            await session.flush()
            first_document = Document(
                user_id=first_user.id,
                original_filename="facts.md",
                content_type="text/markdown",
                size_bytes=len(first_content),
                content_sha256=sha256(first_content).hexdigest(),
                storage_key=f"smoke/{uuid4().hex}",
                status=DocumentStatus.INDEXING,
            )
            second_document = Document(
                user_id=second_user.id,
                original_filename="private.md",
                content_type="text/markdown",
                size_bytes=len(second_content),
                content_sha256=sha256(second_content).hexdigest(),
                storage_key=f"smoke/{uuid4().hex}",
                status=DocumentStatus.INDEXING,
            )
            session.add_all((first_document, second_document))
            await session.flush()
    return first_user, first_document, second_user, second_document


async def run(database: str) -> dict[str, object]:
    """执行真实入库、向量检索、Hybrid 融合与引用检查."""
    started = perf_counter()
    engine = None
    summary: dict[str, object] = {"ok": False}
    try:
        engine = create_async_engine(
            build_orm_database_url(settings).set(database=database),
            pool_size=2,
            max_overflow=0,
            pool_pre_ping=True,
        )
        sessions = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False, autoflush=False)
        public_content = (
            "# Device fact\n\nZX-9000 rated power is exactly 1200 watts.\n\n"
            "# Recovery\n\nRestarting the worker resumes expired indexing leases."
        ).encode()
        private_content = b"# Private\n\nZX-9000 private owner note must stay isolated."
        first_user, first_document, second_user, second_document = await _seed(
            sessions,
            first_content=public_content,
            second_content=private_content,
        )

        embedder = OpenAITextEmbedder.from_settings(settings)
        store = PostgresDocumentChunkStore(
            session_factory=sessions,
            embedder=embedder,
            embedding_model=settings.EMBEDDING_MODEL,
            embedding_version="phase5-v1",
        )
        processor = DocumentIndexProcessor(
            registry=ParserRegistry((MarkdownParser(),)),
            chunker=TokenAwareChunker(chunk_size=80, chunk_overlap=12),
            sink=store,
        )

        await processor.process(
            IndexSource(
                document_id=first_document.id,
                user_id=first_user.id,
                original_filename="facts.md",
                content_type="text/markdown",
                content_sha256=first_document.content_sha256,
                content=public_content,
            )
        )
        await processor.process(
            IndexSource(
                document_id=second_document.id,
                user_id=second_user.id,
                original_filename="private.md",
                content_type="text/markdown",
                content_sha256=second_document.content_sha256,
                content=private_content,
            )
        )

        hybrid = HybridRetriever(
            dense=store,
            sparse=BM25Retriever(store),
            reranker=TokenOverlapReranker(),
        )
        results = await hybrid.search(
            user_id=first_user.id,
            query="ZX-9000 rated power",
            candidate_k=8,
            final_k=4,
        )
        context = ContextAssembler(max_tokens=180).assemble(results)
        owner_isolated = bool(results) and all(item.document_id == first_document.id for item in results)
        exact_fact_retrieved = bool(results) and "1200 watts" in results[0].text
        citations_valid = bool(context.citations) and all(
            any(str(result.chunk_id) == citation.chunk_id for result in results) for citation in context.citations
        )
        citation_id = context.citations[0].citation_id if context.citations else ""
        llm = LLMService(
            LLMRegistry(
                (
                    ModelSpec(
                        alias="primary",
                        provider_model=settings.DEFAULT_LLM_MODEL,
                        api_key=SecretStr(settings.OPENAI_API_KEY),
                        base_url=settings.OPENAI_BASE_URL,
                        temperature=0,
                        max_tokens=80,
                    ),
                ),
                create_openai_chat_model,
            ),
            max_attempts=1,
            total_timeout_seconds=90,
        )
        answer_message = await llm.call(
            (
                HumanMessage(
                    content=(
                        "Use only the evidence below. Answer the rated power of ZX-9000 in one short sentence, "
                        "and include the evidence citation exactly as written.\n\n" + context.text
                    )
                ),
            ),
            aliases=("primary",),
        )
        answer_text = answer_message.content if isinstance(answer_message.content, str) else ""
        answer_has_fact = "1200" in answer_text
        answer_has_citation = bool(citation_id) and f"[{citation_id}]" in answer_text
        summary.update(
            {
                "ok": (
                    owner_isolated
                    and exact_fact_retrieved
                    and citations_valid
                    and answer_has_fact
                    and answer_has_citation
                ),
                "embedding_model": settings.EMBEDDING_MODEL,
                "answer_model": settings.DEFAULT_LLM_MODEL,
                "owner_filter_applied": owner_isolated,
                "exact_fact_retrieved": exact_fact_retrieved,
                "citation_count": len(context.citations),
                "citations_valid": citations_valid,
                "context_within_budget": context.token_count <= 180,
                "answer_has_fact": answer_has_fact,
                "answer_has_citation": answer_has_citation,
                "result_count": len(results),
            }
        )
    finally:
        if engine is not None:
            await engine.dispose()
        summary["elapsed_ms"] = round((perf_counter() - started) * 1000, 2)
    return summary


if __name__ == "__main__":
    database_name = f"deep_research_phase5_{uuid4().hex[:10]}"
    result: dict[str, object] = {"ok": False}
    cleanup_succeeded = False
    try:
        _create_database(database_name)
        _migrate(database_name)
        # psycopg 的 Windows 异步实现需要 SelectorEventLoop；默认 Proactor loop
        # 适合 subprocess/pipe，却不能驱动 psycopg socket。显式 loop_factory 只影响
        # 本 smoke 进程，不修改应用或其他测试的全局 event-loop policy。
        result = asyncio.run(
            run(database_name),
            loop_factory=lambda: asyncio.SelectorEventLoop(selectors.SelectSelector()),
        )
    finally:
        try:
            _drop_database(database_name)
            cleanup_succeeded = True
        except Exception:
            cleanup_succeeded = False
    result["cleanup_ok"] = cleanup_succeeded
    result["ok"] = bool(result.get("ok") and cleanup_succeeded)
    print(json.dumps(result, ensure_ascii=False))
    raise SystemExit(0 if result["ok"] else 1)
