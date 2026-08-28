"""DocumentChunk 的应用查询模型与 pgvector 存储适配器."""

from dataclasses import dataclass
from typing import Any, cast
from uuid import UUID

from sqlalchemy import Table, delete
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlmodel import select

from app.models import DOCUMENT_EMBEDDING_DIMENSIONS, Document, DocumentChunk, DocumentStatus
from app.rag.chunker import TextChunk
from app.rag.retrieval import RetrievalChannel, RetrievedChunk
from app.services.embeddings import TextEmbedder
from app.services.index_worker import IndexSource


@dataclass(frozen=True, slots=True)
class VectorSearchHit:
    """owner 过滤后按 cosine similarity 排序的内部检索结果."""

    chunk_id: UUID
    document_id: UUID
    text: str
    similarity: float
    source_locations: tuple[dict[str, Any], ...]


class PostgresDocumentChunkStore:
    """使用真实 Embedding 与 pgvector 保存、检索文档 chunks."""

    def __init__(
        self,
        *,
        session_factory: async_sessionmaker[AsyncSession],
        embedder: TextEmbedder,
        embedding_model: str,
        embedding_version: str,
    ) -> None:
        """验证 schema 维度并保存可跨调用复用的资源工厂."""
        if embedder.dimensions != DOCUMENT_EMBEDDING_DIMENSIONS:
            raise ValueError("Embedding dimensions must match DocumentChunk schema")
        if not embedding_model.strip() or not embedding_version.strip():
            raise ValueError("Embedding model and version must not be empty")
        self._session_factory = session_factory
        self._embedder = embedder
        self._embedding_model = embedding_model.strip()
        self._embedding_version = embedding_version.strip()

    async def replace(self, *, source: IndexSource, chunks: tuple[TextChunk, ...]) -> None:
        """先在事务外请求向量，再原子替换一个文档的全部 chunk.

        Embedding 是慢外部 I/O，不能占住数据库连接。只有向量数量和维度都通过
        adapter 校验后才进入短事务；事务失败时旧 chunks 保持完整，不出现半批新数据。
        """
        if not chunks:
            raise ValueError("chunks must not be empty")
        vectors = await self._embedder.embed_documents(tuple(chunk.text for chunk in chunks))
        if len(vectors) != len(chunks):
            raise RuntimeError("Embedding count does not match chunk count")

        async with self._session_factory() as session:
            async with session.begin():
                owner_check = await session.execute(
                    select(Document.id).where(
                        Document.id == source.document_id,
                        Document.user_id == source.user_id,
                        Document.status != DocumentStatus.DELETING,
                    )
                )
                if owner_check.scalar_one_or_none() is None:
                    raise RuntimeError("Document is unavailable for chunk replacement")
                chunk_table = self._table()
                await session.execute(delete(chunk_table).where(chunk_table.c.document_id == source.document_id))
                session.add_all(
                    [
                        DocumentChunk(
                            id=chunk.id,
                            user_id=source.user_id,
                            document_id=source.document_id,
                            ordinal=chunk.ordinal,
                            content=chunk.text,
                            content_sha256=chunk.content_sha256,
                            token_count=chunk.token_count,
                            source_locations=[item.model_dump(mode="json") for item in chunk.sources],
                            parser_name=chunk.parser_name,
                            parser_version=chunk.parser_version,
                            chunker_version=chunk.chunker_version,
                            embedding_model=self._embedding_model,
                            embedding_version=self._embedding_version,
                            embedding=list(vector),
                        )
                        for chunk, vector in zip(chunks, vectors, strict=True)
                    ]
                )
                await session.flush()

    async def search(
        self,
        *,
        user_id: UUID,
        query: str,
        top_k: int,
        document_ids: frozenset[UUID] | None = None,
    ) -> tuple[RetrievedChunk, ...]:
        """在 SQL 层先执行 owner/document filter，再按 cosine distance 取 top-k."""
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")
        query_vector = await self._embedder.embed_query(query)
        table = self._table()
        distance = table.c.embedding.cosine_distance(list(query_vector))
        statement = (
            select(DocumentChunk, distance.label("distance"))
            .where(
                table.c.user_id == user_id,
                table.c.embedding_model == self._embedding_model,
                table.c.embedding_version == self._embedding_version,
            )
            .order_by(distance)
            .limit(top_k)
        )
        if document_ids is not None:
            if not document_ids:
                return ()
            statement = statement.where(table.c.document_id.in_(document_ids))

        async with self._session_factory() as session:
            rows = (await session.execute(statement)).all()
        return tuple(
            RetrievedChunk(
                chunk_id=row.id,
                document_id=row.document_id,
                text=row.content,
                score=max(-1.0, min(1.0, 1.0 - float(distance_value))),
                rank=rank,
                channel=RetrievalChannel.DENSE,
                source_locations=tuple(row.source_locations),
            )
            for rank, (row, distance_value) in enumerate(rows, start=1)
        )

    async def list_for_search(
        self,
        *,
        user_id: UUID,
        document_ids: frozenset[UUID] | None,
    ) -> tuple[RetrievedChunk, ...]:
        """为 BM25 返回已在 SQL 层执行 owner/document 过滤的语料."""
        table = self._table()
        statement = (
            select(DocumentChunk).where(table.c.user_id == user_id).order_by(table.c.document_id, table.c.ordinal)
        )
        if document_ids is not None:
            if not document_ids:
                return ()
            statement = statement.where(table.c.document_id.in_(document_ids))
        async with self._session_factory() as session:
            rows = (await session.execute(statement)).scalars().all()
        return tuple(
            RetrievedChunk(
                chunk_id=row.id,
                document_id=row.document_id,
                text=row.content,
                score=0.0,
                rank=rank,
                channel=RetrievalChannel.SPARSE,
                source_locations=tuple(row.source_locations),
            )
            for rank, row in enumerate(rows, start=1)
        )

    @staticmethod
    def _table() -> Table:
        """取得运行时 Table，以访问 pgvector 的 cosine_distance comparator."""
        return cast(Table, vars(DocumentChunk)["__table__"])


__all__ = ["PostgresDocumentChunkStore", "VectorSearchHit"]
