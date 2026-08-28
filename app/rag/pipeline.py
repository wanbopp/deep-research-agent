"""把 parser、chunker 与持久化 sink 组合成 IndexWorker processor."""

from typing import Protocol

from app.rag.chunker import TextChunk, TokenAwareChunker
from app.rag.parsers import DocumentParseError, ParseRequest, ParserRegistry
from app.services.index_worker import IndexProcessingError, IndexSource


class ChunkSink(Protocol):
    """Lab 18/19 管线写入确定性 chunks 的最小能力."""

    async def replace(self, *, source: IndexSource, chunks: tuple[TextChunk, ...]) -> None:
        """以 document 为边界原子替换 chunks；重复调用必须得到相同结果."""
        ...


class DocumentIndexProcessor:
    """串联解析、分块和持久化，不承担 worker 租约或状态收敛."""

    def __init__(self, *, registry: ParserRegistry, chunker: TokenAwareChunker, sink: ChunkSink) -> None:
        """注入三个可独立测试和替换的管线阶段."""
        self._registry = registry
        self._chunker = chunker
        self._sink = sink

    async def process(self, source: IndexSource) -> None:
        """把 worker 的可信原始文件输入转换为可检索 chunks."""
        try:
            parsed = await self._registry.parse(
                ParseRequest(
                    filename=source.original_filename,
                    content_type=source.content_type,
                    content_sha256=source.content_sha256,
                    content=source.content,
                )
            )
            chunks = self._chunker.split(
                document_id=source.document_id,
                content_sha256=source.content_sha256,
                document=parsed,
            )
            await self._sink.replace(source=source, chunks=chunks)
        except DocumentParseError as error:
            # IndexWorker 只认识稳定 IndexProcessingError；这里是解析领域到任务领域
            # 的翻译点，不携带第三方异常或文档正文。
            raise IndexProcessingError(error.code.value) from None


__all__ = ["ChunkSink", "DocumentIndexProcessor"]
