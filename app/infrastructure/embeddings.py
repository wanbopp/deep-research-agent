"""基于 OpenAI-compatible provider 的真实文本向量适配器."""

import math
from collections.abc import Sequence

from langchain_openai import OpenAIEmbeddings
from pydantic import SecretStr

from app.core.config import Settings
from app.core.logging import logger
from app.models.memory import MEMORY_EMBEDDING_DIMENSIONS
from app.services.embeddings import EmbeddingUnavailableError


class OpenAITextEmbedder:
    """使用真实 provider 实现应用层 ``TextEmbedder`` 协议.

    该类只负责 provider 通信和返回值校验。它不拥有数据库事务，也不决定用户
    namespace。文本和向量属于用户派生数据，因此日志只记录操作名和异常类型。
    """

    def __init__(
        self,
        *,
        model: str,
        api_key: SecretStr,
        base_url: str | None,
        dimensions: int,
        timeout_seconds: float,
        max_retries: int,
    ) -> None:
        """构造惰性的真实 Embedding 客户端.

        Args:
            model: provider 公开的 embedding 模型名称。
            api_key: 使用 SecretStr 包装的 provider 凭据。
            base_url: OpenAI-compatible API 根地址；None 使用 SDK 默认地址。
            dimensions: provider 返回且数据库 ``VECTOR(n)`` 接收的固定维度。
            timeout_seconds: 单次 provider 请求的超时秒数。
            max_retries: SDK 在网络或可重试状态码下执行的最大重试次数。

        Raises:
            ValueError: 配置为空、越界，或维度与数据库 schema 不一致。

        Notes:
            构造客户端不会发送网络请求。真实 I/O 发生在 ``embed_query`` 或
            ``embed_documents`` 中，因此该对象可以安全地由应用 lifespan 复用。
        """
        normalized_model = model.strip()
        if not normalized_model:
            raise ValueError("Embedding model must not be empty")
        if not api_key.get_secret_value():
            raise ValueError("Embedding API key must not be empty")
        if dimensions != MEMORY_EMBEDDING_DIMENSIONS:
            raise ValueError("Embedding dimensions must match the PostgreSQL vector schema")
        if timeout_seconds <= 0:
            raise ValueError("Embedding timeout must be greater than 0")
        if max_retries < 0:
            raise ValueError("Embedding retries must not be negative")

        self._dimensions = dimensions
        self._client = OpenAIEmbeddings(
            model=normalized_model,
            api_key=api_key,
            base_url=base_url,
            dimensions=dimensions,
            timeout=timeout_seconds,
            max_retries=max_retries,
            # 关闭 LangChain 的本地长度检查后，文本只会发给真实 provider；
            # 也避免不同 tokenizer 版本在客户端擅自切分并改变输入语义。
            check_embedding_ctx_length=False,
        )

    @classmethod
    def from_settings(cls, config: Settings) -> "OpenAITextEmbedder":
        """从应用配置构造适配器，但不立即请求 provider.

        Args:
            config: 已加载当前环境变量的 Settings。

        Returns:
            可被多个请求并发复用的无状态 Embedding 适配器。
        """
        return cls(
            model=config.EMBEDDING_MODEL,
            api_key=SecretStr(config.OPENAI_API_KEY),
            base_url=config.OPENAI_BASE_URL,
            dimensions=config.EMBEDDING_DIMENSIONS,
            timeout_seconds=config.EMBEDDING_REQUEST_TIMEOUT,
            max_retries=config.EMBEDDING_MAX_RETRIES,
        )

    @property
    def dimensions(self) -> int:
        """返回与 provider 和 PostgreSQL schema 共同固定的向量维度."""
        return self._dimensions

    async def embed_query(self, text: str) -> tuple[float, ...]:
        """把一条真实查询文本转换为经过校验的向量."""
        normalized_text = self._normalize_text(text)
        try:
            vector = await self._client.aembed_query(normalized_text)
        except Exception as exc:
            # 不记录 exc 文本：provider 错误可能回显 URL、请求正文或其他敏感字段。
            logger.warning(
                "embedding_request_failed",
                operation="query",
                error_type=type(exc).__name__,
            )
            raise EmbeddingUnavailableError() from None
        return self._validate_vector(vector)

    async def embed_documents(
        self,
        texts: Sequence[str],
    ) -> tuple[tuple[float, ...], ...]:
        """批量向真实 provider 请求文档向量，并保持输入顺序."""
        if not texts:
            raise ValueError("Embedding documents must not be empty")
        normalized_texts = [self._normalize_text(text) for text in texts]
        try:
            vectors = await self._client.aembed_documents(normalized_texts)
        except Exception as exc:
            logger.warning(
                "embedding_request_failed",
                operation="documents",
                error_type=type(exc).__name__,
            )
            raise EmbeddingUnavailableError() from None

        if len(vectors) != len(normalized_texts):
            raise EmbeddingUnavailableError()
        return tuple(self._validate_vector(vector) for vector in vectors)

    @staticmethod
    def _normalize_text(text: str) -> str:
        """拒绝空文本，同时不在日志或异常中回显原文."""
        normalized_text = text.strip()
        if not normalized_text:
            raise ValueError("Embedding text must not be empty")
        return normalized_text

    def _validate_vector(self, vector: Sequence[float]) -> tuple[float, ...]:
        """验证 provider 返回维度和有限值，阻止坏数据进入 pgvector."""
        try:
            if len(vector) != self._dimensions:
                raise EmbeddingUnavailableError()
            normalized_vector = tuple(float(value) for value in vector)
        except EmbeddingUnavailableError:
            raise
        except (TypeError, ValueError, OverflowError):
            # provider 响应也属于不可信边界。无法转换的元素不能以普通 Python
            # 异常穿过应用层，更不能写入数据库后才等待 pgvector 报错。
            raise EmbeddingUnavailableError() from None
        if not all(math.isfinite(value) for value in normalized_vector):
            raise EmbeddingUnavailableError()
        return normalized_vector


__all__ = ["OpenAITextEmbedder"]
