"""Global GraphRAG：真实 embedding 社区匹配."""

import math
from typing import Protocol
from uuid import UUID

from app.graphrag.schemas import CommunityRecord, GlobalGraphResult
from app.services.embeddings import TextEmbedder


class GlobalCommunityRepository(Protocol):
    """GlobalGraphRetriever 所需的社区读取能力."""

    async def list_communities(self, *, user_id: UUID) -> tuple[CommunityRecord, ...]:
        """只返回可信用户拥有的社区."""
        ...


class GlobalGraphRetriever:
    """使用真实 embedding 在小规模社区摘要上执行语义匹配."""

    def __init__(self, *, repository: GlobalCommunityRepository, embedder: TextEmbedder) -> None:
        """注入用户隔离仓储和真实向量 provider 边界."""
        self._repository = repository
        self._embedder = embedder

    async def search(self, *, user_id: UUID, query: str, top_k: int = 5) -> GlobalGraphResult:
        """对社区摘要排序；没有社区时返回正常空结果."""
        if not query.strip():
            raise ValueError("query must not be blank")
        if top_k <= 0 or top_k > 20:
            raise ValueError("top_k must be between 1 and 20")
        communities = await self._repository.list_communities(user_id=user_id)
        if not communities:
            return GlobalGraphResult(communities=(), scores=())
        query_vector = await self._embedder.embed_query(query)
        vectors = await self._embedder.embed_documents(tuple(f"{item.title}\n{item.summary}" for item in communities))
        ranked = sorted(
            zip(communities, vectors, strict=True),
            key=lambda item: self._cosine(query_vector, item[1]),
            reverse=True,
        )[:top_k]
        return GlobalGraphResult(
            communities=tuple(item[0] for item in ranked),
            scores=tuple(self._cosine(query_vector, item[1]) for item in ranked),
        )

    @staticmethod
    def _cosine(left: tuple[float, ...], right: tuple[float, ...]) -> float:
        """计算有限向量余弦相似度，零向量安全返回 0."""
        if len(left) != len(right):
            raise ValueError("embedding dimensions must match")
        numerator = sum(a * b for a, b in zip(left, right, strict=True))
        denominator = math.sqrt(sum(value * value for value in left)) * math.sqrt(
            sum(value * value for value in right)
        )
        return numerator / denominator if denominator else 0.0


__all__ = ["GlobalGraphRetriever"]
