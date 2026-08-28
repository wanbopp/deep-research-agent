"""GraphRAG 领域错误."""


class GraphRAGError(Exception):
    """GraphRAG 可预期错误的基类."""


class GraphExtractionRejectedError(GraphRAGError):
    """模型候选无法安全绑定到原文时拒绝整次抽取."""

    def __init__(self, reason_code: str) -> None:
        """保存不含原文和模型输出的稳定拒绝原因."""
        self.reason_code = reason_code
        super().__init__("Graph extraction candidate was rejected")


class GraphRepositoryError(GraphRAGError):
    """图仓储无法完成安全读写."""


class GraphCapabilityUnavailableError(GraphRAGError):
    """当前环境缺少请求的图算法能力."""


__all__ = [
    "GraphCapabilityUnavailableError",
    "GraphExtractionRejectedError",
    "GraphRAGError",
    "GraphRepositoryError",
]
