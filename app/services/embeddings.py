"""文本向量化的应用层边界.

本模块只定义“把文本转换为固定维度向量”的能力，不依赖 OpenAI、LangChain
或 HTTP 客户端。长期记忆存储只依赖该协议，因此以后替换 provider 时不需要
修改 MemoryStore 的业务规则。
"""

from collections.abc import Sequence
from typing import Protocol

from app.services.memory import MemoryUnavailableError


class EmbeddingUnavailableError(MemoryUnavailableError):
    """真实 Embedding provider 无法安全完成请求.

    该异常属于 ``MemoryUnavailableError`` 的细分类型。上层既可以统一按长期记忆
    后端故障降级，也可以在监控中单独识别 provider 故障。固定错误文本不会泄露
    API Key、Base URL、用户文本或 provider 原始响应。
    """

    def __init__(self) -> None:
        """创建可安全跨越应用层边界的异常."""
        RuntimeError.__init__(self, "Embedding provider is unavailable")


class TextEmbedder(Protocol):
    """定义长期记忆所需的文本向量化能力.

    Protocol 是结构化接口：实现类无须继承它，只要提供同名属性和异步方法，
    Pyright 就能验证其兼容性。这样应用层不会反向依赖某个 provider SDK。
    """

    @property
    def dimensions(self) -> int:
        """返回每个向量固定包含的浮点数数量."""
        ...

    async def embed_query(self, text: str) -> tuple[float, ...]:
        """把一条检索文本转换为查询向量.

        Args:
            text: 已校验且去除首尾空白的查询文本。

        Returns:
            长度严格等于 ``dimensions`` 的不可变浮点元组。

        Raises:
            ValueError: 文本为空。
            EmbeddingUnavailableError: provider 请求失败或返回非法向量。
        """
        ...

    async def embed_documents(
        self,
        texts: Sequence[str],
    ) -> tuple[tuple[float, ...], ...]:
        """批量把待存储文本转换为文档向量.

        Args:
            texts: 保持输入顺序的非空文本序列。

        Returns:
            与输入一一对应、顺序不变的不可变向量元组。

        Raises:
            ValueError: 序列为空或包含空文本。
            EmbeddingUnavailableError: provider 请求失败或返回非法向量。
        """
        ...


__all__ = ["EmbeddingUnavailableError", "TextEmbedder"]
