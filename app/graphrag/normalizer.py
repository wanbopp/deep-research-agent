"""GraphRAG 名称规范化与原文证据定位."""

import re
import unicodedata

from app.graphrag.errors import GraphExtractionRejectedError
from app.graphrag.schemas import SourceSpan

_WHITESPACE = re.compile(r"\s+")


def normalize_entity_name(value: str) -> str:
    """生成仅用于候选匹配的稳定名称，不覆盖原始 mention.

    Unicode NFKC 会统一全角字符和兼容字符；casefold 比 lower 更适合跨语言
    大小写匹配。这里不删除标点或公司后缀，因为过度清洗会把不同实体误合并。
    """
    normalized = unicodedata.normalize("NFKC", value)
    return _WHITESPACE.sub(" ", normalized).strip().casefold()


def locate_source_span(source: str, excerpt: str, *, start_at: int = 0) -> SourceSpan:
    """把模型返回的逐字摘录绑定为可信字符区间.

    Args:
        source: 服务端从可信 DocumentChunk 读取的完整正文。
        excerpt: 模型声称来自正文的短摘录。
        start_at: 查找重复 mention 时使用的起始位置。

    Returns:
        由服务端计算的左闭右开 SourceSpan。

    Raises:
        GraphExtractionRejectedError: 摘录为空或无法在原文中逐字找到。此时不能
            猜测相似位置，否则引用可能指向模型编造的内容。
    """
    candidate = excerpt.strip()
    if not candidate:
        raise GraphExtractionRejectedError("EVIDENCE_BLANK")
    index = source.find(candidate, start_at)
    if index >= 0:
        return SourceSpan(start=index, end=index + len(candidate), text=source[index : index + len(candidate)])

    # structured LLM 有时只改变大小写或把连续换行折叠为空格。它仍然指向同一段
    # 原文，因此允许“大小写不敏感 + 空白等价”的严格匹配；除此之外不做模糊
    # 相似度或语义猜测，避免把模型改写后的句子错误绑定成原文证据。
    tokens = candidate.split()
    if tokens:
        pattern = r"\s+".join(re.escape(token) for token in tokens)
        match = re.search(pattern, source[start_at:], flags=re.IGNORECASE)
        if match is not None:
            start = start_at + match.start()
            end = start_at + match.end()
            return SourceSpan(start=start, end=end, text=source[start:end])
    raise GraphExtractionRejectedError("EVIDENCE_NOT_FOUND")


__all__ = ["locate_source_span", "normalize_entity_name"]
