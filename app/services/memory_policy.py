"""长期记忆写入前的敏感内容策略."""

import re
from enum import StrEnum
from typing import Protocol


class MemoryRejectionCode(StrEnum):
    """允许调用方稳定处理的记忆拒绝原因."""

    SENSITIVE_CREDENTIAL = "sensitive_credential"


class MemoryRejectedError(ValueError):
    """候选记忆因安全策略被拒绝.

    异常只公开固定代码和固定文案，不携带命中的正文、正则或疑似凭据。
    """

    def __init__(self, code: MemoryRejectionCode) -> None:
        """保存稳定原因代码，不回显候选内容."""
        self.code = code
        super().__init__("Memory candidate was rejected by content policy")


class MemoryContentPolicy(Protocol):
    """定义 MemoryService 写入前必须执行的本地内容检查."""

    def ensure_allowed(self, content: str) -> None:
        """验证候选内容是否允许发送给 provider 并持久化.

        Args:
            content: 已由 ``MemoryCreate`` 校验并去除边界空白的记忆正文。

        Raises:
            MemoryRejectedError: 内容高度疑似包含真实 credential。
        """
        ...


class CredentialMemoryPolicy:
    """使用高精度本地规则拒绝常见 credential 形状.

    这是一道 defense-in-depth 边界，不是完整 DLP 系统。规则刻意要求“字段名加
    长值”或已知 token 形状，避免仅因用户说“不要泄露 API Key”就误拒绝普通偏好。
    """

    _PATTERNS = (
        re.compile(
            r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:api[_ -]?key|access[_ -]?token|secret[_ -]?key|password|passwd)"
            r"\s*[:=]\s*[\"']?[A-Za-z0-9_./+=-]{8,}",
            re.IGNORECASE,
        ),
        re.compile(
            r"\bBearer\s+[A-Za-z0-9._~+/-]{16,}=*",
            re.IGNORECASE,
        ),
        re.compile(r"\bsk-[A-Za-z0-9_-]{16,}\b"),
        re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    )

    def ensure_allowed(self, content: str) -> None:
        """在任何网络或数据库 I/O 前拒绝疑似 credential.

        Args:
            content: 待检查的单条记忆正文。

        Raises:
            MemoryRejectedError: 任一高精度规则命中。
        """
        if any(pattern.search(content) is not None for pattern in self._PATTERNS):
            raise MemoryRejectedError(MemoryRejectionCode.SENSITIVE_CREDENTIAL)


__all__ = [
    "CredentialMemoryPolicy",
    "MemoryContentPolicy",
    "MemoryRejectedError",
    "MemoryRejectionCode",
]
