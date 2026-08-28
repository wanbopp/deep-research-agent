"""解析阶段共享的确定性 Unicode 与空白规范化规则."""

import re
import unicodedata


def normalize_text(text: str) -> str:
    """规范换行与 Unicode，同时保留有语义的段落和代码缩进.

    NFC 会组合等价 Unicode 序列，但不像 NFKC 那样擅自改变全角字符或兼容字形。
    这里只删除不安全控制字符、行尾空白和过量空行，不把所有空白压成一个空格。
    """
    normalized = unicodedata.normalize("NFC", text).replace("\r\n", "\n").replace("\r", "\n")
    normalized = "".join(
        character for character in normalized if character in {"\n", "\t"} or unicodedata.category(character) != "Cc"
    )
    normalized = "\n".join(line.rstrip() for line in normalized.split("\n"))
    return re.sub(r"\n{3,}", "\n\n", normalized).strip()


__all__ = ["normalize_text"]
