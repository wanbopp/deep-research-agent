"""应用级缓存契约与缓存键构造.

定义不依赖 redis-py 的 Cache Protocol。
协议只暴露 get()、set()、delete()

本模块刻意不包含任何 Redis 导入。应用服务只依赖下方的 :class:`Cache`
协议，而基础设施适配器决定数据存储在内存、Redis 还是其他后端。

调用链
业务 service
  -> Cache.get(key)
      -> InMemoryCache
      或 RedisCache
  -> 命中：返回字符串
  -> 未命中：返回 None
  -> Cache.set(key, value, ttl_seconds)
  -> Cache.delete(key)

Redis 失败时
redis-py exception
  -> RedisCache
  -> CacheUnavailableError
  -> 业务 service 决定回源数据库还是终止请求



"""

from hashlib import sha256
import re
from typing import Protocol
import unicodedata


_CACHE_KEY_PREFIX = "deep-research:cache"
_SAFE_KEY_SEGMENT = re.compile(r"^[a-z][a-z0-9_-]{0,63}$")  # 用来验证 Redis key 中的 namespace 和 version 段是否安全。


class CacheUnavailableError(RuntimeError):
    """缓存后端无法安全完成操作.

    错误消息刻意保持稳定：适配器不得在此异常中包含 Redis 地址、密码、原始 key、
    缓存值或底层异常信息。调用方可自行决定是否 fail-open 并回退到数据源读取。
    """

    def __init__(self) -> None:
        """创建可安全跨越应用边界的异常."""
        super().__init__("Cache backend is unavailable")


class Cache(Protocol):
    """描述应用服务所需的缓存能力.

    这是一个结构化协议而非具体实现。只要类提供了签名相同的方法即视为满足
    协议，无需继承 ``Cache``。值始终保持为字符串，序列化由使用缓存的业务
    服务自行负责。
    """

    async def get(self, key: str) -> str | None:
        """读取一个已序列化的值.

        Args:
            key: 由 :func:`build_cache_key` 生成的安全版本化键。

        Returns:
            缓存命中时返回序列化值，未命中时返回 ``None``。

        Raises:
            CacheUnavailableError: 后端无法判定命中或未命中。
                调用服务自行决定是否允许此操作 fail-open。
        """
        ...

    async def set(self, key: str, value: str, *, ttl_seconds: int) -> None:
        """存储一个带显式生存时间的序列化值.

        Args:
            key: 由 :func:`build_cache_key` 生成的安全版本化键。
            value: 已序列化的应用数据。适配器必须将其视为不透明内容。
            ttl_seconds: 自动过期前的正整数秒数。

        Raises:
            ValueError: ``ttl_seconds`` 不是正整数。
            CacheUnavailableError: 后端无法存储该值。
        """
        ...

    async def delete(self, key: str) -> None:
        """幂等移除一个缓存值.

        Args:
            key: 由 :func:`build_cache_key` 生成的安全版本化键。

        Raises:
            CacheUnavailableError: 后端无法完成失效操作。

        删除不存在的键仍视为成功。这使应用服务无需依赖后端特定的
        整数删除计数。
        """
        ...


def build_cache_key(
    *,
    namespace: str,
    version: str,
    identity: str,
) -> str:
    """构造稳定、版本化的键，不暴露原始身份.

    Args:
        namespace: 稳定的缓存用途标识，如 ``chat_session_list``。必须是应用代码
            控制的短小写标识符。
        version: 序列化或行为版本号，如 ``v1``。递增版本号可以在不扫描 Redis
            删除的情况下隔离不兼容的旧条目。
        identity: 区分条目的所有者/查询身份。哈希前会移除边界空白并标准化 Unicode。

    Returns:
        形如 ``deep-research:cache:v1:<namespace>:<sha256-digest>`` 的键。

    Raises:
        ValueError: 某个段不安全，或 identity 标准化后为空。

    Security:
        SHA-256 确保用户 ID、邮箱、Prompt、Token 和查询文本不会出现在可观察的
        Redis key 中。它不做授权：调用方在缓存未命中时仍必须执行 owner-scoped
        数据库查询。
    """
    _validate_key_segment("namespace", namespace)
    _validate_key_segment("version", version)

    # NFC 使规范等价的 Unicode 文本产生同一缓存身份；
    # strip 移除意外的边界空白。领域特定的规范化（如邮箱转小写）
    # 仍由调用服务负责。
    normalized_identity = unicodedata.normalize("NFC", identity).strip()
    if not normalized_identity:
        raise ValueError("identity must not be empty")

    digest = sha256(normalized_identity.encode("utf-8")).hexdigest()
    return f"{_CACHE_KEY_PREFIX}:{version}:{namespace}:{digest}"


def _validate_key_segment(name: str, value: str) -> None:
    """拒绝为空、动态或包含分隔符的 Redis key 段.

    Args:
        name: 稳定验证消息中使用的可读参数名。
        value: 应用代码提供的候选 namespace 或 version。

    Raises:
        ValueError: ``value`` 不是安全的小写 key 段。
    """
    if not _SAFE_KEY_SEGMENT.fullmatch(value):
        raise ValueError(
            f"{name} must start with a lowercase letter and contain only "
            "lowercase letters, digits, underscores, or hyphens"
        )


__all__ = ["Cache", "CacheUnavailableError", "build_cache_key"]
