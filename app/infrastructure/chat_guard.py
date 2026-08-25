"""Redis-backed execution guard for Chat Agent threads."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from hashlib import sha256

from redis.asyncio import Redis
from redis.exceptions import LockError, RedisError

from app.services.chat_guard import (
    ChatExecutionGuardUnavailableError,
    ChatThreadBusyError,
)

# Redis 只看到固定前缀和 SHA-256 摘要，不保存原始 user UUID 或公开 thread ID。
# 前缀用于运维识别用途，摘要用于保持相同内部 key 稳定映射到同一把分布式锁。
CHAT_EXECUTION_LOCK_PREFIX = "deep-research:chat-execution:"


class RedisChatExecutionGuard:
    """使用 redis-py Lua Lock 协调跨 worker 的 Chat Graph 执行.

    一个 ``RedisChatExecutionGuard`` 可以被应用级 ``ChatService`` 单例长期复用；
    每次 ``hold()`` 都会创建独立 Lock 对象和 owner token，因此并发任务不会共享
    可变 token。真正的协调状态存储在 Redis，而不是当前 Python 进程内。
    """

    def __init__(
        self,
        redis_client: Redis,
        *,
        lease_seconds: float,
    ) -> None:
        """保存共享 Redis client，并验证锁的最大生存时间.

        Args:
            redis_client: 由 FastAPI lifespan 创建并在 shutdown 关闭的异步客户端。
                guard 只借用该客户端，绝不能自行关闭它。
            lease_seconds: Redis 锁自动过期时间。它必须长于 ChatService 的 Graph
                总超时，为正常 ``finally`` 释放保留余量；进程崩溃时又能防止锁永久
                残留。该参数使用秒，允许浮点数。

        Raises:
            ValueError: lease_seconds 小于或等于 0。
        """
        if lease_seconds <= 0:
            raise ValueError("lease_seconds must be greater than 0")

        self._redis_client = redis_client
        self._lease_seconds = lease_seconds

    @staticmethod
    def _build_lock_name(internal_thread_id: str) -> str:
        """把内部 thread ID 转换为不含原文的稳定 Redis key.

        Args:
            internal_thread_id: ChatService 使用可信 user_id 与公开 thread ID 构造的
                checkpoint 身份。它不会来自模型输出或工具参数。

        Returns:
            固定业务前缀加 SHA-256 十六进制摘要。相同输入得到相同 key，不同输入
            几乎必然得到不同 key，因此锁粒度与 checkpoint 粒度保持一致。

        Raises:
            ValueError: internal_thread_id 为空或只包含空白字符。

        Notes:
            摘要的目的主要是避免 Redis key 和诊断工具直接暴露内部身份，不是把
            thread ID 当作密码加密。真正的身份信任仍来自 API 认证链。
        """
        if not internal_thread_id.strip():
            raise ValueError("internal_thread_id must not be empty")

        digest = sha256(internal_thread_id.encode("utf-8")).hexdigest()
        return f"{CHAT_EXECUTION_LOCK_PREFIX}{digest}"

    @asynccontextmanager
    async def hold(self, internal_thread_id: str) -> AsyncIterator[None]:
        """以 fail-fast 方式持有一个内部 thread 的分布式执行权.

        Args:
            internal_thread_id: 与 LangGraph ``configurable.thread_id`` 完全相同的
                服务端内部标识。方法只使用其摘要作为 Redis key。

        Yields:
            没有业务值；进入 ``async with`` 代码块本身就代表已经取得执行权。

        Raises:
            ChatThreadBusyError: 相同内部 thread 的锁已被另一个请求持有。
            ChatExecutionGuardUnavailableError: Redis 获取或释放失败，无法继续保证
                同 thread 线性执行。该错误采用 fail-closed，不会降级为无锁执行。
            ValueError: internal_thread_id 为空。

        Notes:
            redis-py Lock 内部使用 ``SET key token NX PX`` 获取锁，并使用 Lua 比较
            owner token 后删除。即使旧 lease 已过期并被新请求取得，旧请求也不能
            删除新 owner 的锁。``blocking=False`` 保证 busy 请求立即返回，不排队
            等待，也不会进入后面的 Graph 或产生模型费用。
        """
        lock_name = self._build_lock_name(internal_thread_id)

        # 每次调用创建独立 Lock，避免不同 asyncio task 共享 owner token。
        # thread_local=False 在这里是明确表达 token 属于这个 Lock 实例，而不是
        # 操作系统线程；实际并发单位是 asyncio task 和跨进程 Redis 客户端。
        lock = self._redis_client.lock(
            lock_name,
            timeout=self._lease_seconds,
            blocking=False,
            thread_local=False,
        )

        try:
            acquired = await lock.acquire()  # 尝试获取分布式锁
        except RedisError as error:
            # 固定应用错误不包含 Redis host、密码、原始 key 或驱动错误文本。
            raise ChatExecutionGuardUnavailableError("Chat execution guard is unavailable") from error

        if not acquired:
            raise ChatThreadBusyError("Chat thread is already being processed")

        try:
            yield  # 暂停 hold() 住执行async with 内部代码块
        finally:
            # 正常完成、interrupt return、timeout、provider/tool 异常以及单次任务取消
            # 都会经过 finally。若进程直接崩溃无法运行 finally，Redis TTL 是最后的
            # 自动恢复边界，因此 lease_seconds 不能为 None。
            try:
                await lock.release()
            except (LockError, RedisError) as error:
                # 释放失败意味着我们无法再证明互斥状态，必须显式失败而不是记录后
                # 假装成功。锁仍有 TTL，最终不会永久阻塞该 thread。
                raise ChatExecutionGuardUnavailableError("Chat execution guard release failed") from error
