"""异步安全的密码哈希边界.

密码哈希既不属于 ORM，也不属于 HTTP route。这个模块把 ``pwdlib`` 的同步
Argon2id 操作包装成很小的 application-service API，让后续注册和登录流程只需要
表达两个业务动作：生成可持久化哈希，以及校验候选密码。

Argon2id 是 Argon2 密码哈希算法的混合变体（2015 年 Password Hashing Competition
冠军）。它结合 Argon2i 的抗 GPU 侧信道特性和 Argon2d 的抗 tradeoff 特性，通过
三个可调参数控制计算成本：

- **时间成本 (t)**：迭代次数，越高越慢；
- **内存成本 (m)**：每次哈希消耗的内存，越高越难被并行暴力破解；
- **并行度 (p)**：可使用的线程数。

每次哈希自动生成随机 salt，输出的编码串形如
``$argon2id$v=19$m=65536,t=3,p=4$<salt>$<hash>``，把算法版本、参数和 salt
全部编码在内，因此验证时只需这一个字符串，不需要额外存储 salt 或参数。
"""

from anyio import to_thread
from pydantic import SecretStr
from pwdlib import PasswordHash


class PasswordHashingError(RuntimeError):
    """密码哈希无法完成时对上层暴露的安全错误.

    底层库异常可能包含实现细节。这里使用固定文本，并通过异常链保留供开发者
    调试的原始原因；route 层以后只应映射这个稳定异常，不能把 ``str(exc)``
    直接发送给客户端。
    """

    def __init__(self) -> None:
        """构造不包含明文密码或哈希值的固定错误信息."""
        super().__init__("Password hashing failed")


class PasswordHasher:
    """使用推荐 Argon2id 参数生成和验证密码哈希.

    心智模型：

    1. ``SecretStr`` 降低密码被 repr 或普通日志意外打印的风险，但它不是加密。
    2. 只有本类会调用 ``get_secret_value()``，明文不会继续传给 ORM/Repository。
    3. ``PasswordHash.recommended()`` 负责选择并编码算法参数、随机 salt 和结果。
    4. Argon2id 消耗 CPU 与内存，必须在线程中执行，避免阻塞 FastAPI 事件循环。

    ``PasswordHash`` 可以注入，便于以后集中升级参数或执行 hash 迁移；默认路径
    始终使用 pwdlib 当前推荐配置，而不是在业务代码中散落算法常量。
    """

    def __init__(self, password_hash: PasswordHash | None = None) -> None:
        """保存无状态、可复用的 pwdlib 配置对象."""
        self._password_hash = password_hash or PasswordHash.recommended()

    async def hash(self, password: SecretStr) -> str:
        """在线程中把明文密码转换为可持久化的 Argon2id 编码串.

        返回字符串包含算法标识、参数、随机 salt 和摘要，因此验证时无需另存
        salt。它可以进入数据库，但仍属于敏感 credential，不应写普通日志。

        Raises:
            ValueError: 密码为空。schema 已经阻止公开请求中的空密码，这个检查
                继续保护绕过 HTTP 直接调用 service 的内部代码。
            PasswordHashingError: 底层哈希操作失败，且不会向上泄漏实现细节。
        """
        plain_password = password.get_secret_value()
        if not plain_password:
            raise ValueError("password must not be empty")

        try:
            # run_sync 只把 CPU/内存密集的同步函数搬到工作线程；当前协程会挂起，
            # 事件循环因此仍能处理其他请求。结果返回后再沿正常 async 调用链继续。
            return await to_thread.run_sync(self._password_hash.hash, plain_password)
        except Exception as exc:
            raise PasswordHashingError from exc

    async def verify(self, password: SecretStr, password_hash: str) -> bool:
        """在线程中验证候选密码，任何无效 credential 都统一返回 ``False``.

        调用者只需要知道凭据是否匹配，不需要区分“密码错误”和“数据库中的哈希
        损坏”。统一返回 false 可以让未来登录接口使用同一种安全失败响应，避免
        将底层算法信息暴露给外部调用者。
        """
        plain_password = password.get_secret_value()
        if not plain_password or not password_hash:
            return False

        try:
            return await to_thread.run_sync(
                self._password_hash.verify,
                plain_password,
                password_hash,
            )
        except Exception:
            # malformed hash 不是客户端需要理解的异常类型。后续 AuthService 可在
            # 固定事件名下记录“credential invalid”，但不得记录密码或哈希原文。
            return False
