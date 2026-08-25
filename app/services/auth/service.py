"""注册与登录认证用例的应用服务.

AuthService 位于 HTTP route 与底层安全/持久化组件之间。它不理解状态码，也不执行
原始 SQL；它只负责按照安全顺序组合 PasswordHasher、UserRepository、事务和
TokenService，使每条认证路径只有一个明确的职责所有者。
"""

from sqlalchemy.ext.asyncio import AsyncSession

from app.repositories import RepositoryConflictError, UserRepository
from app.schemas.auth import LoginRequest, RegisterRequest, TokenResponse
from app.services.auth.passwords import PasswordHasher
from app.services.auth.tokens import TokenService


# 未知邮箱仍要执行一次真实 Argon2id verify，避免它明显快于“邮箱存在但密码错误”。
# 这是公开的 dummy credential，不对应任何用户；即使知道其明文也无法登录，因为
# user is None 的分支最终始终抛 InvalidCredentialsError。
DUMMY_PASSWORD_HASH = (
    "$argon2id$v=19$m=65536,t=3,p=4$B5vt9Rmx0F/O87jOZga3Bg$O3Ue2EcNtM+1cgbhKB1M11Ipn5yicVwujGcXlu16c20"
)


class AuthServiceError(RuntimeError):
    """注册和登录可预期业务错误的基类."""


class EmailAlreadyRegisteredError(AuthServiceError):
    """数据库唯一约束表明规范化邮箱已经注册."""

    def __init__(self) -> None:
        """使用固定文本，禁止把用户提交的邮箱带入异常或日志."""
        super().__init__("Email is already registered")


class InvalidCredentialsError(AuthServiceError):
    """未知邮箱、错误密码或损坏 credential 的统一登录错误."""

    def __init__(self) -> None:
        """不区分具体失败原因，避免形成账户枚举接口."""
        super().__init__("Email or password is incorrect")


class AuthService:
    """编排注册和登录，不依赖 FastAPI 或 Agent runtime.

    组件职责：

    - ``AsyncSession``：一次认证请求的数据库工作单元；
    - ``UserRepository``：执行 User 查询和写入，但不 commit；
    - ``PasswordHasher``：唯一允许短暂解包 SecretStr 的密码学边界；
    - ``TokenService``：只为已经认证成功的可信 user_id 签发 access token。

    route 不能绕过本类直接创建 User 或签 token。否则事务、密码校验和错误统一规则
    会散落到 HTTP 层，后续 CLI、后台任务或 Agent 工具也无法安全复用认证用例。
    """

    def __init__(
        self,
        *,
        session: AsyncSession,
        password_hasher: PasswordHasher,
        token_service: TokenService,
    ) -> None:
        """保存当前请求的 Session，并组合可复用的无状态安全服务.

        Args:
            session: 请求级 AsyncSession。AuthService 拥有写事务，不能跨请求或并发
                Agent 节点共享。
            password_hasher: 真实 Argon2id hash/verify 服务。
            token_service: 真实 JWT 签发服务；只有认证成功路径会调用它。
        """
        self._session = session
        self._users = UserRepository(session)
        self._password_hasher = password_hasher
        self._token_service = token_service

    async def register(self, request: RegisterRequest) -> TokenResponse:
        """创建用户并在事务提交成功后签发 access token.

        哈希发生在 ``session.begin()`` 之前，因为 Argon2id 是昂贵计算，不应在等待
        期间占用数据库事务。INSERT 则必须放入 begin；唯一约束负责处理两个相同
        邮箱并发注册的最终竞态，不能只依赖一次“先查询是否存在”。

        Raises:
            EmailAlreadyRegisteredError: PostgreSQL 唯一约束拒绝重复邮箱。
        """
        password_hash = await self._password_hasher.hash(request.password)

        try:
            async with self._session.begin():
                user = await self._users.create(
                    email=str(request.email),
                    password_hash=password_hash,
                )
        except RepositoryConflictError as exc:
            # 异常离开 begin 后事务已经 rollback，再转换成不依赖数据库/HTTP 的
            # 业务错误。异常文本不包含 email、密码或 password_hash。
            raise EmailAlreadyRegisteredError from exc

        # 只有退出 begin 并成功 commit 后才签 token。若写入失败，调用链永远不会
        # 产生一个指向不存在用户的有效 JWT。
        return self._token_service.create_access_token(user_id=user.id)

    async def login(self, request: LoginRequest) -> TokenResponse:
        """使用统一失败行为验证凭据，成功后签发新的 access token.

        查询不到用户时不能立即返回。我们改用预计算 dummy hash 执行同类 Argon2id
        verify，再统一抛错，从而减少邮箱是否存在造成的明显响应时间差异。

        Raises:
            InvalidCredentialsError: 邮箱未知、密码错误或存储的哈希无法验证。
        """
        user = await self._users.get_by_email(str(request.email))
        password_hash = user.password_hash if user is not None else DUMMY_PASSWORD_HASH
        password_matches = await self._password_hasher.verify(
            request.password,
            password_hash,
        )

        # 先完成 verify 再统一判断。即使请求密码碰巧匹配 dummy hash，user 为 None
        # 仍然失败；调用方也无法从异常类型或文本区分两种情况。
        if user is None or not password_matches:
            raise InvalidCredentialsError

        return self._token_service.create_access_token(user_id=user.id)
