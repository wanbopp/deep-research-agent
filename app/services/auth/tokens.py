"""JWT access token 的签发、验证与安全错误边界.

本模块不依赖 FastAPI route 或数据库。它只回答两个确定性问题：如何把可信用户 ID
签成短期 token，以及如何把外部 token 验证成结构可靠的 claims。后续 dependency
还必须根据 ``claims.sub`` 查询 User；TokenService 本身不会把账户是否存在作为假设。
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Final
from uuid import UUID, uuid4

from jose import jwt
from jose.exceptions import ExpiredSignatureError, JWTError
from pydantic import SecretStr, ValidationError

from app.schemas.auth import AccessTokenClaims, TokenResponse

if TYPE_CHECKING:
    from app.core.config import Settings


SUPPORTED_HMAC_ALGORITHMS: Final = frozenset({"HS256", "HS384", "HS512"})
KNOWN_INSECURE_SECRETS: Final = frozenset(
    {
        "",
        "change-me",
        "change-me-in-production",
        "secret",
    }
)
MINIMUM_SECRET_BYTES: Final = 32
MAXIMUM_ACCESS_TOKEN_TTL: Final = timedelta(days=1)
DEFAULT_CLOCK_SKEW_SECONDS: Final = 5

Clock = Callable[[], datetime]


def _utc_now() -> datetime:
    """返回带 UTC 时区的当前时间，作为默认可注入时钟."""
    return datetime.now(UTC)


class TokenServiceError(RuntimeError):
    """TokenService 对上层暴露的稳定、安全错误基类."""


class TokenConfigurationError(TokenServiceError):
    """JWT 算法、secret 或有效期配置不安全."""

    def __init__(self, message: str = "JWT configuration is invalid") -> None:
        """使用不包含 secret 的固定配置错误文本."""
        super().__init__(message)


class TokenCreationError(TokenServiceError):
    """access token 无法签发."""

    def __init__(self) -> None:
        """隐藏底层签名实现细节和 claims 内容."""
        super().__init__("Access token could not be created")


class InvalidAccessTokenError(TokenServiceError):
    """token 的签名、结构、用途或时间声明无效."""

    def __init__(self) -> None:
        """构造不回显原始 token 的稳定错误."""
        super().__init__("Access token is invalid")


class AccessTokenExpiredError(InvalidAccessTokenError):
    """签名有效但已经超过 ``exp`` 的内部错误分类.

    该分类便于服务端指标区分过期与其他无效 token。未来 HTTP dependency 应把
    两者映射成相同的 401 公共响应，避免向外部暴露不必要的认证诊断信息。
    """

    def __init__(self) -> None:
        """构造不包含 claims 或 token 的固定过期错误."""
        TokenServiceError.__init__(self, "Access token has expired")


class TokenService:
    """使用对称 HMAC secret 签发并验证短期 access token.

    安全顺序非常重要：

    1. ``jwt.decode`` 先限制算法并验证签名和标准时间 claims；
    2. ``AccessTokenClaims`` 再验证 UUID、用途和时间窗口的数据形状；
    3. 最后检查 ``iat`` 没有明显来自未来；
    4. 通过后的 ``sub`` 仍需在 9E 查询数据库，才能变成 AuthenticatedUser。

    任何一步失败都不会返回部分 payload。Agent、tool 或 route 因此没有机会读取
    一个“虽然能解码，但尚未建立信任”的 user_id。
    """

    def __init__(
        self,
        *,
        secret_key: SecretStr,
        algorithm: str = "HS256",
        access_token_ttl: timedelta = timedelta(minutes=30),
        clock_skew_seconds: int = DEFAULT_CLOCK_SKEW_SECONDS,
        clock: Clock = _utc_now,
    ) -> None:
        """验证安全配置，并保存签发和验签所需的不可变参数.

        Args:
            secret_key: 仅由服务端持有的 HMAC secret。不得来自请求、Prompt、Agent
                state 或客户端配置；UTF-8 编码后至少需要 32 字节。
            algorithm: 签名算法名称，目前只接受 ``HS256/HS384/HS512``。签发和
                验签必须使用同一配置，不能信任 token 自带的 ``alg``。
            access_token_ttl: access token 的生存时间，用来计算 ``exp = iat + ttl``。
                refresh/session token 以后使用独立策略，不能复用这个参数无限续期。
            clock_skew_seconds: 验证时间 claims 时允许的最大时钟偏差，范围 0 到 60。
            clock: 返回带时区 ``datetime`` 的时钟函数。默认返回 UTC 当前时间。

        Raises:
            TokenConfigurationError: secret 太短或是已知弱值、算法不在 allowlist、
                ttl 越界，或者时钟偏差不在允许范围内。
        """
        secret_value = secret_key.get_secret_value()
        if secret_value in KNOWN_INSECURE_SECRETS or len(secret_value.encode("utf-8")) < MINIMUM_SECRET_BYTES:
            raise TokenConfigurationError
        if algorithm not in SUPPORTED_HMAC_ALGORITHMS:
            # 当前实现使用共享 secret，因此只允许明确列出的 HMAC 算法。调用 decode
            # 时还会把同一个算法放进 allowlist，防止 token header 自行选择算法。
            raise TokenConfigurationError
        if access_token_ttl <= timedelta(0) or access_token_ttl > MAXIMUM_ACCESS_TOKEN_TTL:
            raise TokenConfigurationError
        if clock_skew_seconds < 0 or clock_skew_seconds > 60:
            raise TokenConfigurationError

        # 保存 SecretStr 而非原始字符串，降低调试器对象展示和意外 repr 泄密风险。
        self._secret_key = secret_key
        # create_access_token 用它签名；decode_access_token 用它限制可接受算法。
        self._algorithm = algorithm
        # 每次签发时基于当前 iat 计算 exp，不在构造阶段预先计算绝对过期时间。
        self._access_token_ttl = access_token_ttl
        # 同时用于 python-jose 的 leeway 和未来 iat 检查，保持两处容差一致。
        self._clock_skew_seconds = clock_skew_seconds
        # 保存函数而不是保存某个 datetime，保证每次签发和验证都读取新的时间。
        self._clock = clock

    @classmethod
    def from_settings(cls, config: Settings) -> "TokenService":
        """从应用配置构造 service，同时复用同一组安全校验.

        Args:
            config: 应用集中配置。这里只读取 JWT secret、算法和分钟有效期；数据库、
                LLM 等其他配置不会进入 TokenService。

        Returns:
            已通过 ``__init__`` 全部安全校验的 TokenService。

        Raises:
            TokenConfigurationError: Settings 中任一 JWT 配置不安全或越界。
        """
        return cls(
            secret_key=SecretStr(config.JWT_SECRET_KEY),
            algorithm=config.JWT_ALGORITHM,
            access_token_ttl=timedelta(
                minutes=config.JWT_ACCESS_TOKEN_EXPIRE_MINUTES,
            ),
        )

    def create_access_token(self, *, user_id: UUID) -> TokenResponse:
        """为可信用户 ID 签发一个短期 bearer access token.

        调用者必须已经通过注册或登录流程取得可信 ``user_id``。本方法不接受邮箱、
        密码或任意附加字典，避免把敏感信息和未审查 claims 塞进 JWT payload。
        """
        issued_at = self._now()
        expires_at = issued_at + self._access_token_ttl

        # JWT 标准时间字段使用 NumericDate（Unix 秒）。UUID 也显式转换为字符串，
        # 让编码后的 payload 与 python-jose 的标准 claim 校验保持兼容。
        payload = {
            "sub": str(user_id),
            "iat": int(issued_at.timestamp()),
            "exp": int(expires_at.timestamp()),
            "jti": str(uuid4()),
            "token_type": "access",
        }

        try:
            access_token = jwt.encode(
                payload,
                self._secret_value,
                algorithm=self._algorithm,
            )
        except Exception as exc:
            raise TokenCreationError from exc

        return TokenResponse(
            access_token=access_token,
            expires_at=expires_at,
        )

    def decode_access_token(self, token: str) -> AccessTokenClaims:
        """验签并把外部 token 转换为严格的 access claims.

        ``algorithms=[self._algorithm]`` 是关键防线：服务端配置决定可接受算法，
        不能相信 JWT header 自报的 ``alg``。``require_*`` 则保证关键 claims 缺失时
        立即失败，而不是产生一个字段不完整的“半可信身份”。
        """
        if not token:
            raise InvalidAccessTokenError

        try:
            payload = jwt.decode(
                token,
                self._secret_value,
                algorithms=[self._algorithm],
                options={
                    "require_sub": True,
                    "require_iat": True,
                    "require_exp": True,
                    "require_jti": True,
                    "leeway": self._clock_skew_seconds,
                },
            )
            claims = AccessTokenClaims.model_validate(payload)
        except ExpiredSignatureError as exc:
            raise AccessTokenExpiredError from exc
        except (JWTError, ValidationError, TypeError, ValueError) as exc:
            raise InvalidAccessTokenError from exc

        # python-jose 会验证 iat 的类型，但不会拒绝“来自未来”的合法整数。
        # 允许少量时钟偏差，超过后仍按无效 token 处理。
        latest_allowed_issued_at = self._now() + timedelta(
            seconds=self._clock_skew_seconds,
        )
        if claims.iat > latest_allowed_issued_at:
            raise InvalidAccessTokenError

        return claims

    def _now(self) -> datetime:
        """读取并规范化可注入时钟，拒绝无时区时间."""
        value = self._clock()
        if value.tzinfo is None or value.utcoffset() is None:
            raise TokenConfigurationError("JWT clock must return an aware datetime")
        return value.astimezone(UTC)

    @property
    def _secret_value(self) -> str:
        """仅在 python-jose 调用边界短暂解包 secret."""
        return self._secret_key.get_secret_value()
