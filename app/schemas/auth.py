"""认证 API 与可信用户上下文的数据模型.

本模块只定义数据形状和输入边界，不执行密码哈希、JWT 签发或数据库访问。
这些确定性安全操作会分别由后续 PasswordHasher、TokenService 和 AuthService 负责。


RegisterRequest：注册密码要求 12-128 个字符。
LoginRequest：允许验证历史短密码，只要求非空。
TokenResponse：token 可进入 HTTP 响应，但不会出现在对象 repr。
AuthenticatedUser：只包含可信的 user_id/email

"""

from typing import Literal
from uuid import UUID

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    EmailStr,
    Field,
    SecretStr,
    field_validator,
)


MIN_REGISTRATION_PASSWORD_LENGTH = 12
MAX_PASSWORD_LENGTH = 128


def _normalize_email(value: object) -> object:
    """规范化字符串邮箱，同时保留非字符串值交给 Pydantic 拒绝.

    邮箱可以安全地去除首尾空白并统一大小写，方便唯一约束稳定工作。不能使用
    ``str(value)`` 强制转换，否则整数等非法输入可能被悄悄接受。
    """
    return value.strip().casefold() if isinstance(value, str) else value


class _StrictAuthModel(BaseModel):
    """为认证模型提供一致的不可变、禁止额外字段配置."""

    # extra="forbid" 会立即暴露客户端拼错的字段；frozen=True 防止已经验证的
    # 认证数据在 route/service 之间传递时被意外修改。
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        # 认证输入校验失败时，异常文本也不能回显密码。API exception handler
        # 仍需继续只提取 loc/msg/type，因为 errors() 结构可能保留原始 input。
        hide_input_in_errors=True,
    )


class _EmailAuthModel(_StrictAuthModel):
    """需要规范化邮箱的认证模型公共边界."""

    email: EmailStr

    @field_validator("email", mode="before")
    @classmethod
    def normalize_email(cls, value: object) -> object:
        """在 EmailStr 格式验证前清理邮箱."""
        return _normalize_email(value)


class RegisterRequest(_EmailAuthModel):
    """创建用户账户时接收的公开请求.

    ``SecretStr`` 只负责减少 repr、日志和默认 JSON 序列化中的意外泄漏；它不是
    密码哈希。后续 AuthService 必须显式调用 ``get_secret_value()``，并立即把
    明文交给 PasswordHasher，不能把明文保存到 ORM、日志、异常或 Agent state。
    """

    password: SecretStr = Field(
        min_length=MIN_REGISTRATION_PASSWORD_LENGTH,
        max_length=MAX_PASSWORD_LENGTH,
        description="仅在注册处理期间短暂存在的明文密码",
    )


class LoginRequest(_EmailAuthModel):
    """使用已有账户凭据登录时接收的公开请求.

    登录只检查密码非空和安全上限，不重复使用注册时的最小长度规则。将来密码
    策略变严后，旧用户仍应能登录并在受控流程中升级密码。
    """

    # 密码不能 strip：空格可能是用户有意设置的密码字符。邮箱与密码的
    # 规范化规则不同，这是认证实现中很容易被忽略的一条数据边界。
    password: SecretStr = Field(
        min_length=1,
        max_length=MAX_PASSWORD_LENGTH,
        description="仅在密码校验期间短暂存在的明文密码",
    )


class TokenResponse(_StrictAuthModel):
    """认证成功后返回给客户端的 Bearer access token."""

    # access token 必须出现在 HTTP JSON 响应中，否则客户端无法认证后续请求；
    # repr=False 只阻止它出现在模型的调试表示中。它并不阻止 model_dump() 输出，
    # 因而应用仍然禁止记录整个 TokenResponse 或完整 Authorization header。
    access_token: str = Field(min_length=1, repr=False)
    token_type: Literal["bearer"] = "bearer"

    # AwareDatetime 拒绝没有时区的 datetime，避免客户端把本地时间误解为 UTC。
    expires_at: AwareDatetime


class AuthenticatedUser(_EmailAuthModel):
    """JWT 验证完成后在服务端内部传递的可信用户上下文.

    该对象不携带原始 JWT、密码或密码哈希。Route、application service、Agent
    runtime 和 tool 应只读取这里的 ``user_id``，不能采用 Prompt 或模型生成的
    user_id 作为授权依据。
    """

    user_id: UUID


# 控制导出访问和声明模块公开API边界
__all__ = [
    "AuthenticatedUser",
    "LoginRequest",
    "RegisterRequest",
    "TokenResponse",
]
