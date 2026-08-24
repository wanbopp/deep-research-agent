"""在无网络、无数据库条件下验收认证 schema 的安全数据边界."""

import json
from datetime import UTC, datetime
from uuid import uuid4

from pydantic import SecretStr, ValidationError

from app.schemas.auth import (
    AuthenticatedUser,
    LoginRequest,
    RegisterRequest,
    TokenResponse,
)


def _rejects_invalid_registration(payload: dict[str, object]) -> bool:
    """返回注册 payload 是否被 Pydantic 正确拒绝."""
    try:
        RegisterRequest.model_validate(payload)
    except ValidationError:
        return True
    return False


def _rejects_naive_expiration() -> bool:
    """验证 TokenResponse 拒绝没有时区的过期时间."""
    try:
        TokenResponse(
            access_token=".".join(("header", "payload", "signature")),
            expires_at=datetime.now(),
        )
    except ValidationError:
        return True
    return False


def _validation_error_hides_password() -> bool:
    """验证 Pydantic 异常文本不会包含被拒绝的密码原文."""
    rejected_password = "too-short"
    try:
        RegisterRequest(
            email="learner@example.com",
            password=SecretStr(rejected_password),
        )
    except ValidationError as exc:
        return rejected_password not in str(exc)
    return False


def main() -> int:
    """执行认证 schema 的最小安全 smoke 并打印布尔摘要."""
    # 全部值都是本进程临时生成的合成数据。变量内容不会进入最终输出，避免
    # 养成“为了验收而打印密码/token”的危险调试习惯。
    registration_password = "R" * 12
    login_password = " padded login value "
    access_token = ".".join(("header", "payload", "signature"))

    registration = RegisterRequest(
        email="  Learner@Example.COM  ",
        password=SecretStr(registration_password),
    )
    login = LoginRequest(
        email="Learner@Example.COM",
        password=SecretStr(login_password),
    )
    token = TokenResponse(
        access_token=access_token,
        expires_at=datetime.now(UTC),
    )
    authenticated_user = AuthenticatedUser(
        user_id=uuid4(),
        email=registration.email,
    )

    registration_json = registration.model_dump_json()
    registration_repr = repr(registration)
    token_repr = repr(token)
    token_payload = token.model_dump(mode="json")

    checks = {
        # 邮箱属于标识符，可以安全清理空白并统一大小写。
        "email_normalized": str(registration.email) == "learner@example.com",
        # 密码属于秘密原文，不能像邮箱一样 strip 或 casefold。
        "login_password_preserved": login.password.get_secret_value() == login_password,
        # SecretStr 的职责是降低 repr/默认 JSON 中意外泄漏，不代表已经哈希。
        "password_hidden_from_repr": registration_password not in registration_repr,
        "password_hidden_from_json": registration_password not in registration_json,
        "password_hidden_from_validation_error": _validation_error_hides_password(),
        # 注册执行当前密码下限；登录仍允许校验历史上较短的非空密码。
        "short_registration_rejected": _rejects_invalid_registration(
            {
                "email": "learner@example.com",
                "password": "short",
            }
        ),
        "short_login_accepted": LoginRequest(
            email="learner@example.com",
            password=SecretStr("x"),
        ).password.get_secret_value()
        == "x",
        "extra_field_rejected": _rejects_invalid_registration(
            {
                "email": "learner@example.com",
                "password": registration_password,
                "role": "admin",
            }
        ),
        # token 必须进入 HTTP JSON 响应，但 repr 中不能出现完整凭据。
        "token_hidden_from_repr": access_token not in token_repr,
        "token_available_to_response": token_payload["access_token"] == access_token,
        "token_type_is_bearer": token.token_type == "bearer",
        "naive_expiration_rejected": _rejects_naive_expiration(),
        # 可信用户上下文只保留身份字段，不携带密码或 token。
        "authenticated_user_is_minimal": set(authenticated_user.model_fields_set) == {"user_id", "email"},
    }

    ok = all(checks.values())
    print(json.dumps({"ok": ok, **checks}))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
