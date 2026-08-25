"""使用真实 HMAC JWT 验收 TokenService 的签发与信任边界.

本 smoke 不 mock ``python-jose``。它在当前进程生成随机高强度 secret，真实签发
合法和故障 token，再通过生产 TokenService 验证。最终 JSON 只输出布尔结论与耗时，
不会输出 secret、完整 token 或 payload，适合保留为后续断点学习脚本。
"""

import json
import secrets
from datetime import UTC, datetime, timedelta
from time import perf_counter
from uuid import UUID, uuid4

from jose import jwt
from pydantic import SecretStr

from app.core.config import Settings
from app.services.auth import (
    AccessTokenExpiredError,
    InvalidAccessTokenError,
    TokenConfigurationError,
    TokenService,
)


def _elapsed_ms(started_at: float) -> float:
    """返回不含任何 credential 的毫秒耗时."""
    return round((perf_counter() - started_at) * 1000, 2)


def _base_payload(*, user_id: UUID) -> dict[str, object]:
    """构造由 smoke 自己签名的标准 payload，用于故障分支测试.

    正常 token 必须调用 ``TokenService.create_access_token``。这里直接编码只用于
    制造生产 API 不允许创建的错误状态，例如错误 token_type 或缺少 jti；随后仍由
    生产 ``decode_access_token`` 执行真实验签和结构校验。
    """
    issued_at = datetime.now(UTC)
    return {
        "sub": str(user_id),
        "iat": int(issued_at.timestamp()),
        "exp": int((issued_at + timedelta(minutes=5)).timestamp()),
        "jti": str(uuid4()),
        "token_type": "access",
    }


def _tamper_payload(token: str) -> str:
    """修改 JWT payload 的一个字符，同时保留原签名以触发验签失败."""
    header, payload, signature = token.split(".")
    index = len(payload) // 2
    replacement = "A" if payload[index] != "A" else "B"
    tampered_payload = f"{payload[:index]}{replacement}{payload[index + 1 :]}"
    return ".".join((header, tampered_payload, signature))


def _run_smoke() -> dict[str, object]:
    """覆盖正常签发、claims 校验和主要安全失败分支."""
    started_at = perf_counter()

    # token_urlsafe(48) 提供远高于 32 字节下限的随机 secret。它只存在于当前进程，
    # 不写环境文件，也不出现在 JSON 或异常文本中。
    raw_secret = secrets.token_urlsafe(48)
    secret = SecretStr(raw_secret)
    user_id = uuid4()
    service = TokenService(
        secret_key=secret,
        access_token_ttl=timedelta(minutes=30),
    )

    # 第一部分：真实签发两次。即使用户与签发秒相同，随机 jti 也应使 token 不同。
    first_response = service.create_access_token(user_id=user_id)
    second_response = service.create_access_token(user_id=user_id)
    first_claims = service.decode_access_token(first_response.access_token)
    second_claims = service.decode_access_token(second_response.access_token)

    valid_token_round_trip = first_claims.sub == user_id and first_claims.token_type == "access"
    unique_jti = first_claims.jti != second_claims.jti
    time_window_valid = (
        first_claims.iat.tzinfo is not None
        and first_claims.exp.tzinfo is not None
        and first_claims.exp > first_claims.iat
        and first_response.expires_at > datetime.now(UTC)
    )
    response_repr_hides_token = first_response.access_token not in repr(first_response)

    # 生产接线不会手写 timedelta，而是从 Settings 创建 service。这里用随机 secret
    # 覆盖 smoke 自己的 Settings 实例，验证字段名和“分钟”单位确实连接正确。
    runtime_config = Settings()
    runtime_config.JWT_SECRET_KEY = raw_secret
    runtime_config.JWT_ALGORITHM = "HS256"
    runtime_config.JWT_ACCESS_TOKEN_EXPIRE_MINUTES = 15
    configured_service = TokenService.from_settings(runtime_config)
    configured_response = configured_service.create_access_token(user_id=user_id)
    settings_factory_round_trip = (
        configured_service.decode_access_token(configured_response.access_token).sub == user_id
    )

    safe_error_messages: list[str] = []

    # 第二部分：只改 payload、不重签。即使修改后的内容仍像 JSON，原签名也不再匹配。
    try:
        service.decode_access_token(_tamper_payload(first_response.access_token))
    except InvalidAccessTokenError as exc:
        tampered_token_rejected = True
        safe_error_messages.append(str(exc))
    else:
        tampered_token_rejected = False

    # 第三部分：用另一把真实 secret 验证同一个 token，必须失败。
    wrong_secret_service = TokenService(
        secret_key=SecretStr(secrets.token_urlsafe(48)),
    )
    try:
        wrong_secret_service.decode_access_token(first_response.access_token)
    except InvalidAccessTokenError as exc:
        wrong_secret_rejected = True
        safe_error_messages.append(str(exc))
    else:
        wrong_secret_rejected = False

    # 第四部分：通过过去时钟真实签发一个已经过期的 token。正常 service 使用当前
    # 时钟 decode，python-jose 应在返回 payload 前抛 ExpiredSignatureError。
    past_service = TokenService(
        secret_key=secret,
        access_token_ttl=timedelta(minutes=1),
        clock=lambda: datetime.now(UTC) - timedelta(hours=2),
    )
    expired_token = past_service.create_access_token(user_id=user_id).access_token
    try:
        service.decode_access_token(expired_token)
    except AccessTokenExpiredError as exc:
        expired_token_rejected = True
        safe_error_messages.append(str(exc))
    else:
        expired_token_rejected = False

    # 第五部分：签名完全正确，但用途不对。签名只能证明“内容没被改”，不能证明
    # “这个 token 适合当前接口”，因此 token_type 还必须经过 Literal 校验。
    wrong_type_payload = _base_payload(user_id=user_id)
    wrong_type_payload["token_type"] = "session"
    wrong_type_token = jwt.encode(wrong_type_payload, raw_secret, algorithm="HS256")
    try:
        service.decode_access_token(wrong_type_token)
    except InvalidAccessTokenError as exc:
        wrong_token_type_rejected = True
        safe_error_messages.append(str(exc))
    else:
        wrong_token_type_rejected = False

    # 第六部分：签名正确但缺少 jti。require_jti 与 Pydantic 模型形成双层保护，
    # 缺少身份追踪字段的 token 不能降级成部分可信对象。
    missing_claim_payload = _base_payload(user_id=user_id)
    del missing_claim_payload["jti"]
    missing_claim_token = jwt.encode(
        missing_claim_payload,
        raw_secret,
        algorithm="HS256",
    )
    try:
        service.decode_access_token(missing_claim_token)
    except InvalidAccessTokenError as exc:
        missing_claim_rejected = True
        safe_error_messages.append(str(exc))
    else:
        missing_claim_rejected = False

    # 第七部分：python-jose 只检查 iat 是整数，因此额外制造未来签发时间，验证
    # TokenService 自己的时钟偏差检查确实生效。
    future_payload = _base_payload(user_id=user_id)
    future_issued_at = datetime.now(UTC) + timedelta(hours=1)
    future_payload["iat"] = int(future_issued_at.timestamp())
    future_payload["exp"] = int((future_issued_at + timedelta(minutes=5)).timestamp())
    future_token = jwt.encode(future_payload, raw_secret, algorithm="HS256")
    try:
        service.decode_access_token(future_token)
    except InvalidAccessTokenError as exc:
        future_iat_rejected = True
        safe_error_messages.append(str(exc))
    else:
        future_iat_rejected = False

    # 最后一项验证配置 Gate。本项目不允许示例 secret 进入任何环境的 TokenService；
    # production 因此会明确拒绝启动认证组件，而不是静默使用公开默认值。
    try:
        TokenService(secret_key=SecretStr("change-me-in-production"))
    except TokenConfigurationError as exc:
        insecure_default_rejected = True
        safe_error_messages.append(str(exc))
    else:
        insecure_default_rejected = False

    errors_hide_credentials = all(
        raw_secret not in message and first_response.access_token not in message and expired_token not in message
        for message in safe_error_messages
    )

    ok = all(
        (
            valid_token_round_trip,
            unique_jti,
            time_window_valid,
            response_repr_hides_token,
            settings_factory_round_trip,
            tampered_token_rejected,
            wrong_secret_rejected,
            expired_token_rejected,
            wrong_token_type_rejected,
            missing_claim_rejected,
            future_iat_rejected,
            insecure_default_rejected,
            errors_hide_credentials,
        )
    )
    return {
        "ok": ok,
        "valid_token_round_trip": valid_token_round_trip,
        "unique_jti": unique_jti,
        "time_window_valid": time_window_valid,
        "response_repr_hides_token": response_repr_hides_token,
        "settings_factory_round_trip": settings_factory_round_trip,
        "tampered_token_rejected": tampered_token_rejected,
        "wrong_secret_rejected": wrong_secret_rejected,
        "expired_token_rejected": expired_token_rejected,
        "wrong_token_type_rejected": wrong_token_type_rejected,
        "missing_claim_rejected": missing_claim_rejected,
        "future_iat_rejected": future_iat_rejected,
        "insecure_default_rejected": insecure_default_rejected,
        "errors_hide_credentials": errors_hide_credentials,
        "elapsed_ms": _elapsed_ms(started_at),
    }


def main() -> int:
    """打印一行不含 token/secret 的 JSON 验收摘要."""
    started_at = perf_counter()
    try:
        summary = _run_smoke()
    except Exception as exc:
        print(
            json.dumps(
                {
                    "ok": False,
                    "error_type": type(exc).__name__,
                    "elapsed_ms": _elapsed_ms(started_at),
                }
            )
        )
        return 1

    print(json.dumps(summary))
    return 0 if summary["ok"] is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
