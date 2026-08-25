"""在无网络、无数据库条件下验收业务会话 API schema.

这个 smoke 只验证公开数据结构，不验证 Repository 所有权查询或 checkpoint 清理。
它不会调用模型；10F 后续真正改变 Agent 行为时，仍需运行真实 provider Gate。
"""

import json
from datetime import UTC, datetime, timedelta
from uuid import uuid4

from pydantic import ValidationError

from app.schemas.chat_session import (
    ChatSessionCreateRequest,
    ChatSessionListResponse,
    ChatSessionResponse,
    DEFAULT_CHAT_SESSION_TITLE,
    MAX_CHAT_SESSION_TITLE_LENGTH,
)


def _rejects_create_payload(payload: dict[str, object]) -> bool:
    """检查创建请求是否拒绝指定 payload.

    Args:
        payload: 模拟客户端提交的 JSON 对象。

    Returns:
        Pydantic 拒绝输入时返回 True，否则返回 False。
    """
    try:
        ChatSessionCreateRequest.model_validate(payload)
    except ValidationError:
        return True
    return False


def _rejects_response_payload(payload: dict[str, object]) -> bool:
    """检查响应模型是否拒绝矛盾或不完整的 service 输出.

    Args:
        payload: 模拟 application service 组装的数据。

    Returns:
        Pydantic 拒绝数据时返回 True，否则返回 False。
    """
    try:
        ChatSessionResponse.model_validate(payload)
    except ValidationError:
        return True
    return False


def main() -> int:
    """执行 10F-A schema smoke，并输出不包含业务数据的布尔摘要."""
    now = datetime.now(UTC)
    first_thread_id = uuid4()
    second_thread_id = uuid4()

    # 创建请求只允许标题。首尾空白在长度校验前清理，未提交标题时使用稳定默认值。
    normalized_request = ChatSessionCreateRequest(title="  Research notes  ")
    default_request = ChatSessionCreateRequest()

    # 响应使用业务 ChatSession.id 作为公开 thread_id。两个时间字段必须带时区，
    # 且 updated_at 不能早于 created_at。
    first_response = ChatSessionResponse(
        thread_id=first_thread_id,
        title=normalized_request.title,
        created_at=now,
        updated_at=now,
    )
    second_response = ChatSessionResponse(
        thread_id=second_thread_id,
        title=default_request.title,
        created_at=now,
        updated_at=now + timedelta(seconds=1),
    )

    # tuple 保持 Python 侧不可变；序列化后仍是普通 JSON array。
    collection = ChatSessionListResponse(sessions=(first_response, second_response))
    first_payload = json.loads(first_response.model_dump_json())
    collection_payload = json.loads(collection.model_dump_json())

    # 直接赋值会触发 Pydantic 的运行时 frozen 校验；只有真的抛出
    # ValidationError，才能证明调用方无法修改已经验证的请求对象。
    frozen_rejected = False
    try:
        normalized_request.title = "Changed"
    except ValidationError:
        frozen_rejected = True

    checks = {
        "title_is_stripped": normalized_request.title == "Research notes",
        "default_title_is_stable": default_request.title == DEFAULT_CHAT_SESSION_TITLE,
        "blank_title_rejected": _rejects_create_payload({"title": "   "}),
        "long_title_rejected": _rejects_create_payload({"title": "x" * (MAX_CHAT_SESSION_TITLE_LENGTH + 1)}),
        "extra_field_rejected": _rejects_create_payload({"title": "Research notes", "user_id": str(uuid4())}),
        "model_is_frozen": frozen_rejected,
        "thread_id_is_uuid": first_response.thread_id == first_thread_id,
        "thread_id_serializes_as_uuid": first_payload["thread_id"] == str(first_thread_id),
        "naive_timestamp_rejected": _rejects_response_payload(
            {
                "thread_id": first_thread_id,
                "title": "Research notes",
                "created_at": datetime.now(),
                "updated_at": datetime.now(),
            }
        ),
        "reversed_timestamp_rejected": _rejects_response_payload(
            {
                "thread_id": first_thread_id,
                "title": "Research notes",
                "created_at": now,
                "updated_at": now - timedelta(seconds=1),
            }
        ),
        "list_order_is_stable": [item["thread_id"] for item in collection_payload["sessions"]]
        == [str(first_thread_id), str(second_thread_id)],
        "list_serializes_as_array": isinstance(collection_payload["sessions"], list),
    }

    ok = all(checks.values())
    print(json.dumps({"ok": ok, **checks}))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
