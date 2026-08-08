"""跨 API 共享的基础响应模型."""

from pydantic import BaseModel


class ErrorDetail(BaseModel):
    """单个字段级校验错误."""

    field: str
    message: str
    type: str


class ErrorBody(BaseModel):
    """统一错误响应中的 error 对象."""

    code: str
    message: str
    details: list[ErrorDetail] | None = None


class ErrorResponse(BaseModel):
    """所有 HTTP 错误共享的顶层响应结构."""

    error: ErrorBody
    request_id: str
