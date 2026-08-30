"""工具曝光、风险和批准策略."""

from app.tools.contracts import ToolDescriptor, ToolExecutionContext, ToolExposure


class ToolAuthorizationError(PermissionError):
    """工具未向模型开放."""


class ToolApprovalRequired(PermissionError):
    """高风险工具缺少服务端确认的批准事实."""


class ToolPolicy:
    """在执行和网络调用之前实施确定性授权."""

    def authorize(self, descriptor: ToolDescriptor, context: ToolExecutionContext) -> None:
        """模型只能调用 MODEL 工具；批准集合必须来自可信上下文."""
        if descriptor.exposure is not ToolExposure.MODEL:
            raise ToolAuthorizationError("tool is not exposed to the model")
        if descriptor.requires_approval and descriptor.name not in context.approved_tool_names:
            raise ToolApprovalRequired("tool approval is required")


__all__ = ["ToolApprovalRequired", "ToolAuthorizationError", "ToolPolicy"]
