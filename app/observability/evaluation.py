"""离线评测结果所需的可复现版本元数据."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class EvalRunMetadata:
    """关联一次评测使用的代码、模型、Prompt 与数据集版本."""

    code_revision: str
    model_versions: tuple[str, ...]
    prompt_versions: tuple[str, ...]
    dataset_version: str

    def __post_init__(self) -> None:
        """拒绝无法复现的空版本字段."""
        values = (
            self.code_revision,
            self.dataset_version,
            *self.model_versions,
            *self.prompt_versions,
        )
        if not self.model_versions or not self.prompt_versions or any(not value.strip() for value in values):
            raise ValueError("evaluation metadata versions must not be empty")


__all__ = ["EvalRunMetadata"]
