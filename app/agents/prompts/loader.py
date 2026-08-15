"""Load and render versioned Agent prompt templates."""

from dataclasses import dataclass
from functools import lru_cache
from importlib.resources import files
from typing import Final


@dataclass(frozen=True, slots=True)
class _PromptSpec:
    """描述一个逻辑 prompt 对应的文件及变量契约."""

    # filename 固定具体版本，调用方只使用稳定的逻辑名称。
    filename: str

    # variables 明确模板允许的全部输入，防止调用方与模板悄悄漂移。
    variables: frozenset[str]


# 这是 prompt 逻辑名称的唯一入口，不允许调用方传入任意文件路径。
# 将来升级到 v2 时，可以修改映射或扩展版本选择策略，而不污染 Agent node。
_PROMPT_SPECS: Final[dict[str, _PromptSpec]] = {
    "research_plan": _PromptSpec(
        filename="research_plan.v1.md",
        variables=frozenset({"topic", "max_steps"}),
    ),
}


def _get_prompt_spec(name: str) -> _PromptSpec:
    """根据稳定逻辑名称取得 prompt 配置."""
    try:
        return _PROMPT_SPECS[name]
    except KeyError:
        # 对外只暴露领域层面的未知名称，不泄露内部字典查找细节。
        raise ValueError(f"Unknown prompt: {name!r}") from None


@lru_cache(maxsize=None)
def load_prompt(name: str) -> str:
    """读取并缓存版本化 prompt 模板."""
    spec = _get_prompt_spec(name)

    # 使用包资源而不是当前工作目录拼路径，使 editable 安装和 wheel 安装
    # 都能定位同一份模板。这里只缓存静态原文，不缓存任何用户输入。
    prompt_file = files("app.agents.prompts").joinpath(spec.filename)
    return prompt_file.read_text(encoding="utf-8").strip()


def render_prompt(
    name: str,
    /,
    **variables: object,
) -> str:
    """严格校验变量集合并渲染指定 prompt."""
    spec = _get_prompt_spec(name)
    provided = frozenset(variables)
    missing = spec.variables - provided
    unexpected = provided - spec.variables

    if missing or unexpected:
        # 错误信息只记录变量名，不包含用户输入值或完整 prompt。
        raise ValueError(f"Prompt variables mismatch: missing={sorted(missing)!r}, unexpected={sorted(unexpected)!r}")

    # 只有变量集合完全匹配后才渲染，避免把未替换占位符发送给模型。
    return load_prompt(name).format_map(variables)
