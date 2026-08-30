"""加载、校验并标识版本化 Agent Prompt 资源."""

import json
from dataclasses import dataclass
from functools import lru_cache
from hashlib import sha256
from importlib.resources import files
from typing import Final


@dataclass(frozen=True, slots=True)
class PromptSpec:
    """一个逻辑 Prompt 的固定版本和低信任输入契约."""

    name: str
    version: str
    filename: str
    input_fields: frozenset[str]


@dataclass(frozen=True, slots=True)
class PromptArtifact:
    """一次模型调用可审计但不包含用户数据的 Prompt 工件."""

    name: str
    version: str
    content: str
    content_sha256: str


# 逻辑名称是业务代码唯一允许使用的入口；文件名和版本只能在这里升级。
# 输入字段用于阻止调用方悄悄遗漏数据或把身份、凭据等额外字段传给模型。
_PROMPT_SPECS: Final[dict[str, PromptSpec]] = {
    "chat_assistant": PromptSpec("chat_assistant", "v1", "chat_assistant.v1.md", frozenset()),
    "research_plan": PromptSpec("research_plan", "v2", "research_plan.v2.md", frozenset({"topic", "max_steps"})),
    "research_validate": PromptSpec(
        "research_validate", "v2", "research_validate.v2.md", frozenset({"topic", "plan", "evidence"})
    ),
    "research_write": PromptSpec(
        "research_write",
        "v2",
        "research_write.v2.md",
        frozenset({"topic", "facts", "conflicts", "validation_summary"}),
    ),
    "graphrag_extract": PromptSpec("graphrag_extract", "v2", "graphrag_extract.v2.md", frozenset({"content"})),
    "graphrag_extract_repair": PromptSpec(
        "graphrag_extract_repair", "v2", "graphrag_extract_repair.v2.md", frozenset({"content"})
    ),
    "graphrag_query_entity": PromptSpec(
        "graphrag_query_entity", "v2", "graphrag_query_entity.v2.md", frozenset({"query"})
    ),
    "graphrag_community_summary": PromptSpec(
        "graphrag_community_summary", "v2", "graphrag_community_summary.v2.md", frozenset({"facts"})
    ),
    "graphrag_global_map": PromptSpec(
        "graphrag_global_map", "v2", "graphrag_global_map.v2.md", frozenset({"question", "community"})
    ),
    "graphrag_global_reduce": PromptSpec(
        "graphrag_global_reduce", "v2", "graphrag_global_reduce.v2.md", frozenset({"question", "claims"})
    ),
    "chat_title": PromptSpec("chat_title", "v2", "chat_title.v2.md", frozenset({"user_message", "assistant_message"})),
    "memory_extract": PromptSpec(
        "memory_extract", "v2", "memory_extract.v2.md", frozenset({"user_message", "assistant_message"})
    ),
}


def get_prompt_spec(name: str) -> PromptSpec:
    """根据稳定逻辑名称返回不可变 Prompt 规格."""
    try:
        return _PROMPT_SPECS[name]
    except KeyError:
        raise ValueError(f"Unknown prompt: {name!r}") from None


@lru_cache(maxsize=None)
def load_prompt_artifact(name: str) -> PromptArtifact:
    """读取固定版本系统 Prompt，并计算可复现内容哈希."""
    spec = get_prompt_spec(name)
    content = files("app.agents.prompts").joinpath(spec.filename).read_text(encoding="utf-8").strip()
    if not content:
        raise ValueError(f"Prompt resource is empty: {name!r}")
    return PromptArtifact(
        name=spec.name,
        version=spec.version,
        content=content,
        content_sha256=sha256(content.encode("utf-8")).hexdigest(),
    )


def load_prompt(name: str) -> str:
    """兼容只需要系统指令正文的调用方."""
    return load_prompt_artifact(name).content


def render_prompt_input(name: str, /, **variables: object) -> str:
    """严格校验字段后，把低信任输入编码为确定性 JSON.

    JSON 只作为 HumanMessage 中的数据载体，绝不能与系统指令拼接。错误信息只
    包含字段名称，不包含用户正文、证据或凭据。
    """
    spec = get_prompt_spec(name)
    provided = frozenset(variables)
    missing = spec.input_fields - provided
    unexpected = provided - spec.input_fields
    if missing or unexpected:
        raise ValueError(f"Prompt inputs mismatch: missing={sorted(missing)!r}, unexpected={sorted(unexpected)!r}")
    return json.dumps(variables, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def registered_prompt_versions() -> tuple[str, ...]:
    """返回稳定排序的 ``name:version`` 集合，供启动检查和评测元数据使用."""
    return tuple(f"{name}:{_PROMPT_SPECS[name].version}" for name in sorted(_PROMPT_SPECS))


def load_all_prompt_artifacts() -> tuple[PromptArtifact, ...]:
    """在进程启动阶段验证所有已注册资源都存在且非空."""
    return tuple(load_prompt_artifact(name) for name in sorted(_PROMPT_SPECS))


__all__ = [
    "PromptArtifact",
    "PromptSpec",
    "get_prompt_spec",
    "load_all_prompt_artifacts",
    "load_prompt",
    "load_prompt_artifact",
    "registered_prompt_versions",
    "render_prompt_input",
]
