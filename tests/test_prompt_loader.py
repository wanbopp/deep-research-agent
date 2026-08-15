"""Versioned Agent prompt loader contract tests."""

import pytest

from app.agents.prompts.loader import load_prompt, render_prompt


def test_prompt_loader_renders_caches_and_rejects_invalid_inputs() -> None:
    """Prompt loader 应隔离渲染内容、缓存模板并严格校验变量."""
    # 测试不能依赖其他测试或模块导入留下的缓存状态。
    load_prompt.cache_clear()

    # 渲染两种不同的请求
    first = render_prompt(
        "research_plan",
        topic="AI safety",
        max_steps=2,
    )
    second = render_prompt(
        "research_plan",
        topic="RAG evaluation",
        max_steps=3,
    )

    assert "AI safety" in first
    assert "RAG evaluation" not in first

    assert "RAG evaluation" in second
    assert "AI safety" not in second

    assert "{topic}" not in first
    assert "{max_steps}" not in first

    # 检查缓存
    cache_info = load_prompt.cache_info()

    assert cache_info.misses == 1
    assert cache_info.hits >= 1

    # 合并检查三种失败
    private_value = "PRIVATE-TOPIC-MUST-NOT-LEAK"

    # 缺少变量
    with pytest.raises(ValueError) as missing_error:
        render_prompt(
            "research_plan",
            topic=private_value,
        )

    assert "max_steps" in str(missing_error.value)
    assert private_value not in str(missing_error.value)

    # 额外变量
    with pytest.raises(ValueError) as unexpected_error:
        render_prompt(
            "research_plan",
            topic="AI safety",
            max_steps=2,
            private_note=private_value,
        )

    assert "private_note" in str(unexpected_error.value)
    assert private_value not in str(unexpected_error.value)

    # 未知逻辑名称
    with pytest.raises(ValueError, match="Unknown prompt"):
        load_prompt("not_registered")
