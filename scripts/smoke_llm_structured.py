"""Run one bounded real-provider smoke test for structured LLM output."""

import asyncio
import json
import logging
import os
from time import perf_counter
from app.agents.prompts.loader import render_prompt  # noqa: E402
from langchain_core.messages import SystemMessage  # noqa: E402

# Keep provider SDK debug logs from printing request payloads during the smoke test.
# The current application logger derives its level from DEBUG rather than LOG_LEVEL.
os.environ["DEBUG"] = "false"
MAX_STEPS = 1
from pydantic import SecretStr  # noqa: E402

from app.core.config import settings  # noqa: E402
from app.schemas.llm import ModelSpec  # noqa: E402
from app.schemas.research import ResearchPlan  # noqa: E402
from app.services.llm.factory import create_openai_chat_model  # noqa: E402
from app.services.llm.registry import LLMRegistry  # noqa: E402
from app.services.llm.service import LLMService  # noqa: E402

# The logging module has finished configuring its handlers at this point. Raising
# both the root logger and its handlers keeps provider request payloads out of stdout.
logging.getLogger().setLevel(logging.WARNING)
for configured_handler in logging.getLogger().handlers:
    configured_handler.setLevel(logging.WARNING)

EXPECTED_TOPIC = "Python structured output smoke"


async def run_smoke() -> int:
    """Call the configured provider once and print only safe result metadata."""
    started_at = perf_counter()

    if not settings.OPENAI_API_KEY:
        print(json.dumps({"ok": False, "error_type": "MissingApiKey"}))
        return 1

    spec = ModelSpec(
        alias="primary",
        provider_model=settings.DEFAULT_LLM_MODEL,
        api_key=SecretStr(settings.OPENAI_API_KEY),
        base_url=settings.OPENAI_BASE_URL,
        temperature=settings.DEFAULT_LLM_TEMPERATURE,
        # GPT-5 may consume part of the completion budget before emitting data.
        # 1024 is bounded but leaves room for reasoning and one structured step.
        max_tokens=min(settings.MAX_TOKENS, 1024),
    )
    registry = LLMRegistry([spec], create_openai_chat_model)
    service = LLMService(
        registry,
        # A smoke test should not repeat a paid request when configuration is wrong.
        max_attempts=1,
        retry_wait_multiplier=0,
        # The provider can take more than 30 seconds for GPT-5 even on a tiny prompt.
        # Keep the project's configured 60-second ceiling without enabling retries.
        total_timeout_seconds=min(settings.LLM_TOTAL_TIMEOUT, 60),
    )

    rendered_prompt = render_prompt(
        "research_plan",
        topic=EXPECTED_TOPIC,
        max_steps=MAX_STEPS,
    )
    try:
        plan = await service.call_structured(
            [
                SystemMessage(
                    content=rendered_prompt,
                ),
            ],
            response_model=ResearchPlan,
            aliases=("primary",),
        )
    except Exception as error:
        print(
            json.dumps(
                {
                    "ok": False,
                    "error_type": type(error).__name__,
                    "schema_name": getattr(error, "schema_name", None),
                    "structured_error_type": getattr(error, "error_type", None),
                    "status_code": getattr(error, "status_code", None),
                    "elapsed_ms": round((perf_counter() - started_at) * 1000, 2),
                }
            )
        )
        return 1

    first_step = plan.steps[0] if plan.steps else None

    topic_matches = plan.topic == EXPECTED_TOPIC
    step_count = len(plan.steps)
    query_count = len(first_step.search_queries) if first_step is not None else 0

    ok = (
        isinstance(plan, ResearchPlan)
        and topic_matches
        and 1 <= step_count <= MAX_STEPS
        and first_step is not None
        and first_step.step_number == 1
        and bool(first_step.objective.strip())
        and query_count >= 1
    )

    print(
        json.dumps(
            {
                "ok": ok,
                "response_type": type(plan).__name__,
                "model": settings.DEFAULT_LLM_MODEL,
                "topic_matches": topic_matches,
                "step_count": step_count,
                "query_count": query_count,
                "elapsed_ms": round(
                    (perf_counter() - started_at) * 1000,
                    2,
                ),
            }
        )
    )

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(run_smoke()))
