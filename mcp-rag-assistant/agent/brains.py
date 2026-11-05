from __future__ import annotations

import os

from .baselines import ExtractiveLLM
from .mcp_agent import LLM, AnthropicLLM, OpenAICompatLLM


def llm_from_env(model: str | None = None) -> tuple[LLM, str] | None:
    """Claude if ANTHROPIC_API_KEY is set, else any OpenAI-compatible API if LLM_MODEL is set."""
    if os.environ.get("ANTHROPIC_API_KEY"):
        name = model or os.environ.get("AGENT_MODEL", "claude-sonnet-5-5")
        return AnthropicLLM(model=name), f"Claude ({name})"
    if os.environ.get("LLM_MODEL"):
        name, base = os.environ["LLM_MODEL"], os.environ.get("LLM_BASE_URL")
        return OpenAICompatLLM(name, base, os.environ.get("LLM_API_KEY")), f"{name} via {base or 'api.openai.com'}"
    return None


def make_brain(kind: str = "auto", model: str | None = None) -> tuple[LLM, str]:
    """kind: 'baseline' (no LLM), 'llm' (from environment) or 'auto' (llm if configured, else baseline)."""
    if kind not in {"auto", "baseline", "llm"}:
        raise ValueError("brain must be auto, baseline or llm")
    if kind != "baseline":
        found = llm_from_env(model)
        if found:
            return found
        if kind == "llm":
            raise SystemExit("No LLM configured: set ANTHROPIC_API_KEY, or LLM_MODEL (+ LLM_BASE_URL, LLM_API_KEY).")
    return ExtractiveLLM(), "extractive baseline (no LLM: returns the top passage verbatim)"
