"""Command line front-end.

    python -m agent.cli --demo                 # scripted tour, safe defaults
    python -m agent.cli -t "add note Hi: there then list notes"
    python -m agent.cli                        # interactive chat
    python -m agent.cli --offline --approval yes

Uses Claude when ANTHROPIC_API_KEY is set, otherwise the offline rule-based planner.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path
from typing import Any

from .mcp_agent import (
    ROOT,
    AnthropicLLM,
    Event,
    MCPAgent,
    OfflineLLM,
    OpenAICompatLLM,
    approve_all,
    deny_all,
    load_config,
)

ICONS = {"tool_call": "->", "tool_result": "<-", "tool_error": "!!", "approval": "??", "assistant": "AI"}

DEMO_TASKS = [
    "add note MCP basics: servers expose tools, resources and prompts over JSON-RPC",
    "add note Agent loop: reason, call a tool, observe, repeat",
    "search notes for agent then list notes",
    "calculate (12.5 * 4) / 3 + 2**8",
    "calculate __import__('os').system('id')",  # rejected safely: the calculator never uses eval()
    "read file ../config.json",  # rejected safely: path traversal outside the sandbox
    "delete note 1",  # destructive -> needs human approval
]


def print_event(ev: Event) -> None:
    print(f"   {ICONS.get(ev.kind, '  ')} {ev.kind:<11} {ev.text[:300]}")


async def ask_human(name: str, args: dict[str, Any]) -> bool:
    answer = await asyncio.to_thread(input, f"   Allow {name} with {args}? [y/N] ")
    return answer.strip().lower() in {"y", "yes"}


def pick_approval(mode: str):
    if mode == "yes":
        return approve_all
    if mode == "no":
        return deny_all
    return ask_human if sys.stdin.isatty() else deny_all  # unattended -> safe default


def build_llm(args: argparse.Namespace):
    """Pick the brain: Claude, any OpenAI-compatible API, or the offline planner."""
    if not args.offline:
        if os.environ.get("ANTHROPIC_API_KEY"):
            return AnthropicLLM(model=args.model), f"Claude ({args.model})"
        if os.environ.get("LLM_MODEL"):
            base = os.environ.get("LLM_BASE_URL")
            llm = OpenAICompatLLM(os.environ["LLM_MODEL"], base, os.environ.get("LLM_API_KEY"))
            return llm, f"{os.environ['LLM_MODEL']} via {base or 'api.openai.com'}"
    return OfflineLLM(), "offline rule-based planner (set LLM_MODEL, or ANTHROPIC_API_KEY, for a real model)"


async def amain(args: argparse.Namespace) -> int:
    specs, needs_approval = load_config(args.config)
    llm, label = build_llm(args)
    print(f"[brain] {label}")

    async with MCPAgent(specs, llm, needs_approval, pick_approval(args.approval), on_event=print_event) as agent:
        print(f"[mcp] connected to {len(agent.clients)} servers, discovered {len(agent.tools)} tools:")
        for name, t in agent.tools.items():
            print(f"      {name:<22} {t.description.splitlines()[0][:70]}")

        if args.demo:
            for task in DEMO_TASKS:
                print(f"\n> {task}")
                result = await agent.run(task)
                print(f"   answer: {result.answer}")
            print("\n[mcp] resources:", await agent.list_resources())
            print("[mcp] notes://stats ->", await agent.read_resource("notes", "notes://stats"))
            print("[mcp] prompt ->", await agent.get_prompt("notes", "summarize_notes", {"topic": "agents"}))
            return 0

        if args.task:
            result = await agent.run(args.task)
            print(f"\n{result.answer}")
            return 0

        history: list[dict] = []
        print("\nType a task, or /tools, /resources, /quit.")
        while True:
            try:
                line = (await asyncio.to_thread(input, "\n> ")).strip()
            except EOFError:
                break
            if line in {"/quit", "/exit"}:
                break
            if line == "/tools":
                print("\n".join(agent.tools))
            elif line == "/resources":
                print(await agent.list_resources())
            elif line:
                result = await agent.run(line, history)
                history = result.messages
                print(f"\n{result.answer}")
    return 0


def main() -> None:
    p = argparse.ArgumentParser(description="MCP agent")
    p.add_argument("-t", "--task", help="run one task and exit")
    p.add_argument("--demo", action="store_true", help="run the scripted demo")
    p.add_argument("--offline", action="store_true", help="force the offline planner")
    p.add_argument("--approval", choices=["ask", "yes", "no"], default="ask", help="how to handle approval-gated tools")
    p.add_argument("--model", default=os.environ.get("AGENT_MODEL", "claude-sonnet-5-5"))
    p.add_argument("--config", default=str(ROOT / "config.json"))
    raise SystemExit(asyncio.run(amain(p.parse_args())))


if __name__ == "__main__":
    main()
