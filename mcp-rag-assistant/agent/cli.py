from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path

from .brains import make_brain
from .mcp_agent import ROOT, DEFAULT_SYSTEM, Event, MCPAgent, deny_all, load_config

ICONS = {"tool_call": "->", "tool_result": "<-", "tool_error": "!!", "approval": "??", "assistant": "AI"}

DEMO_QUESTIONS = [
    "Who introduced MCP and when?",
    "How should a file-reading tool prevent path traversal?",
    "What constant is commonly used in reciprocal rank fusion?",
    "What is the capital of Australia?",  
]


def show_event(ev: Event) -> None:
    if ev.kind == "assistant":
        return  
    first_line = ev.text.splitlines()[0] if ev.text else ""
    print(f"   {ICONS.get(ev.kind, '  ')} {ev.kind:<11} {first_line[:140]}")


async def amain(args: argparse.Namespace) -> int:
    specs, approvals = load_config(args.config)
    system = json.loads(Path(args.config).read_text()).get("system_prompt", DEFAULT_SYSTEM)
    llm, label = make_brain(args.brain, args.model)
    print(f"[brain] {label}")

    async with MCPAgent(specs, llm, approvals, deny_all, system=system, on_event=show_event) as agent:
        print(f"[mcp] connected; tools: {', '.join(agent.tools)}")
        if args.demo:
            for question in DEMO_QUESTIONS:
                print(f"\n> {question}")
                result = await agent.run(question)
                print(f"   answer: {result.answer}")
            print("\n[mcp] resource rag://stats ->", await agent.read_resource("docs", "rag://stats"))
            return 0
        if args.task:
            print(f"\n{(await agent.run(args.task)).answer}")
            return 0

        history: list[dict] = []
        print("\nAsk about your documents. Commands: /docs, /reindex, /quit")
        while True:
            try:
                line = (await asyncio.to_thread(input, "\n> ")).strip()
            except EOFError:
                break
            if line in {"/quit", "/exit"}:
                break
            if line == "/docs":
                print(await agent.read_resource("docs", "rag://documents"))
            elif line == "/reindex":
                print((await agent.clients["docs"].call_tool("index_documents", {})).content[0].text)
            elif line:
                result = await agent.run(line, history)
                history = result.messages
                print(f"\n{result.answer}")
    return 0


def main() -> None:
    p = argparse.ArgumentParser(description="RAG assistant on MCP")
    p.add_argument("-t", "--task", help="ask one question and exit")
    p.add_argument("--demo", action="store_true", help="run the scripted demo questions")
    p.add_argument("--brain", choices=["auto", "baseline", "llm"], default="auto")
    p.add_argument("--model", default=None, help="Claude model name (only used with ANTHROPIC_API_KEY)")
    p.add_argument("--config", default=str(ROOT / "config.json"))
    raise SystemExit(asyncio.run(amain(p.parse_args())))


if __name__ == "__main__":
    main()
