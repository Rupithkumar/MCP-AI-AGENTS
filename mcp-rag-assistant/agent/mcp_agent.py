"""A small MCP agent.

Architecture
------------
    user task
        |
        v
   MCPAgent.run()  --- loop --->  LLM.next_turn()      (Claude, or the offline planner)
        |                              |
        |  tool_use blocks             |
        v                              v
   approval gate  ->  MCP Client  ->  MCP server (stdio subprocess)
        |
        v
   tool_result blocks fed back to the LLM until it answers without calling a tool

The agent is a *host* in MCP terms: it owns one MCP `Client` (1:1 connection) per
configured server, discovers their tools with tools/list, and forwards the model's
tool calls with tools/call. Tool names are namespaced `<server>__<tool>` so two
servers can expose tools with the same name.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Protocol

from mcp import Client
from mcp.client.stdio import StdioServerParameters

ROOT = Path(__file__).resolve().parent.parent
SEP = "__"
MAX_RESULT_CHARS = 8_000  

DEFAULT_SYSTEM = (
    "You are a helpful assistant with access to tools provided by MCP servers. "
    "Use tools when they help; never invent tool results. If a tool returns an error, "
    "read it and try to correct the call. If the user denies an action, accept that and "
    "do not retry it. Answer concisely."
)


@dataclass
class ServerSpec:
    name: str
    command: str
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)


def _subst(value: str) -> str:
    value = value.replace("${ROOT}", str(ROOT))
    return re.sub(r"\$\{env:([A-Za-z_][A-Za-z0-9_]*)\}", lambda m: os.environ.get(m.group(1), ""), value)


def load_config(path: str | Path = ROOT / "config.json") -> tuple[list[ServerSpec], set[str]]:
    data = json.loads(Path(path).read_text())
    specs = []
    for name, cfg in data["servers"].items():
        command = sys.executable if cfg["command"] == "python" else cfg["command"]
        specs.append(
            ServerSpec(
                name=name,
                command=command,
                args=[_subst(a) for a in cfg.get("args", [])],
                env={k: _subst(v) for k, v in cfg.get("env", {}).items()},
            )
        )
    return specs, set(data.get("requires_approval", []))



@dataclass
class ToolCall:
    id: str
    name: str
    args: dict[str, Any]


@dataclass
class Turn:
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0


class LLM(Protocol):
    async def next_turn(self, system: str, messages: list[dict], tools: list[dict]) -> Turn: ...


class AnthropicLLM:
    def __init__(self, model: str = "claude-sonnet-5-5", client: Any = None, max_tokens: int = 2048):
        if client is None:
            from anthropic import AsyncAnthropic

            client = AsyncAnthropic()
        self.client, self.model, self.max_tokens = client, model, max_tokens

    async def next_turn(self, system: str, messages: list[dict], tools: list[dict]) -> Turn:
        resp = await self.client.messages.create(
            model=self.model, max_tokens=self.max_tokens, system=system, tools=tools, messages=messages
        )
        turn = Turn()
        for block in resp.content:
            if block.type == "text":
                turn.text += block.text
            elif block.type == "tool_use":
                turn.tool_calls.append(ToolCall(block.id, block.name, dict(block.input)))
        usage = getattr(resp, "usage", None)
        turn.input_tokens = getattr(usage, "input_tokens", 0) or 0
        turn.output_tokens = getattr(usage, "output_tokens", 0) or 0
        return turn


class OpenAICompatLLM:
    def __init__(self, model: str, base_url: str | None = None, api_key: str | None = None,
                 client: Any = None):
        if client is None:
            from openai import AsyncOpenAI

            client = AsyncOpenAI(base_url=base_url, api_key=api_key or "not-needed")  
        self.client, self.model = client, model

    @staticmethod
    def convert_messages(system: str, messages: list[dict]) -> list[dict]:
        out: list[dict] = [{"role": "system", "content": system}]
        for m in messages:
            content = m["content"]
            if isinstance(content, str):
                out.append({"role": m["role"], "content": content})
            elif m["role"] == "assistant":
                text = "".join(b["text"] for b in content if b["type"] == "text")
                calls = [
                    {"id": b["id"], "type": "function",
                     "function": {"name": b["name"], "arguments": json.dumps(b["input"])}}
                    for b in content if b["type"] == "tool_use"
                ]
                msg: dict[str, Any] = {"role": "assistant", "content": text or None}
                if calls:
                    msg["tool_calls"] = calls
                out.append(msg)
            else:  
                for b in content:
                    if b["type"] == "tool_result":
                        prefix = "ERROR: " if b.get("is_error") else ""
                        out.append({"role": "tool", "tool_call_id": b["tool_use_id"], "content": prefix + b["content"]})
        return out

    async def next_turn(self, system: str, messages: list[dict], tools: list[dict]) -> Turn:
        oa_tools = [
            {"type": "function", "function": {"name": t["name"], "description": t["description"], "parameters": t["input_schema"]}}
            for t in tools
        ]
        resp = await self.client.chat.completions.create(
            model=self.model, messages=self.convert_messages(system, messages), tools=oa_tools,
        )  
        msg = resp.choices[0].message
        usage = getattr(resp, "usage", None)
        turn = Turn(
            text=msg.content or "",
            input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            output_tokens=getattr(usage, "completion_tokens", 0) or 0,
        )
        for i, tc in enumerate(msg.tool_calls or []):
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError: 
                args = {}
            turn.tool_calls.append(ToolCall(tc.id or f"call_{len(messages)}_{i}", tc.function.name, args))
        return turn


class OfflineLLM:
    HELP = (
        "(offline mode) Try: 'add note Title: body', 'search notes for X', 'list notes', "
        "'delete note 1', 'calculate 2*(3+4)', 'what time is it', 'read file hello.txt'. "
        "Chain with 'then'."
    )

    def __init__(self) -> None:
        self._n = 0

    async def next_turn(self, system: str, messages: list[dict], tools: list[dict]) -> Turn:
        last = messages[-1]["content"]
        if isinstance(last, list):  
            lines = [b["content"] for b in last if b.get("type") == "tool_result"]
            return Turn(text="\n".join(lines))
        names = {t["name"] for t in tools}
        calls = []
        for part in re.split(r"\s+then\s+", last.strip(), flags=re.I):
            call = self._plan(part.strip().rstrip("?.!"))
            if call and call[0] in names:
                self._n += 1
                calls.append(ToolCall(f"offline-{self._n}", call[0], call[1]))
        return Turn(tool_calls=calls) if calls else Turn(text=self.HELP)

    @staticmethod
    def _plan(text: str) -> tuple[str, dict] | None:
        low = text.lower()
        if m := re.match(r"(?:add|save|create) (?:a )?note[: ]+(.+)", text, re.I):
            title, _, body = m.group(1).partition(":")
            return f"notes{SEP}add_note", {"title": title.strip(), "body": (body or title).strip()}
        if m := re.match(r"(?:search|find)(?: notes)?(?: for| about)? (.+)", text, re.I):
            return f"notes{SEP}search_notes", {"query": m.group(1).strip()}
        if re.match(r"(?:list|show)(?: all| my)? notes", low):
            return f"notes{SEP}list_notes", {}
        if m := re.match(r"delete note #?(\d+)", low):
            return f"notes{SEP}delete_note", {"note_id": int(m.group(1))}
        if m := re.match(r"(?:calculate|calc|compute|what is|what's)\s+(.+)", text, re.I):
            return f"utils{SEP}calculate", {"expression": m.group(1).strip()}
        if "time" in low:
            return f"utils{SEP}current_time", {}
        if m := re.match(r"read(?: file)? (\S+)", text, re.I):
            return f"utils{SEP}read_text_file", {"path": m.group(1)}
        return None


ApprovalFn = Callable[[str, dict[str, Any]], Awaitable[bool]]


async def deny_all(name: str, args: dict[str, Any]) -> bool:
    return False


async def approve_all(name: str, args: dict[str, Any]) -> bool:
    return True


@dataclass
class Event:
    kind: str  
    text: str


@dataclass
class RunResult:
    answer: str
    messages: list[dict]
    events: list[Event]
    steps: int
    input_tokens: int = 0
    output_tokens: int = 0
    seconds: float = 0.0


@dataclass
class _Tool:
    server: str
    name: str
    description: str
    schema: dict[str, Any]


class MCPAgent:
    def __init__(
        self,
        specs: list[ServerSpec],
        llm: LLM,
        requires_approval: set[str] | None = None,
        approve: ApprovalFn = deny_all,
        system: str = DEFAULT_SYSTEM,
        max_steps: int = 8,
        on_event: Callable[[Event], None] | None = None,
    ):
        self.specs, self.llm = specs, llm
        self.requires_approval = requires_approval or set()
        self.approve, self.system, self.max_steps, self.on_event = approve, system, max_steps, on_event
        self.clients: dict[str, Client] = {}
        self.tools: dict[str, _Tool] = {}
        self._stack = AsyncExitStack()

    async def __aenter__(self) -> "MCPAgent":
        await self._stack.__aenter__()
        try:
            for spec in self.specs:
                params = StdioServerParameters(command=spec.command, args=spec.args, env=spec.env or None, cwd=ROOT)
                client = await self._stack.enter_async_context(Client(params))
                self.clients[spec.name] = client
                for t in (await client.list_tools()).tools:
                    self.tools[f"{spec.name}{SEP}{t.name}"] = _Tool(spec.name, t.name, t.description or "", t.input_schema)
        except BaseException:
            await self._stack.aclose()
            raise
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self._stack.aclose()

   
    def tool_definitions(self) -> list[dict]:
        """Tool list in the shape the Anthropic Messages API expects."""
        return [
            {"name": ns, "description": f"[{t.server}] {t.description}", "input_schema": t.schema}
            for ns, t in self.tools.items()
        ]

    async def list_resources(self) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {}
        for name, client in self.clients.items():
            res = await client.list_resources()
            tpl = await client.list_resource_templates()
            out[name] = [str(r.uri) for r in res.resources] + [t.uri_template for t in tpl.resource_templates]
        return out

    async def read_resource(self, server: str, uri: str) -> str:
        result = await self.clients[server].read_resource(uri)
        return "\n".join(getattr(c, "text", "") or "" for c in result.contents)

    async def get_prompt(self, server: str, name: str, arguments: dict[str, str] | None = None) -> str:
        result = await self.clients[server].get_prompt(name, arguments or {})
        return "\n".join(getattr(m.content, "text", "") for m in result.messages)

   
    async def run(self, task: str, history: list[dict] | None = None) -> RunResult:
        started = time.perf_counter()
        self._tin = self._tout = 0
        result = await self._run(task, history)
        result.input_tokens, result.output_tokens = self._tin, self._tout
        result.seconds = time.perf_counter() - started
        return result

    async def _run(self, task: str, history: list[dict] | None = None) -> RunResult:
        messages = list(history or []) + [{"role": "user", "content": task}]
        events: list[Event] = []
        defs = self.tool_definitions()

        for step in range(1, self.max_steps + 1):
            turn = await self.llm.next_turn(self.system, messages, defs)
            self._tin += turn.input_tokens
            self._tout += turn.output_tokens
            blocks: list[dict] = []
            if turn.text:
                blocks.append({"type": "text", "text": turn.text})
            blocks += [{"type": "tool_use", "id": c.id, "name": c.name, "input": c.args} for c in turn.tool_calls]
            if not blocks:  
                return RunResult("", messages, events, step)
            messages.append({"role": "assistant", "content": blocks})

            if not turn.tool_calls:
                self._emit(events, "assistant", turn.text)
                return RunResult(turn.text, messages, events, step)

            results = [await self._execute(call, events) for call in turn.tool_calls]
            messages.append({"role": "user", "content": results})

        note = f"Stopped after {self.max_steps} steps without a final answer."
        return RunResult(note, messages, events, self.max_steps)

    async def _execute(self, call: ToolCall, events: list[Event]) -> dict:
        self._emit(events, "tool_call", f"{call.name}({json.dumps(call.args)})")

        def result(text: str, is_error: bool) -> dict:
            self._emit(events, "tool_error" if is_error else "tool_result", text)
            return {"type": "tool_result", "tool_use_id": call.id, "content": text[:MAX_RESULT_CHARS], "is_error": is_error}

        tool = self.tools.get(call.name)
        if tool is None:
            return result(f"Unknown tool '{call.name}'. Available: {', '.join(sorted(self.tools))}", True)

        if call.name in self.requires_approval:
            ok = await self.approve(call.name, call.args)
            self._emit(events, "approval", f"{call.name}: {'approved' if ok else 'DENIED'}")
            if not ok:
                return result("The user denied this action. Do not retry it.", True)

        try:
            res = await self.clients[tool.server].call_tool(tool.name, call.args)
        except Exception as exc:  
            return result(f"{type(exc).__name__}: {exc}", True)
        text = "\n".join(c.text for c in res.content if getattr(c, "type", "") == "text") or "(no output)"
        return result(text, bool(res.is_error))

    def _emit(self, events: list[Event], kind: str, text: str) -> None:
        ev = Event(kind, text)
        events.append(ev)
        if self.on_event:
            self.on_event(ev)
