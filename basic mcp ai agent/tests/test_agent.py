"""Integration tests: every test spawns the real MCP servers over stdio."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent import AnthropicLLM, MCPAgent, OfflineLLM, OpenAICompatLLM, approve_all, deny_all
from agent.mcp_agent import ROOT, ServerSpec, Turn, ToolCall

SERVERS = ROOT / "servers"
NEEDS_APPROVAL = {"notes__delete_note"}


@pytest.fixture
def specs(tmp_path: Path) -> list[ServerSpec]:
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    (sandbox / "hello.txt").write_text("hello sandbox")
    return [
        ServerSpec("notes", sys.executable, [str(SERVERS / "notes_server.py")], {"NOTES_DB": str(tmp_path / "n.db")}),
        ServerSpec("utils", sys.executable, [str(SERVERS / "utils_server.py")], {"SANDBOX_DIR": str(sandbox)}),
    ]


class ScriptedLLM:
    """Replays a fixed list of turns; records what it was sent."""

    def __init__(self, turns: list[Turn]):
        self.turns, self.seen = list(turns), []

    async def next_turn(self, system, messages, tools):
        self.seen.append((list(messages), tools))
        return self.turns.pop(0) if len(self.turns) > 1 else self.turns[0]


# ------------------------------------------------------------- discovery
async def test_discovers_namespaced_tools_from_both_servers(specs):
    async with MCPAgent(specs, OfflineLLM()) as agent:
        assert set(agent.clients) == {"notes", "utils"}
        assert {"notes__add_note", "notes__delete_note", "utils__calculate", "utils__read_text_file"} <= set(agent.tools)
        for d in agent.tool_definitions():
            assert d["input_schema"]["type"] == "object"
            assert d["description"].startswith("[")


async def test_resources_and_prompts(specs):
    async with MCPAgent(specs, OfflineLLM()) as agent:
        resources = await agent.list_resources()
        assert "notes://stats" in resources["notes"] and "notes://note/{note_id}" in resources["notes"]
        await agent.run("add note Alpha: first")
        assert "total_notes=1" in await agent.read_resource("notes", "notes://stats")
        assert "Alpha" in await agent.read_resource("notes", "notes://note/1")
        prompt = await agent.get_prompt("notes", "summarize_notes", {"topic": "agents"})
        assert "agents" in prompt


# ---------------------------------------------------------- agent loop
async def test_add_then_search_in_one_run(specs):
    async with MCPAgent(specs, OfflineLLM()) as agent:
        await agent.run("add note Groceries: milk and eggs")
        result = await agent.run("search notes for milk then calculate 6*7")
        assert "Groceries" in result.answer and "42" in result.answer
        assert [e.kind for e in result.events].count("tool_call") == 2


async def test_unknown_tool_is_reported_and_loop_continues(specs):
    llm = ScriptedLLM([Turn(tool_calls=[ToolCall("1", "nope__nothing", {})]), Turn(text="recovered")])
    async with MCPAgent(specs, llm) as agent:
        result = await agent.run("go")
    assert result.answer == "recovered"
    tool_result = llm.seen[1][0][-1]["content"][0]
    assert tool_result["is_error"] and "Unknown tool" in tool_result["content"]


async def test_max_steps_stops_a_runaway_loop(specs):
    llm = ScriptedLLM([Turn(tool_calls=[ToolCall("1", "utils__current_time", {})])])  # never answers
    async with MCPAgent(specs, llm, max_steps=3) as agent:
        result = await agent.run("loop forever")
    assert result.steps == 3 and result.answer.startswith("Stopped after 3 steps")


# ------------------------------------------------------- human approval
async def test_destructive_tool_denied_keeps_the_note(specs):
    async with MCPAgent(specs, OfflineLLM(), NEEDS_APPROVAL, deny_all) as agent:
        await agent.run("add note Keep me: important")
        result = await agent.run("delete note 1")
        assert any(e.kind == "approval" and "DENIED" in e.text for e in result.events)
        assert "Keep me" in (await agent.run("list notes")).answer


async def test_destructive_tool_approved_deletes_the_note(specs):
    async with MCPAgent(specs, OfflineLLM(), NEEDS_APPROVAL, approve_all) as agent:
        await agent.run("add note Temp: bye")
        result = await agent.run("delete note 1")
        assert "Deleted note #1" in result.answer
        assert "No notes yet" in (await agent.run("list notes")).answer


# ------------------------------------------------------------- security
async def test_calculator_never_executes_code(specs):
    async with MCPAgent(specs, OfflineLLM()) as agent:
        result = await agent.run("calculate __import__('os').system('echo pwned')")
        assert any(e.kind == "tool_error" for e in result.events)
        assert "pwned" not in result.answer
        assert "division by zero" in (await agent.run("calculate 1/0")).answer


async def test_file_reader_is_confined_to_sandbox(specs):
    async with MCPAgent(specs, OfflineLLM()) as agent:
        assert "hello sandbox" in (await agent.run("read file hello.txt")).answer
        escaped = await agent.run("read file ../../../../etc/passwd")
        assert any(e.kind == "tool_error" for e in escaped.events)
        assert "root:" not in escaped.answer


# ---------------------------------------- Claude adapter (fake API client)
def _resp(*blocks):
    return SimpleNamespace(content=list(blocks))


async def test_anthropic_adapter_round_trip(specs):
    """Verifies message/tool_result wiring with a fake client. Does NOT call the real API."""
    calls = []

    class FakeMessages:
        responses = [
            _resp(
                SimpleNamespace(type="text", text="Let me calculate."),
                SimpleNamespace(type="tool_use", id="tu_1", name="utils__calculate", input={"expression": "6*7"}),
            ),
            _resp(SimpleNamespace(type="text", text="The answer is 42.")),
        ]

        async def create(self, **kwargs):
            calls.append(kwargs)
            return self.responses.pop(0)

    fake_client = SimpleNamespace(messages=FakeMessages())
    async with MCPAgent(specs, AnthropicLLM(client=fake_client)) as agent:
        result = await agent.run("what is 6*7?")

    assert result.answer == "The answer is 42."
    assert {t["name"] for t in calls[0]["tools"]} >= {"utils__calculate", "notes__add_note"}
    second_call_messages = calls[1]["messages"]
    assert second_call_messages[1]["content"][1]["type"] == "tool_use"
    tool_result = second_call_messages[2]["content"][0]
    assert tool_result == {"type": "tool_result", "tool_use_id": "tu_1", "content": "42", "is_error": False}


# ------------------------------- OpenAI-compatible adapter (fake client)
async def test_openai_compat_adapter_round_trip(specs):
    """Verifies format conversion (tool_calls / tool role) with a fake client, not a live API."""
    calls = []

    def completion(content=None, tool_calls=None):
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content, tool_calls=tool_calls))])

    tc = SimpleNamespace(id="call_9", function=SimpleNamespace(name="utils__calculate", arguments='{"expression": "6*7"}'))
    responses = [completion(tool_calls=[tc]), completion(content="It is 42.")]

    class FakeCompletions:
        async def create(self, **kwargs):
            calls.append(kwargs)
            return responses.pop(0)

    fake = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))
    async with MCPAgent(specs, OpenAICompatLLM("test-model", client=fake)) as agent:
        result = await agent.run("what is 6*7?")

    assert result.answer == "It is 42."
    first_tools = calls[0]["tools"]
    assert first_tools[0]["type"] == "function" and "parameters" in first_tools[0]["function"]
    msgs = calls[1]["messages"]
    assert msgs[0]["role"] == "system"
    assert msgs[2]["tool_calls"][0] == {
        "id": "call_9", "type": "function",
        "function": {"name": "utils__calculate", "arguments": '{"expression": "6*7"}'},
    }
    assert msgs[3] == {"role": "tool", "tool_call_id": "call_9", "content": "42"}


async def test_openai_compat_survives_broken_tool_json(specs):
    """A small local model emitting invalid JSON must produce a recoverable tool error, not a crash."""
    bad = SimpleNamespace(id="", function=SimpleNamespace(name="utils__calculate", arguments="{not json"))
    seq = [
        SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=None, tool_calls=[bad]))]),
        SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="sorry, retrying", tool_calls=None))]),
    ]

    class FakeCompletions:
        async def create(self, **kwargs):
            return seq.pop(0)

    fake = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))
    async with MCPAgent(specs, OpenAICompatLLM("m", client=fake)) as agent:
        result = await agent.run("go")
    assert result.answer == "sorry, retrying"
    assert any(e.kind == "tool_error" for e in result.events)
