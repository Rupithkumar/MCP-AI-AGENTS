from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent import ExtractiveLLM, MCPAgent, OpenAICompatLLM, load_config
from agent.baselines import NOT_FOUND
from agent.mcp_agent import ServerSpec, ToolCall, Turn

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def specs(tmp_path: Path) -> list[ServerSpec]:
    shutil.copytree(ROOT / "docs", tmp_path / "docs")
    env = {"DOCS_DIR": str(tmp_path / "docs"), "RAG_DB": str(tmp_path / "r.db"), "EMBEDDER": ""}
    return [ServerSpec("docs", sys.executable, [str(ROOT / "servers" / "docs_server.py")], env)]


async def test_baseline_answers_with_verbatim_passage_and_citation(specs):
    async with MCPAgent(specs, ExtractiveLLM()) as agent:
        assert {"docs__search_docs", "docs__get_chunk"} <= set(agent.tools)
        result = await agent.run("Which transport launches the server as a subprocess?")
    assert "[mcp-transports.md#0]" in result.answer and "stdio" in result.answer.lower()
    assert result.steps == 2


async def test_baseline_refuses_when_nothing_is_found(specs):
    async with MCPAgent(specs, ExtractiveLLM()) as agent:
        assert (await agent.run("What is the capital of Australia?")).answer == NOT_FOUND


class ScriptedLLM:
    def __init__(self, turns):
        self.turns = list(turns)

    async def next_turn(self, system, messages, tools):
        return self.turns.pop(0)


async def test_run_result_accumulates_tokens_and_time(specs):
    llm = ScriptedLLM([
        Turn(tool_calls=[ToolCall("1", "docs__list_documents", {})], input_tokens=100, output_tokens=7),
        Turn(text="done", input_tokens=250, output_tokens=3),
    ])
    async with MCPAgent(specs, llm) as agent:
        result = await agent.run("list them")
    assert (result.input_tokens, result.output_tokens) == (350, 10)
    assert result.seconds > 0 and result.answer == "done"


async def test_token_counters_reset_between_runs(specs):
    llm = ScriptedLLM([Turn(text="a", input_tokens=10), Turn(text="b", input_tokens=20)])
    async with MCPAgent(specs, llm) as agent:
        first, second = await agent.run("1"), await agent.run("2")
    assert (first.input_tokens, second.input_tokens) == (10, 20)


async def test_openai_adapter_reads_usage():
    resp = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="hi", tool_calls=None))],
        usage=SimpleNamespace(prompt_tokens=42, completion_tokens=5),
    )

    class FakeCompletions:
        async def create(self, **kwargs):
            return resp

    fake = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))
    turn = await OpenAICompatLLM("m", client=fake).next_turn("sys", [{"role": "user", "content": "x"}], [])
    assert (turn.text, turn.input_tokens, turn.output_tokens) == ("hi", 42, 5)


def test_config_expands_root_and_env_placeholders(tmp_path, monkeypatch):
    monkeypatch.setenv("MY_SETTING", "hashing")
    monkeypatch.delenv("UNSET_SETTING", raising=False)
    cfg = tmp_path / "c.json"
    cfg.write_text(json.dumps({"servers": {"s": {"command": "python", "args": ["${ROOT}/x.py"],
                    "env": {"A": "${env:MY_SETTING}", "B": "${env:UNSET_SETTING}"}}}}))
    (spec,), approvals = load_config(cfg)
    assert spec.env == {"A": "hashing", "B": ""} and spec.args == [f"{ROOT}/x.py"] and approvals == set()


def test_shipped_config_is_valid():
    (spec,), approvals = load_config(ROOT / "config.json")
    assert spec.name == "docs" and Path(spec.args[0]).exists()
    assert "system_prompt" in json.loads((ROOT / "config.json").read_text())
