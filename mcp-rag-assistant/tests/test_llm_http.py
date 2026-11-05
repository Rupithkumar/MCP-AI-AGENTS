from __future__ import annotations

import json
import shutil
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from agent import MCPAgent, make_brain
from agent.check_llm import check
from agent.mcp_agent import ServerSpec

ROOT = Path(__file__).resolve().parent.parent


def completion(content=None, tool_calls=None) -> dict:
    message = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = tool_calls
    return {
        "id": "chatcmpl-test", "object": "chat.completion", "created": 0, "model": "mock",
        "choices": [{"index": 0, "message": message, "finish_reason": "tool_calls" if tool_calls else "stop"}],
        "usage": {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18},
    }


def tool_call(name: str, args: dict) -> list[dict]:
    return [{"id": "call_1", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}]


class MockModel(BaseHTTPRequestHandler):
    mode = "tools"
    requests: list[dict] = []

    def log_message(self, *args):  
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        MockModel.requests.append(body)
        if MockModel.mode == "unauthorized":
            return self._send(401, {"error": {"message": "Invalid API Key", "type": "invalid_request_error"}})
        if MockModel.mode == "no_tools":
            return self._send(200, completion(content="The answer is 42."))
        tool_results = [m for m in body["messages"] if m["role"] == "tool"]
        names = {t["function"]["name"] for t in body.get("tools", [])}
        if tool_results:  
            text = tool_results[-1]["content"]
            return self._send(200, completion(content=f"The result is {text}."))
        if "add_numbers" in names:
            return self._send(200, completion(tool_calls=tool_call("add_numbers", {"a": 17, "b": 25})))
        question = [m for m in body["messages"] if m["role"] == "user"][-1]["content"]
        return self._send(200, completion(tool_calls=tool_call("docs__search_docs", {"query": question, "k": 2})))

    def _send(self, status: int, payload: dict):
        data = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


@pytest.fixture
def mock_llm(monkeypatch):
    server = ThreadingHTTPServer(("127.0.0.1", 0), MockModel)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    MockModel.mode, MockModel.requests = "tools", []
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("NO_PROXY", "127.0.0.1")
    monkeypatch.setenv("no_proxy", "127.0.0.1")
    monkeypatch.setenv("LLM_BASE_URL", f"http://127.0.0.1:{server.server_port}/v1")
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("LLM_MODEL", "mock-model")
    yield MockModel
    server.shutdown()


async def test_check_llm_passes_with_a_working_tool_calling_model(mock_llm, capsys):
    assert await check() == 0
    out = capsys.readouterr().out
    assert "mock-model via http://127.0.0.1" in out and "called add_numbers" in out and "PASS" in out
    first = mock_llm.requests[0]
    assert first["model"] == "mock-model" and first["tools"][0]["function"]["name"] == "add_numbers"
    assert first["tools"][0]["function"]["parameters"]["required"] == ["a", "b"]
    roles = [m["role"] for m in mock_llm.requests[1]["messages"]]
    assert roles == ["system", "user", "assistant", "tool"]


async def test_check_llm_explains_a_rejected_key(mock_llm, capsys):
    mock_llm.mode = "unauthorized"
    assert await check() == 1
    out = capsys.readouterr().out
    assert "ERROR" in out and "key was rejected" in out


async def test_check_llm_flags_a_model_that_ignores_tools(mock_llm, capsys):
    mock_llm.mode = "no_tools"
    assert await check() == 1
    assert "without calling the tool" in capsys.readouterr().out


def test_check_llm_without_configuration_exits_with_instructions(monkeypatch):
    for var in ("ANTHROPIC_API_KEY", "LLM_MODEL"):
        monkeypatch.delenv(var, raising=False)
    with pytest.raises(SystemExit, match="No LLM configured"):
        make_brain("llm")


async def test_full_agent_through_the_real_sdk_with_citations_and_token_counts(mock_llm, tmp_path):
    shutil.copytree(ROOT / "docs", tmp_path / "docs")
    env = {"DOCS_DIR": str(tmp_path / "docs"), "RAG_DB": str(tmp_path / "r.db"), "EMBEDDER": ""}
    specs = [ServerSpec("docs", sys.executable, [str(ROOT / "servers" / "docs_server.py")], env)]
    llm, label = make_brain("llm")
    async with MCPAgent(specs, llm) as agent:
        result = await agent.run("Who introduced MCP and when?")
    assert label.startswith("mock-model")
    assert result.answer.startswith("The result is Found 2 passage(s)")  
    assert "[1] mcp-overview.md#0 |" in result.answer
    assert (result.input_tokens, result.output_tokens) == (22, 14)  
    assert result.steps == 2
