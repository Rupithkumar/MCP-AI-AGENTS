# MCP Agent Project

A from-scratch **Model Context Protocol (MCP)** project: two MCP servers, and an agent that
connects to both, discovers their tools, and runs a tool-use loop with Claude. No agent
framework, just the MCP SDK, so every moving part is visible.

```
 user task
    │
    ▼
 MCPAgent.run() ──loop──► LLM  (Claude via API, or offline planner)
    │  tool_use                │
    ▼                          │ tool_result
 approval gate ► MCP Client ► MCP server (stdio subprocess)
                     ├── notes  : SQLite notebook  (tools + resources + prompts)
                     └── utils  : calculator, clock, sandboxed file reader
```

## What it demonstrates

| MCP concept | Where |
|---|---|
| **Server primitives**: tools, resources (static and templated), prompts | `servers/notes_server.py` |
| **Stdio transport**, one client connection per server | `MCPAgent.__aenter__` |
| **Dynamic discovery** (`tools/list`) and namespacing (`server__tool`) | `agent/mcp_agent.py` |
| **The agent loop**: reason, call tool, observe, repeat, with a step cap | `MCPAgent.run` |
| **Error handling**: model-visible `ToolError` so the model can self-correct | both servers |
| **Human-in-the-loop** approval for destructive tools | `requires_approval` in `config.json` |
| **Security**: AST-based calculator (no `eval`), path-traversal-proof file reader | `servers/utils_server.py` |
| **Context hygiene**: tool output truncated before reaching the model | `MAX_RESULT_CHARS` |
| **Integration tests** against real server subprocesses | `tests/test_agent.py` |

## Run it

```bash
pip install -r requirements.txt

python -m agent.cli --demo --approval no   # scripted tour (destructive call is denied)
python -m agent.cli                        # interactive chat (asks before deleting)
python -m agent.cli -t "add note Hi: there then list notes"
pytest                                     # 12 tests
```

**Choosing the brain** (checked in this order):

1. `ANTHROPIC_API_KEY` set → Claude (default `claude-sonnet-5-5`, override with `--model`).
2. `LLM_MODEL` set → any OpenAI-compatible API (`LLM_BASE_URL`, `LLM_API_KEY`).
3. Otherwise → the offline planner.

With a real model you can give open-ended tasks like *"Remove the note about the agent
loop and tell me what's left."*

**Free option A, Ollama (runs on your PC, no key):**
```powershell
ollama pull llama3.1          # any model that supports tool calling
$env:LLM_MODEL="llama3.1"
$env:LLM_BASE_URL="http://localhost:11434/v1"
python -m agent.cli
```

**Free option B, a hosted free tier (Groq, Gemini, OpenRouter...)**: create a key on the
provider's site, then set the three variables (Groq's base URL is
`https://api.groq.com/openai/v1`; Gemini's is `https://generativelanguage.googleapis.com/v1beta/openai/`).
Model names and free-tier limits change, so check the provider's current list and pick one that supports tool calling.

**Without a key:** an `OfflineLLM` keyword planner stands in for the model so the whole MCP
pipeline still runs. It is a testing aid and not an intelligent agent; it only recognises a
few commands (chain them with `then`).

## Add your own server

1. Write a server with `MCPServer` (see `servers/utils_server.py`, about 80 lines).
2. Add an entry under `servers` in `config.json`.
3. Restart the agent. Its tools appear automatically, with no agent code changes. That is the point of MCP.

## Notes and limitations

- Built against **mcp 2.x**, where `FastMCP` became `MCPServer` and `ToolError` controls what the model sees.
- Tool calls within one model turn run sequentially, which keeps approval prompts ordered.
- The Claude and OpenAI-compatible paths are covered by tests using fake API clients (message format and
  `tool_result` wiring), but have **not** been run against a live API in this build. Try one once.
- Small local models are weaker at tool calling than Claude: they may pick the wrong tool or emit malformed
  arguments (the agent turns these into recoverable tool errors).
- Only the stdio transport is used. Remote servers would use Streamable HTTP with OAuth.

## Ideas to extend it

- Add a Streamable HTTP server and connect to it remotely.
- Use *sampling* so a server can ask the host's model for completions.
- Expose MCP resources to the model as context, not just tools.
- Add a web-search or GitHub server and let the agent chain across three servers.
