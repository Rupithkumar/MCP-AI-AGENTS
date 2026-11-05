from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pytest
from mcp import Client
from mcp.client.stdio import StdioServerParameters

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def docs_dir(tmp_path: Path) -> Path:
    target = tmp_path / "docs"
    shutil.copytree(ROOT / "docs", target)
    return target


def params(docs_dir: Path, db: Path, **extra: str) -> StdioServerParameters:
    env = {"DOCS_DIR": str(docs_dir), "RAG_DB": str(db), "EMBEDDER": "", **extra}
    return StdioServerParameters(command=sys.executable, args=[str(ROOT / "servers" / "docs_server.py")], env=env)


def text(result) -> str:
    return "\n".join(c.text for c in result.content)


async def test_server_exposes_tools_resources_and_prompt(docs_dir, tmp_path):
    async with Client(params(docs_dir, tmp_path / "r.db")) as c:
        assert {t.name for t in (await c.list_tools()).tools} == {
            "index_documents", "search_docs", "get_chunk", "list_documents",
        }
        assert {str(r.uri) for r in (await c.list_resources()).resources} == {"rag://stats", "rag://documents"}
        assert [p.name for p in (await c.list_prompts()).prompts] == ["answer_with_citations"]
        prompt = await c.get_prompt("answer_with_citations", {"question": "Why?"})
        assert "search_docs" in prompt.messages[0].content.text


async def test_search_returns_cited_passages_and_auto_indexes(docs_dir, tmp_path):
    async with Client(params(docs_dir, tmp_path / "r.db")) as c:  
        result = await c.call_tool("search_docs", {"query": "Who introduced MCP and when?", "k": 3})
        assert not result.is_error
        out = text(result)
        assert "(retrieval: bm25)" in out and "[1] mcp-overview.md#0 |" in out
        assert "November 2024" in out
        stats = (await c.read_resource("rag://stats")).contents[0].text
        assert "documents=9" in stats and "retrieval=bm25" in stats


async def test_unrelated_query_returns_a_clear_no_results_message(docs_dir, tmp_path):
    async with Client(params(docs_dir, tmp_path / "r.db")) as c:
        out = text(await c.call_tool("search_docs", {"query": "capital of Australia"}))
        assert out.startswith("No relevant passages found")


async def test_get_chunk_with_context_and_error_cases(docs_dir, tmp_path):
    async with Client(params(docs_dir, tmp_path / "r.db")) as c:
        ok = text(await c.call_tool("get_chunk", {"citation": "mcp-security.md#1", "context": 1}))
        assert ">> [mcp-security.md#1]" in ok and "[mcp-security.md#0]" in ok and "[mcp-security.md#2]" in ok
        assert "path traversal" in ok.lower()
        bracketed = await c.call_tool("get_chunk", {"citation": "[mcp-security.md#1]", "context": 0})
        assert not bracketed.is_error
        for bad in ("nonsense", "mcp-security.md#999", "#3", "mcp-security.md#x"):
            result = await c.call_tool("get_chunk", {"citation": bad})
            assert result.is_error, bad


async def test_empty_query_is_a_readable_tool_error(docs_dir, tmp_path):
    async with Client(params(docs_dir, tmp_path / "r.db")) as c:
        result = await c.call_tool("search_docs", {"query": "   "})
        assert result.is_error and "must not be empty" in text(result)


async def test_reindex_picks_up_new_files_and_no_tool_takes_a_path(docs_dir, tmp_path):
    async with Client(params(docs_dir, tmp_path / "r.db")) as c:
        assert "no relevant" in text(await c.call_tool("search_docs", {"query": "kubernetes"})).lower()
        (docs_dir / "ops.md").write_text("# Ops\n\nDeploy the server to Kubernetes with a Helm chart.")
        report = text(await c.call_tool("index_documents", {}))
        assert "1 added" in report and "9 unchanged" in report
        assert "ops.md#0" in text(await c.call_tool("search_docs", {"query": "kubernetes"}))
        assert "ops.md (1 passages)" in text(await c.call_tool("list_documents", {}))
        for tool in (await c.list_tools()).tools:
            assert not {"path", "folder", "directory"} & set(tool.input_schema.get("properties", {}))


async def test_hashing_embedder_enables_hybrid_through_the_environment(docs_dir, tmp_path):
    async with Client(params(docs_dir, tmp_path / "r.db", EMBEDDER="hashing")) as c:
        out = text(await c.call_tool("search_docs", {"query": "reciprocal rank fusion constant"}))
        assert "(retrieval: hybrid)" in out and "retrieval-methods.md" in out
