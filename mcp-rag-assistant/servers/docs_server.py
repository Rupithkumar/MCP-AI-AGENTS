from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  

from mcp.server.mcpserver import MCPServer  
from mcp.server.mcpserver.exceptions import ToolError  

from rag import Hit, Retriever, Store, embedder_from_env, index_folder 

DOCS_DIR = Path(os.environ.get("DOCS_DIR") or ROOT / "docs").resolve()
DB_PATH = Path(os.environ.get("RAG_DB") or ROOT / "workspace" / "rag.db")

mcp = MCPServer(
    "docs",
    instructions=(
        "Search a private document collection. Call search_docs before answering factual questions, "
        "answer only from the returned passages and cite them as [file#n]."
    ),
)

_state: dict = {}


def _retriever() -> Retriever:
    if "retriever" not in _state:
        store = Store(DB_PATH)
        embedder = embedder_from_env()
        if store.stats()["chunks"] == 0:
            index_folder(store, DOCS_DIR, embedder)
        _state["store"], _state["embedder"] = store, embedder
        _state["retriever"] = Retriever(store, embedder)
    return _state["retriever"]


def _format_hits(query: str, hits: list[Hit], method: str) -> str:
    if not hits:
        return f'No relevant passages found for "{query}". The documents may not cover this topic.'
    lines = [f'Found {len(hits)} passage(s) for "{query}" (retrieval: {method}). Cite sources as [file#n].', ""]
    for h in hits:
        c = h.chunk
        where = f"{c.section or '-'}" + (f" (p.{c.page})" if c.page else "")
        lines += [f"[{h.rank}] {c.cite} | {where} | score {h.score:.2f}", c.text, ""]
    return "\n".join(lines).rstrip()


@mcp.tool()
def index_documents() -> str:
    _retriever()  
    report = index_folder(_state["store"], DOCS_DIR, _state["embedder"])
    _state["retriever"].reload()
    return report.summary()


@mcp.tool()
def search_docs(query: str, k: int = 5) -> str:
    if not query.strip():
        raise ToolError("query must not be empty")
    retriever = _retriever()
    method = retriever.resolve_method("auto")
    return _format_hits(query, retriever.search(query, k=max(1, min(k, 10)), method=method), method)


@mcp.tool()
def get_chunk(citation: str, context: int = 1) -> str:
    doc, sep, number = citation.strip().strip("[]").rpartition("#")
    if not sep or not number.isdigit() or not doc:
        raise ToolError(f"citation must look like 'file.md#3', got {citation!r}")
    ord_, context = int(number), max(0, min(context, 3))
    rows = _retriever().store.get_chunks(doc, ord_ - context, ord_ + context)
    if not any(r.ord == ord_ for r in rows):
        raise ToolError(f"no passage {citation!r}; use search_docs to find valid citations")
    return "\n\n".join(f"{'>> ' if r.ord == ord_ else ''}[{r.cite}] {r.section or '-'}\n{r.text}" for r in rows)


@mcp.tool()
def list_documents() -> str:
    docs = _retriever().store.list_docs()
    if not docs:
        return "No documents are indexed."
    return "\n".join(f"{d['name']} ({d['n_chunks']} passages)" for d in docs)


@mcp.resource("rag://stats")
def stats() -> str:
    r = _retriever()
    s = r.store.stats()
    return f"documents={s['documents']}; chunks={s['chunks']}; retrieval={r.resolve_method('auto')}; embedding_models={s['embedding_models']}"


@mcp.resource("rag://documents")
def documents() -> str:
    """Names of all indexed documents."""
    return "\n".join(d["name"] for d in _retriever().store.list_docs())

@mcp.prompt()
def answer_with_citations(question: str) -> str:
    return (
        f"Answer this question using only the document collection: {question}\n"
        "Call search_docs first. Cite every claim as [file#n]. "
        "If the passages do not contain the answer, say you could not find it."
    )


if __name__ == "__main__":
    mcp.run("stdio")
