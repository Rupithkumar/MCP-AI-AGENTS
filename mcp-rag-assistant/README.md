# MCP RAG Research Assistant + Evaluation Harness

An agent that answers questions about **your own documents** (`.md`, `.txt`, `.pdf`) with **citations**,
built on the Model Context Protocol, plus a harness that **measures** whether it is any good.

```
 question
    │
    ▼
 MCPAgent ──loop──► brain  (Claude / any OpenAI-compatible LLM / no-LLM baseline)
    │ tool calls
    ▼
 MCP client ──stdio──► docs MCP server ──► retriever ──► SQLite index
                         ├ search_docs      BM25  (+ dense embeddings, optional)
                         ├ get_chunk        fused with Reciprocal Rank Fusion
                         ├ index_documents
                         └ list_documents   resources: rag://stats, rag://documents · prompt: answer_with_citations
```

## What it demonstrates

| Area | Where |
|---|---|
| MCP server with tools, resources and a prompt; no tool accepts a filesystem path | `servers/docs_server.py` |
| Heading-aware, sentence-based chunking with overlap; chunks never span a PDF page, so page citations are exact | `rag/chunker.py` |
| Incremental indexing by file hash (add / update / remove), SQLite storage | `rag/indexer.py`, `rag/store.py` |
| BM25 from scratch, optional dense embeddings, hybrid fusion (RRF) | `rag/bm25.py`, `rag/retriever.py` |
| Agent loop that tracks steps, seconds and tokens per question | `agent/mcp_agent.py` |
| A no-LLM **baseline** brain to compare real models against | `agent/baselines.py` |
| Retrieval benchmark: hit@k, MRR, facts@k, easy vs hard questions | `evaluation/retrieval_eval.py` |
| Agent benchmark: correctness, citation validity, **fabricated-citation rate**, **abstention**, cost, latency | `evaluation/agent_eval.py` |
| 57 tests, including a check that every fact in the dataset really appears in its source docs | `tests/` |

## Setup (Windows PowerShell, from this folder)

```powershell
pip install -r requirements.txt
python -m pytest -q                      # expect: 56 passed, 1 skipped
```

## Run it

```powershell
python -m agent.cli --demo               # scripted questions (works with no key)
python -m agent.cli                      # interactive: ask anything; /docs /reindex /quit
python -m agent.cli -t "Who introduced MCP and when?"
```

Put your own files in `docs\` and type `/reindex` (only changed files are re-processed).
Without an LLM configured, the **extractive baseline** answers: it returns the best passage verbatim
with its citation. To use a real model, set variables in the same window, for example Groq:

```powershell
$env:LLM_API_KEY="your-key"
$env:LLM_BASE_URL="https://api.groq.com/openai/v1"
$env:LLM_MODEL="<a tool-calling model from your Groq models list>"
python -m agent.cli
```

Claude works the same way via `ANTHROPIC_API_KEY`. The first line printed (`[brain] ...`) tells you which brain is active.

### Connect a free model and test it

1. Create a free key at console.groq.com (API Keys). Free-tier limits change, so check the site.
2. Set the three variables above, then list the models your key can use and pick a larger chat model that supports tool calling:
   ```powershell
   python -c "import os; from openai import OpenAI; c=OpenAI(base_url=os.environ['LLM_BASE_URL'], api_key=os.environ['LLM_API_KEY']); print('\n'.join(sorted(m.id for m in c.models.list().data)))"
   ```
3. Check the connection (about 10 seconds, two tiny model calls):
   ```powershell
   python -m agent.check_llm
   ```
   `PASS` means the key, URL, model name and tool calling all work. Any other result prints a hint.
4. Use it: `python -m agent.cli`, then benchmark it:
   ```powershell
   python -m evaluation.agent_eval --brains llm --limit 5 --delay 2
   python -m evaluation.agent_eval --brains baseline,llm --delay 2
   ```

**No key, runs locally (Ollama):** install Ollama, run `ollama pull <a tool-calling model>`, then set
`$env:LLM_BASE_URL="http://localhost:11434/v1"`, `$env:LLM_API_KEY="ollama"` and `$env:LLM_MODEL="<that model>"`.

`$env:` settings last only for the current window. Never paste keys into files or commit them.

## Evaluate it

```powershell
python -m evaluation.retrieval_eval                       # retriever only: deterministic, seconds
python -m evaluation.agent_eval --brains baseline         # free
python -m evaluation.agent_eval --brains baseline,llm --delay 2   # compare with your LLM (mind rate limits)
```

Results are printed and saved to `evaluation\results\` (JSON + Markdown).
The dataset (`evaluation\dataset.jsonl`) has 28 easy and 12 hard (paraphrased) answerable questions and
6 unanswerable ones. Each answerable question lists its source documents and "fact groups" the answer must contain.

### Results measured while building this (BM25 only, 9 documents, 42 chunks)

| method | tier | n | hit@1 | hit@3 | hit@5 | mrr | facts@3 | facts@5 |
|---|---|---|---|---|---|---|---|---|
| bm25 | easy | 28 | 1.00 | 1.00 | 1.00 | 1.00 | 0.96 | 1.00 |
| bm25 | hard | 12 | 0.75 | 0.92 | 0.92 | 0.82 | 0.83 | 0.92 |
| bm25 | all | 40 | 0.93 | 0.97 | 0.97 | 0.95 | 0.93 | 0.97 |

Retrieval returned nothing for only 1 of 6 unanswerable questions.

Extractive baseline through the full agent: **80%** of answerable questions correct (facts present and the right
source cited), **0 fabricated citations**, but it refused only **1 of 6** unanswerable questions. Reading this:

- Keyword search is excellent when the question reuses the documents' words and drops on paraphrases (hard tier).
  That gap is exactly where dense or hybrid retrieval should help.
- Plain retrieval cannot tell "the top passage is on-topic but doesn't answer this" from a real answer, so it
  answers almost everything. Refusing correctly needs judgement, which is what a real LLM brain adds. Run the
  `baseline,llm` comparison to see whether yours does.

## Optional: dense and hybrid retrieval

```powershell
pip install sentence-transformers
$env:EMBEDDER="st:all-MiniLM-L6-v2"      # or openai:<model> with EMBED_BASE_URL / EMBED_API_KEY (e.g. Ollama)
python -m evaluation.retrieval_eval      # now also reports dense and hybrid rows
```

## Honest limitations

- **Not run against a live LLM or real embedding model.** Those paths are covered by fake-client tests, and the
  OpenAI-compatible LLM path is also tested with the real `openai` SDK over HTTP against a local mock server
  (`tests/test_llm_http.py`), which checks the protocol but not any real model's behaviour. Run `agent_eval --brains baseline,llm` once with your key to get real numbers, and paste any error.
- `EMBEDDER=hashing` exists for tests only: it is lexical, not semantic.
- `dense_min=0.25` (the cosine cut-off that lets dense search return nothing) is a starting point, not calibrated.
- The eval's fact checks are keyword-based: cheap and repeatable, but a correct answer in unexpected words can fail.
  The dataset was written together with the documents, so treat the numbers as a regression tool for *this* project
  and not as a general benchmark. Add your own documents and questions for a meaningful score.
- Vector search is a brute-force scan, fine for tens of thousands of chunks. Beyond that, move to FAISS, Chroma or pgvector.
- Built against **mcp 2.x**. Code is checked against the Python 3.10 grammar and tested on 3.13.

## Ideas to extend it

- A cross-encoder **reranker** on the top 20 candidates, then add it as a fourth row in the benchmark.
- **LLM-as-judge** scoring for faithfulness, spot-checked by hand against the keyword metric.
- A **Streamable HTTP** deployment of the docs server with an API key.
- Parent-child retrieval: search small chunks, return the whole section.
