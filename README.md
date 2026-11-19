# MCP AI Agents

A small collection of projects exploring the Model Context Protocol (MCP), tool-using agents and retrieval-augmented generation (RAG).

## Projects

### MCP RAG Research Assistant
An assistant that answers questions about Markdown, text and PDF documents with citations. It includes an MCP document server, a SQLite index, hybrid retrieval options and an evaluation harness for measuring retrieval and answer quality.

[Read the full guide](mcp-rag-assistant/README.md)

### Basic MCP AI Agent
A smaller project demonstrating an agent working with two MCP servers.

[basic mcp ai agent folder](basic%20mcp%20ai%20agent/)

## What the RAG project demonstrates

- MCP tools, resources and prompts for document search
- Incremental indexing with SQLite storage
- BM25 retrieval, optional dense embeddings and reciprocal-rank fusion
- Page-aware PDF citations and citation checks
- Evaluation for retrieval quality, answer correctness, citation validity, abstention, latency and cost
- A no-LLM extractive baseline for local demos

## Try the demo

Use Python 3.10 or newer. From the repository root, move into the <code>mcp-rag-assistant</code> directory, install its requirements with <code>pip install -r requirements.txt</code>, then run <code>python -m agent.cli --demo</code>.

The demo runs without an API key. See the [project guide](mcp-rag-assistant/README.md) for interactive use, model configuration, evaluation commands, measured results and limitations.

## Repository layout

- <code>basic mcp ai agent/</code> — a basic agent connected to two servers
- <code>mcp-rag-assistant/</code> — document RAG assistant, MCP server, tests and evaluation tools
