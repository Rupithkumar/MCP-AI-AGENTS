from __future__ import annotations

from pathlib import Path

import pytest

from agent.mcp_agent import Event
from evaluation.agent_eval import score_answer, summarize
from evaluation.metrics import (
    Question,
    extract_citations,
    facts_satisfied,
    first_relevant_rank,
    has_phrase,
    hit_at_k,
    is_abstention,
    load_questions,
    reciprocal_rank,
    seen_citations,
)
from evaluation.retrieval_eval import evaluate_retrieval
from rag import Retriever, Store, index_folder

ROOT = Path(__file__).resolve().parent.parent

def test_has_phrase_matches_from_word_boundary_only():
    assert has_phrase("Text about chunking.", "chunk")
    assert not has_phrase("An unchunked text.", "chunk")
    assert has_phrase("Use 8–10 steps", "8-10")  
    assert has_phrase("RETRIEVAL-AUGMENTED generation", "retrieval-augmented")


def test_facts_need_every_group_and_any_phrase_inside_a_group():
    groups = [["tools"], ["resources", "files"]]
    assert facts_satisfied("tools and files", groups)
    assert not facts_satisfied("only tools", groups)
    assert facts_satisfied("anything", [])


@pytest.mark.parametrize(
    "text, expected",
    [
        ("I could not find this in the documents.", True),
        ("I couldn’t find that.", True),
        ("The documents do not mention Kubernetes.", True),
        ("There is no information about that.", True),
        ("Anthropic introduced MCP in November 2024.", False),
    ],
)
def test_abstention_detection(text, expected):
    assert is_abstention(text) is expected


def test_citation_extraction_handles_lists_and_ignores_other_brackets():
    answer = "A [a.md#1] and B [b.md#2, c.md#10]. See [1] and [note] too."
    assert extract_citations(answer) == ["a.md#1", "b.md#2", "c.md#10"]


def test_rank_metrics():
    rank = first_relevant_rank(["x.md", "y.md", "z.md"], ["y.md"])
    assert rank == 2 and reciprocal_rank(rank) == 0.5
    assert hit_at_k(rank, 1) is False and hit_at_k(rank, 3) is True
    assert first_relevant_rank(["x.md"], ["q.md"]) is None and reciprocal_rank(None) == 0.0



Q = Question("q", "Who introduced MCP?", ["mcp-overview.md"], [["anthropic"], ["2024"]], True)
U = Question("u", "Capital of Australia?", [], [], False)


def events(*cites: str) -> list[Event]:
    result = "Found passages.\n\n" + "\n".join(f"[{i}] {c} | Section | score 1.00\ntext" for i, c in enumerate(cites, 1))
    return [Event("tool_call", 'docs__search_docs({"query": "x"})'), Event("tool_result", result)]


def test_correct_grounded_answer():
    row = score_answer(Q, "Anthropic introduced it in 2024 [mcp-overview.md#0].", events("mcp-overview.md#0"))
    assert row["correct"] and row["facts_ok"] and row["cited_expected"] and row["n_bad_citations"] == 0
    assert row["used_search"]


def test_fabricated_citation_is_flagged_and_wrong_source_is_not_correct():
    row = score_answer(Q, "Anthropic, 2024 [mcp-overview.md#9].", events("mcp-overview.md#0"))
    assert row["n_bad_citations"] == 1  # #9 was never returned by the search
    row = score_answer(Q, "Anthropic, 2024 [agent-loop.md#1].", events("agent-loop.md#1"))
    assert row["facts_ok"] and not row["cited_expected"] and not row["correct"]


def test_unanswerable_is_correct_only_when_the_assistant_refuses():
    assert score_answer(U, "I could not find this in the documents.", events())["correct"]
    assert not score_answer(U, "Canberra [mcp-overview.md#0].", events("mcp-overview.md#0"))["correct"]


def test_summary_rates():
    rows = [
        {"answerable": True, "correct": True, "facts_ok": True, "cited_expected": True, "used_search": True,
         "n_citations": 1, "n_bad_citations": 0, "steps": 2, "seconds": 1.0, "input_tokens": 100, "output_tokens": 10},
        {"answerable": True, "correct": False, "facts_ok": True, "cited_expected": False, "used_search": True,
         "n_citations": 1, "n_bad_citations": 1, "steps": 4, "seconds": 3.0, "input_tokens": 300, "output_tokens": 30},
        {"answerable": False, "correct": True, "used_search": False,
         "n_citations": 0, "n_bad_citations": 0, "steps": 1, "seconds": 2.0, "input_tokens": 50, "output_tokens": 5},
    ]
    s = summarize(rows)
    assert s["answerable_accuracy"] == 0.5 and s["facts_rate"] == 1.0 and s["citation_rate"] == 0.5
    assert s["abstention_accuracy"] == 1.0 and s["fabricated_citation_rate"] == 0.5
    assert s["mean_steps"] == pytest.approx(7 / 3) and s["mean_input_tokens"] == 150


@pytest.fixture(scope="module")
def questions():
    return load_questions(ROOT / "evaluation" / "dataset.jsonl")


def test_dataset_is_well_formed(questions):
    assert len({q.id for q in questions}) == len(questions)
    assert sum(not q.answerable for q in questions) >= 5 and sum(q.tier == "hard" for q in questions) >= 10
    docs = {p.name for p in (ROOT / "docs").glob("*.md")}
    for q in questions:
        if q.answerable:
            assert q.sources and q.facts, q.id
            assert set(q.sources) <= docs, f"{q.id}: unknown source"
            assert all(group for group in q.facts), f"{q.id}: empty fact group"
        else:
            assert not q.sources and not q.facts, q.id


def test_every_required_fact_really_appears_in_its_source_documents(questions):
    """Guards against dataset typos: a fact the documents never state could never be scored correct."""
    for q in questions:
        if q.answerable:
            text = "\n".join((ROOT / "docs" / s).read_text(encoding="utf-8") for s in q.sources)
            assert facts_satisfied(text, q.facts), f"{q.id}: facts {q.facts} not found in {q.sources}"


def test_retrieval_benchmark_regression_guard(questions, tmp_path):
    store = Store(tmp_path / "eval.db")
    index_folder(store, ROOT / "docs")
    result = evaluate_retrieval(Retriever(store), questions, ["bm25"])
    easy = result["methods"]["bm25"]["by_tier"]["easy"]
    allq = result["methods"]["bm25"]["by_tier"]["all"]
    assert easy["hit@5"] >= 0.9 and allq["mrr"] >= 0.8  
    assert set(result["methods"]["bm25"]["by_tier"]) == {"easy", "hard", "all"}
