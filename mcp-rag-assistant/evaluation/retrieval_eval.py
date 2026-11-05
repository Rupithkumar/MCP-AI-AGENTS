"""Benchmark the retriever alone (no LLM, deterministic, runs in seconds).

    python -m evaluation.retrieval_eval

For every answerable question we look at the ranked chunks and report:
    hit@k        a chunk from an expected source document is in the top k
    MRR          mean of 1/rank of the first such chunk
    facts@k      the top-k chunks together contain every required fact (a stricter check than hit@k,
                 because the right document can still be the wrong passage)
Questions come in two tiers: 'easy' (shares words with the documents) and 'hard' (paraphrased, little
word overlap). For unanswerable questions we report how often retrieval returns nothing, which is
the signal an assistant can use to refuse.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rag import Retriever, Store, embedder_from_env, index_folder  # noqa: E402

from .metrics import (  # noqa: E402
    Question,
    facts_satisfied,
    first_relevant_rank,
    hit_at_k,
    load_questions,
    mean,
    reciprocal_rank,
)

KS = (1, 3, 5)
TIERS = ("easy", "hard", "all")


def _summarize(rows: list[dict]) -> dict:
    summary = {f"hit@{k}": mean(r[f"hit@{k}"] for r in rows) for k in KS}
    summary["mrr"] = mean(r["reciprocal_rank"] for r in rows)
    summary.update({f"facts@{k}": mean(r[f"facts@{k}"] for r in rows) for k in KS})
    summary["n"] = len(rows)
    return summary


def evaluate_retrieval(retriever: Retriever, questions: list[Question], methods: list[str]) -> dict:
    answerable = [q for q in questions if q.answerable]
    unanswerable = [q for q in questions if not q.answerable]
    out: dict = {"n_answerable": len(answerable), "n_unanswerable": len(unanswerable), "methods": {}}
    for method in methods:
        rows = []
        for q in answerable:
            hits = retriever.search(q.question, k=max(KS), method=method)
            rank = first_relevant_rank([h.chunk.doc for h in hits], q.sources)
            row = {"id": q.id, "tier": q.tier, "rank": rank, "reciprocal_rank": reciprocal_rank(rank)}
            for k in KS:
                row[f"hit@{k}"] = hit_at_k(rank, k)
                row[f"facts@{k}"] = facts_satisfied("\n".join(h.chunk.text for h in hits[:k]), q.facts)
            rows.append(row)
        empty = [len(retriever.search(q.question, k=max(KS), method=method)) == 0 for q in unanswerable]
        by_tier = {t: _summarize([r for r in rows if t == "all" or r["tier"] == t]) for t in TIERS}
        out["methods"][method] = {
            "by_tier": {t: s for t, s in by_tier.items() if s["n"]},
            "empty_on_unanswerable": mean(empty),
            "per_question": rows,
        }
    return out


def to_markdown(result: dict) -> str:
    cols = ["hit@1", "hit@3", "hit@5", "mrr", "facts@3", "facts@5"]
    lines = ["| method | tier | n | " + " | ".join(cols) + " |", "|---|---|---|" + "---|" * len(cols)]
    for method, data in result["methods"].items():
        for tier, s in data["by_tier"].items():
            lines.append(f"| {method} | {tier} | {s['n']} | " + " | ".join(f"{s[c]:.2f}" for c in cols) + " |")
    lines.append("")
    for method, data in result["methods"].items():
        lines.append(f"- {method}: returned nothing for {data['empty_on_unanswerable']:.0%} of the "
                     f"{result['n_unanswerable']} unanswerable questions")
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description="Retrieval benchmark")
    p.add_argument("--docs", default=str(ROOT / "docs"))
    p.add_argument("--db", default=str(ROOT / "workspace" / "rag_eval.db"))
    p.add_argument("--dataset", default=str(ROOT / "evaluation" / "dataset.jsonl"))
    p.add_argument("--out", default=str(ROOT / "evaluation" / "results"))
    args = p.parse_args()

    embedder = embedder_from_env()  # dense / hybrid are only evaluated when EMBEDDER is set
    store = Store(args.db)
    print(index_folder(store, args.docs, embedder).summary())
    retriever = Retriever(store, embedder)
    methods = ["bm25"] + (["dense", "hybrid"] if retriever.has_dense else [])
    if not retriever.has_dense:
        print("(dense and hybrid skipped: set EMBEDDER to evaluate them)")

    result = evaluate_retrieval(retriever, load_questions(args.dataset), methods)
    print("\n" + to_markdown(result))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    (out / f"retrieval-{stamp}.json").write_text(json.dumps(result, indent=2))
    (out / f"retrieval-{stamp}.md").write_text(to_markdown(result) + "\n")
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
