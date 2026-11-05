from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent.brains import make_brain  
from agent.mcp_agent import DEFAULT_SYSTEM, Event, MCPAgent, deny_all, load_config  

from .metrics import ( 
    Question,
    extract_citations,
    facts_satisfied,
    is_abstention,
    load_questions,
    mean,
    seen_citations,
)


def score_answer(q: Question, answer: str, events: list[Event]) -> dict:
    """Score one finished run. Pure function: easy to test with hand-made events."""
    returned = seen_citations("\n".join(e.text for e in events if e.kind == "tool_result"))
    cites = extract_citations(answer)
    abstained = is_abstention(answer)
    row = {
        "id": q.id,
        "answerable": q.answerable,
        "used_search": any(e.kind == "tool_call" and "search_docs" in e.text for e in events),
        "abstained": abstained,
        "n_citations": len(cites),
        "n_bad_citations": sum(1 for c in cites if c not in returned),
        "tool_calls": sum(1 for e in events if e.kind == "tool_call"),
    }
    if q.answerable:
        row["facts_ok"] = facts_satisfied(answer, q.facts)
        row["cited_expected"] = any(c.split("#")[0] in q.sources for c in cites)
        row["correct"] = row["facts_ok"] and row["cited_expected"]
    else:
        row["correct"] = abstained  
    return row


def summarize(rows: list[dict]) -> dict:
    ans = [r for r in rows if r["answerable"]]
    unans = [r for r in rows if not r["answerable"]]
    cited = [r for r in rows if r["n_citations"]]
    ok = [r for r in rows if not r.get("error")]
    return {
        "questions": len(rows),
        "errors": len(rows) - len(ok),
        "answerable_accuracy": mean(r["correct"] for r in ans),
        "facts_rate": mean(r.get("facts_ok", False) for r in ans),
        "citation_rate": mean(r.get("cited_expected", False) for r in ans),
        "abstention_accuracy": mean(r["correct"] for r in unans),
        "search_rate": mean(r["used_search"] for r in rows),
        "fabricated_citation_rate": mean(r["n_bad_citations"] > 0 for r in cited),
        "mean_steps": mean(r["steps"] for r in ok),
        "mean_seconds": mean(r["seconds"] for r in ok),
        "mean_input_tokens": mean(r["input_tokens"] for r in ok),
        "mean_output_tokens": mean(r["output_tokens"] for r in ok),
    }


async def run_brain(
    kind: str, questions: list[Question], config: str, delay: float, retries: int,
    timeout: float = 120.0, partial: Path | None = None,
) -> tuple[str, list[dict]]:
    llm, label = make_brain(kind)
    specs, approvals = load_config(config)
    system = json.loads(Path(config).read_text()).get("system_prompt", DEFAULT_SYSTEM)
    rows: list[dict] = []
    async with MCPAgent(specs, llm, approvals, deny_all, system=system) as agent:
        for q in questions:
            result, error = None, None
            for attempt in range(retries + 1):
                try:
                    result, error = await asyncio.wait_for(agent.run(q.question), timeout), None
                    break
                except asyncio.TimeoutError:  
                    error = f"TimeoutError: no answer within {timeout:.0f}s"
                    break
                except Exception as exc:  
                    error = f"{type(exc).__name__}: {exc}"
                    await asyncio.sleep(2 ** (attempt + 1))
            if result is None:
                rows.append({"id": q.id, "answerable": q.answerable, "used_search": False, "abstained": False,
                             "n_citations": 0, "n_bad_citations": 0, "tool_calls": 0, "correct": False,
                             "steps": 0, "seconds": 0.0, "input_tokens": 0, "output_tokens": 0, "error": error, "answer": ""})
            else:
                row = score_answer(q, result.answer, result.events)
                row.update(steps=result.steps, seconds=round(result.seconds, 3), input_tokens=result.input_tokens,
                           output_tokens=result.output_tokens, answer=result.answer)
                rows.append(row)
            if partial:  
                partial.write_text(json.dumps({label: {"summary": summarize(rows), "rows": rows}}, indent=2))
            mark = "ok " if rows[-1]["correct"] else "BAD"
            print(f"  [{kind}] {q.id} {mark} {rows[-1]['seconds']:.1f}s" + (f"  ERROR {error[:80]}" if error else ""))
            if delay:
                await asyncio.sleep(delay)
    return label, rows


def to_markdown(results: dict[str, dict]) -> str:
    metrics = [
        ("answerable_accuracy", "Answerable questions correct (facts AND right source cited)"),
        ("facts_rate", "Required facts present in answer"),
        ("citation_rate", "Cited an expected source"),
        ("abstention_accuracy", "Unanswerable questions correctly refused"),
        ("fabricated_citation_rate", "Answers citing a passage the search never returned (lower is better)"),
        ("search_rate", "Searched before answering"),
        ("mean_steps", "Mean steps"),
        ("mean_seconds", "Mean seconds per question"),
        ("mean_input_tokens", "Mean input tokens"),
        ("mean_output_tokens", "Mean output tokens"),
        ("errors", "Questions that errored"),
    ]
    names = list(results)
    lines = ["| metric | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    for key, label in metrics:
        cells = []
        for name in names:
            v = results[name]["summary"][key]
            cells.append(f"{v:.2f}" if isinstance(v, float) else str(v))
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


async def amain(args: argparse.Namespace) -> None:
    questions = load_questions(args.dataset)
    if args.ids:
        wanted = [i.strip() for i in args.ids.split(",") if i.strip()]
        unknown = set(wanted) - {q.id for q in questions}
        if unknown:
            raise SystemExit(f"Unknown question ids: {', '.join(sorted(unknown))}")
        questions = [q for q in questions if q.id in wanted]
    if args.limit:
        questions = questions[: args.limit]
    results: dict[str, dict] = {}
    for kind in [b.strip() for b in args.brains.split(",") if b.strip()]:
        print(f"Running brain '{kind}' on {len(questions)} questions...")
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)
        partial = out_dir / "partial" / f"{kind}.json"
        partial.parent.mkdir(parents=True, exist_ok=True)
        label, rows = await run_brain(kind, questions, args.config, args.delay, args.retries, args.timeout, partial)
        results[label] = {"summary": summarize(rows), "rows": rows}

    table = to_markdown(results)
    print("\n" + table)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    (out / f"agent-{stamp}.json").write_text(json.dumps(results, indent=2))
    (out / f"agent-{stamp}.md").write_text(table + "\n")
    print(f"\nSaved to {out}")


def main() -> None:
    p = argparse.ArgumentParser(description="Agent benchmark")
    p.add_argument("--brains", default="baseline", help="comma list of: baseline, llm")
    p.add_argument("--limit", type=int, default=0, help="only the first N questions")
    p.add_argument("--delay", type=float, default=0.0, help="seconds to wait between questions")
    p.add_argument("--ids", default="", help="only these question ids, comma separated, e.g. h07,h08,u03")
    p.add_argument("--retries", type=int, default=2)
    p.add_argument("--timeout", type=float, default=120.0, help="give up on one question after this many seconds")
    p.add_argument("--dataset", default=str(ROOT / "evaluation" / "dataset.jsonl"))
    p.add_argument("--config", default=str(ROOT / "config.json"))
    p.add_argument("--out", default=str(ROOT / "evaluation" / "results"))
    asyncio.run(amain(p.parse_args()))


if __name__ == "__main__":
    main()