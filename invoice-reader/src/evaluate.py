from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict
from pathlib import Path

from .common import FIELDS, ROOT, field_correct
from .extract import Cascade, LLMExtractor, RuleExtractor, RuleRepairExtractor
from .generate import LEVELS, OUT, load_truth
from .ocr import read_text
from .validate import validate

RESULTS = ROOT / "results"
SYSTEMS = ("rules", "rules_repair", "llm", "cascade")


def get_text(rec: dict, folder: Path, cache: Path) -> str:
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / f"{rec['id']}.txt"
    if not path.exists():
        path.write_text(read_text(folder / rec["image"]), encoding="utf-8")
    return path.read_text(encoding="utf-8")


def make_llm(delay: float = 0.0):
    if not os.environ.get("LLM_MODEL"):
        raise SystemExit("Set LLM_MODEL (and LLM_BASE_URL, LLM_API_KEY) to use the llm / cascade systems.")
    from openai import OpenAI

    client = OpenAI(base_url=os.environ.get("LLM_BASE_URL"), api_key=os.environ.get("LLM_API_KEY") or "not-needed", max_retries=2)
    return LLMExtractor(client, os.environ["LLM_MODEL"])


def build_system(name: str, llm: LLMExtractor | None):
    if name == "rules":
        return RuleExtractor()
    if name == "rules_repair":
        return RuleRepairExtractor()
    if name == "llm":
        return llm
    if name == "cascade":
        return Cascade(llm)
    raise SystemExit(f"unknown system {name!r}; choose from {', '.join(SYSTEMS)}")


def run(system, truths: list[dict], texts: dict[str, str], delay: float = 0.0) -> dict:
    rows, started = [], time.perf_counter()
    for t in truths:
        res = system.extract(texts[t["id"]])
        record = {k: v for k, v in res.record.items() if k in FIELDS}
        problems = validate(res.record) + ([f"error: {res.error}"] if res.error else [])
        correct = {f: field_correct(f, t, record) for f in FIELDS}
        rows.append({"id": t["id"], "level": t["level"], "truth": {f: t[f] for f in FIELDS}, "pred": record, "correct": correct,
                     "doc_correct": all(correct.values()), "flagged": bool(problems), "problems": problems,
                     "llm_called": res.llm_called, "llm_error": bool(res.error)})
        if delay and res.llm_called:
            time.sleep(delay)
    return {"rows": rows, "seconds": time.perf_counter() - started}


def summarize(name: str, rows: list[dict], seconds: float) -> dict:
    def pct(a, b):
        return a / b if b else 0.0

    n = len(rows)
    wrong = [r for r in rows if not r["doc_correct"]]
    accepted = [r for r in rows if not r["flagged"]]
    out = {
        "name": name, "n": n, "doc_accuracy": pct(sum(r["doc_correct"] for r in rows), n),
        "field_accuracy": {f: pct(sum(r["correct"][f] for r in rows), n) for f in FIELDS},
        "by_level": {lv: {"n": len(sub), "doc_accuracy": pct(sum(r["doc_correct"] for r in sub), len(sub)),
                          "field_accuracy": {f: pct(sum(r["correct"][f] for r in sub), len(sub)) for f in FIELDS}}
                     for lv in LEVELS for sub in [[r for r in rows if r["level"] == lv]] if sub},
        "flagged_rate": pct(len(rows) - len(accepted), n),
        "silent_error_rate": pct(sum(not r["doc_correct"] for r in accepted), n),
        "accepted_accuracy": pct(sum(r["doc_correct"] for r in accepted), len(accepted)),
        "error_catch_rate": pct(sum(r["flagged"] for r in wrong), len(wrong)) if wrong else 1.0,
        "llm_calls": sum(r["llm_called"] for r in rows), "llm_errors": sum(r["llm_error"] for r in rows),
        "seconds": round(seconds, 2),
    }
    worst = defaultdict(int)
    for r in wrong:
        for f in FIELDS:
            if not r["correct"][f]:
                worst[f] += 1
    out["errors_by_field"] = dict(sorted(worst.items(), key=lambda kv: -kv[1]))
    return out


def save(name: str, summary: dict, rows: list[dict], out_dir: Path = RESULTS) -> None:
    (out_dir / "systems").mkdir(parents=True, exist_ok=True)
    (out_dir / "predictions").mkdir(parents=True, exist_ok=True)
    (out_dir / "systems" / f"{name}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out_dir / "predictions" / f"{name}.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate invoice extraction systems")
    p.add_argument("--systems", default="rules,rules_repair")
    p.add_argument("--folder", type=Path, default=OUT)
    p.add_argument("--results", type=Path, default=RESULTS, help="where to write results (use a separate folder per dataset)")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--delay", type=float, default=0.0, help="seconds to wait after each LLM call (free-tier rate limits)")
    a = p.parse_args()
    truths = load_truth(a.folder)
    if a.limit:
        truths = truths[: a.limit]
    print(f"reading {len(truths)} invoices with OCR (cached after the first run)...")
    texts = {t["id"]: get_text(t, a.folder, a.results / "ocr_cache") for t in truths}
    names = [s.strip() for s in a.systems.split(",") if s.strip()]
    llm = make_llm() if {"llm", "cascade"} & set(names) else None
    for name in names:
        out = run(build_system(name, llm), truths, texts, a.delay)
        s = summarize(name, out["rows"], out["seconds"])
        save(name, s, out["rows"], a.results)
        print(f"{name:13s} docs correct={s['doc_accuracy']:.1%}  flagged={s['flagged_rate']:.1%}  silent errors={s['silent_error_rate']:.1%}  "
              f"llm calls={s['llm_calls']}")


if __name__ == "__main__":
    main()
