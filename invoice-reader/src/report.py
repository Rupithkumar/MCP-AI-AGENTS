from __future__ import annotations

import argparse
import json
from pathlib import Path

from .common import FIELDS, ROOT

ORDER = ["rules", "rules_repair", "llm", "cascade"]


def load(results: Path) -> list[dict]:
    found = {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in (results / "systems").glob("*.json")}
    return [found[n] for n in ORDER if n in found]


def build(results: Path) -> str:
    rows = load(results)
    if not rows:
        raise SystemExit(f"No results in {results}. Run: python -m src.evaluate")
    md = ["# Invoice reader: results", "",
          f"{rows[0]['n']} synthetic invoices (clean, mild and harsh scans). Real invoices will score lower: see Limitations.", "",
          "## Headline", "",
          "| System | Docs fully correct | Flagged for review | Silent errors | Accuracy of accepted docs | Errors caught | LLM calls |",
          "|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['name']} | {r['doc_accuracy']:.1%} | {r['flagged_rate']:.1%} | {r['silent_error_rate']:.1%} | "
                  f"{r['accepted_accuracy']:.1%} | {r['error_catch_rate']:.0%} | {r['llm_calls']} |")
    md += ["", "*Silent error* = a wrong document the self-check did not flag, so nobody reviews it. That is the number to minimise.",
           "*Flagged* documents go to a human review queue.", "", "## Accuracy per field", "",
           "| System | " + " | ".join(FIELDS) + " |", "|---|" + "---|" * len(FIELDS)]
    for r in rows:
        md.append(f"| {r['name']} | " + " | ".join(f"{r['field_accuracy'][f]:.0%}" for f in FIELDS) + " |")
    md += ["", "## Accuracy by scan quality (docs fully correct)", "", "| System | clean | mild | harsh |", "|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['name']} | " + " | ".join(f"{r['by_level'][lv]['doc_accuracy']:.0%}" if lv in r["by_level"] else "-" for lv in ("clean", "mild", "harsh")) + " |")
    md += ["", "## Where the errors are", ""]
    for r in rows:
        md.append(f"- **{r['name']}**: " + (", ".join(f"{f} ({c})" for f, c in r["errors_by_field"].items()) or "none"))
    md += ["", "## Limitations", "",
           "- Synthetic invoices with 3 layouts. Real invoices have far more variety, stamps, tables and handwriting.",
           "- Text-only pipeline (OCR then extraction). A vision-language model reading the image directly would avoid OCR mistakes such as the rupee sign.",
           "- Rules were developed while looking at one generated set; the numbers above are from a separate set with a different seed.",
           "- The arithmetic check can only catch errors that break the totals. A garbled vendor name or invoice number passes it."]
    return "\n".join(md) + "\n"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results", type=Path, default=ROOT / "results")
    a = p.parse_args()
    text = build(a.results)
    (a.results / "REPORT.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
