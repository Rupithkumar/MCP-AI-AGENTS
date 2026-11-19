from __future__ import annotations

import json
from pathlib import Path

from .data import ROOT
from .metrics import accuracy, macro_f1

RESULTS = ROOT / "results"
ORDER = ["tfidf", "base_zero", "base_few", "lora", "hosted"]
TRAIN_LOG = ROOT / "outputs" / "lora" / "train_log.json"


def load_results(results: Path = RESULTS) -> list[dict]:
    found = {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in (results / "systems").glob("*.json")}
    return [found[n] for n in ORDER if n in found] + [v for k, v in found.items() if k not in ORDER]


def load_preds(name: str, results: Path = RESULTS) -> list[dict]:
    path = results / "predictions" / f"{name}.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def table(rows: list[dict]) -> list[str]:
    lines = ["| System | n | Accuracy | Macro-F1 | Invalid | Prompt tokens | ms/example |", "|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['name']} | {r['n']} | {r['accuracy']:.3f} | {r['macro_f1']:.3f} | {r['invalid_rate']:.1%} | "
                     f"{r['prompt_tokens_mean']:.0f} | {r['ms_per_example']:.0f} |")
    return lines


def same_subset(rows: list[dict], results: Path = RESULTS) -> list[str]:
    preds = {r["name"]: load_preds(r["name"], results) for r in rows}
    preds = {k: v for k, v in preds.items() if v}
    if len({len(v) for v in preds.values()}) <= 1:
        return []
    shared = set.intersection(*[{p["text"] for p in v} for v in preds.values()])
    if not shared:
        return []
    labels = sorted({p["label"] for v in preds.values() for p in v})
    out = [f"## Same-subset comparison ({len(shared)} examples every system saw)", "",
           "| System | Accuracy | Macro-F1 |", "|---|---|---|"]
    for name, v in preds.items():
        sub = [p for p in v if p["text"] in shared]
        truth, pred = [p["label"] for p in sub], [p["pred"] for p in sub]
        out.append(f"| {name} | {accuracy(truth, pred):.3f} | {macro_f1(truth, pred, labels):.3f} |")
    return out + [""]


def build(results: Path = RESULTS, train_log: Path = TRAIN_LOG) -> str:
    rows = load_results(results)
    if not rows:
        raise SystemExit("No results yet. Run: python -m src.evaluate --systems tfidf")
    md = ["# Fine-tuning vs prompting vs classic ML", "", "All systems are scored on the same held-out test rows.", "",
          "## Results", "", *table(rows), "",
          "Invalid = the model's answer was not one of the known labels (counted as wrong). "
          "Prompt tokens = mean input length per example, a proxy for inference cost.", ""]
    md += same_subset(rows, results)
    if train_log.exists():
        t = json.loads(train_log.read_text(encoding="utf-8"))
        md += ["## Training run", "",
               f"- Base model: {t['base_model']}", f"- Trainable parameters: {t['trainable_params']:,} of {t['total_params']:,} "
               f"({100 * t['trainable_params'] / t['total_params']:.2f}%)",
               f"- LoRA r={t['lora_r']}, alpha={t['lora_alpha']}, lr={t['lr']}, epochs={t['epochs']}, batch={t['batch_size']}",
               f"- Training examples: {t['n_train']}, validation: {t['n_val']}",
               f"- Training time: {t['train_seconds'] / 60:.1f} min" + (f" on {t['gpu']}" if t.get("gpu") else " (CPU)"),
               f"- Train loss: {t['first_train_loss']} to {t['final_train_loss']}", ""]
    md += ["## Most common mistakes", ""]
    for r in rows:
        conf = r.get("top_confusions") or []
        md.append(f"**{r['name']}**")
        md += [f"- {t} → {p} ({c}x)" for t, p, c in conf[:5]] or ["- none"]
        md.append("")
    return "\n".join(md)


def main() -> None:
    text = build()
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "REPORT.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
