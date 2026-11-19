from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .data import ROOT, load_prepared
from .metrics import summarize
from .predictors import HFPredictor, HostedPredictor, Output, TfidfPredictor

RESULTS = ROOT / "results"
SYSTEMS = ("tfidf", "base_zero", "base_few", "lora", "hosted")


def score(name: str, description: str, rows: list[dict], out: Output, labels: list[str], extra: dict | None = None) -> dict:
    truth = [r["label"] for r in rows]
    keep = [i for i, raw in enumerate(out.raw) if not (out.errors and raw == "" and name == "hosted")]
    s = summarize([truth[i] for i in keep], [out.raw[i] for i in keep], labels)
    preds = s.pop("preds")
    n = len(rows)
    result = {
        "name": name, "description": description, **{k: s[k] for k in ("n", "accuracy", "macro_f1", "invalid_rate", "top_confusions")},
        "prompt_tokens_mean": sum(out.prompt_tokens) / n if n else 0.0,
        "ms_per_example": 1000 * out.seconds / n if n else 0.0, "seconds_total": round(out.seconds, 2),
        "errors": out.errors, "notes": out.notes, **(extra or {}),
    }
    detail = [{"text": rows[i]["text"], "label": truth[i], "raw": out.raw[i], "pred": preds[j]} for j, i in enumerate(keep)]
    return {"result": result, "predictions": detail}


def save(name: str, scored: dict, out_dir: Path = RESULTS) -> None:
    (out_dir / "systems").mkdir(parents=True, exist_ok=True)
    (out_dir / "predictions").mkdir(parents=True, exist_ok=True)
    (out_dir / "systems" / f"{name}.json").write_text(json.dumps(scored["result"], indent=2), encoding="utf-8")
    (out_dir / "predictions" / f"{name}.jsonl").write_text(
        "\n".join(json.dumps(p, ensure_ascii=False) for p in scored["predictions"]) + "\n", encoding="utf-8")


def load_lora(base_model: str, adapter: Path):
    from peft import PeftModel

    from .train import load_base

    model, tokenizer = load_base(base_model)
    model = PeftModel.from_pretrained(model, str(adapter))
    return model.merge_and_unload(), tokenizer  


def to_device(model):
    import torch

    return model.to("cuda") if torch.cuda.is_available() else model


def run_system(name: str, data: dict, args: argparse.Namespace) -> dict:
    labels, rows = data["labels"], data["test"]
    if name == "tfidf":
        p = TfidfPredictor(data["train"])
        return score(name, p.description, rows, p.predict([r["text"] for r in rows]), labels)
    if name in {"base_zero", "base_few"}:
        from .train import load_base

        model, tok = load_base(args.model)
        style = "zero_shot" if name == "base_zero" else "few_shot"
        p = HFPredictor(name, f"{args.model}, {style.replace('_', '-')} prompt (no training)", to_device(model), tok, style, labels,
                        data["shots"], args.batch_size)
        return score(name, p.description, rows, p.predict([r["text"] for r in rows]), labels, {"model": args.model})
    if name == "lora":
        adapter = Path(args.adapter)
        if not adapter.exists():
            raise SystemExit(f"No adapter at {adapter}. Train first: python -m src.train")
        base = json.loads((adapter.parent / "train_log.json").read_text(encoding="utf-8"))["base_model"]
        model, tok = load_lora(base, adapter)
        p = HFPredictor(name, f"{base} + LoRA adapter, short prompt", to_device(model), tok, "short", labels, None, args.batch_size)
        return score(name, p.description, rows, p.predict([r["text"] for r in rows]), labels, {"model": base})
    if name == "hosted":
        if not os.environ.get("LLM_MODEL"):
            raise SystemExit("Set LLM_MODEL (and LLM_BASE_URL, LLM_API_KEY) to evaluate a hosted model.")
        limit = args.limit or 200
        p = HostedPredictor(labels, data["shots"], os.environ["LLM_MODEL"], os.environ.get("LLM_BASE_URL"),
                            os.environ.get("LLM_API_KEY"), args.delay)
        return score(name, p.description, rows[:limit], p.predict([r["text"] for r in rows[:limit]]), labels,
                     {"model": os.environ["LLM_MODEL"]})
    raise SystemExit(f"unknown system {name!r}; choose from {', '.join(SYSTEMS)}")


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate systems on the test set")
    p.add_argument("--systems", default="tfidf", help=f"comma list of: {', '.join(SYSTEMS)}")
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct", help="base model for base_zero / base_few")
    p.add_argument("--adapter", default=str(ROOT / "outputs" / "lora" / "adapter"))
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--limit", type=int, default=0, help="only the first N test rows (hosted defaults to 200)")
    p.add_argument("--delay", type=float, default=0.0, help="seconds between hosted calls")
    args = p.parse_args()
    data = load_prepared()
    for name in [s.strip() for s in args.systems.split(",") if s.strip()]:
        if args.limit and name != "hosted":
            data = {**data, "test": data["test"][: args.limit]}
        scored = run_system(name, data, args)
        save(name, scored)
        r = scored["result"]
        print(f"{name:10s} n={r['n']}  accuracy={r['accuracy']:.3f}  macro-F1={r['macro_f1']:.3f}  "
              f"invalid={r['invalid_rate']:.3f}  prompt tokens={r['prompt_tokens_mean']:.0f}  ms/example={r['ms_per_example']:.0f}")
        data = load_prepared()  


if __name__ == "__main__":
    main()
