import json
from pathlib import Path

import pytest

from src import evaluate as ev
from src import report
from src.predictors import HFPredictor, HostedPredictor, TfidfPredictor
from src.train import train


def test_tfidf_learns_synthetic(prepared):
    _, d = prepared
    p = TfidfPredictor(d["train"])
    out = p.predict([r["text"] for r in d["test"]])
    s = ev.score("tfidf", p.description, d["test"], out, d["labels"])
    assert s["result"]["accuracy"] > 0.9 and s["result"]["invalid_rate"] == 0


def test_generation_and_scoring_shapes(prepared, tiny_lm):
    _, d = prepared
    model, tok = tiny_lm
    p = HFPredictor("base_zero", "tiny", model, tok, "zero_shot", d["labels"], d["shots"], batch_size=4, max_new_tokens=4)
    out = p.predict([r["text"] for r in d["test"][:10]])
    assert len(out.raw) == 10 and all(n > 0 for n in out.prompt_tokens)
    s = ev.score("base_zero", "tiny", d["test"][:10], out, d["labels"])
    assert s["result"]["n"] == 10 and 0 <= s["result"]["invalid_rate"] <= 1


def test_train_saves_adapter_and_loss_drops(prepared, tiny_lm, tmp_path):
    data_dir, d = prepared
    model, tok = tiny_lm
    out = tmp_path / "lora"
    summary = train("tiny", out, epochs=6, batch_size=8, lr=3e-3, max_len=96, lora_r=8, data_dir=data_dir, model=model, tokenizer=tok)
    assert (out / "adapter" / "adapter_config.json").exists() and (out / "train_log.json").exists()
    assert summary["final_train_loss"] < summary["first_train_loss"]
    assert 0 < summary["trainable_params"] < summary["total_params"]


class FakeClient:
    class chat:
        class completions:
            calls = 0

            @classmethod
            def create(cls, **kw):
                cls.calls += 1
                if cls.calls % 3 == 0:
                    raise RuntimeError("rate limited")
                msg = type("M", (), {"content": "card_lost"})
                return type("R", (), {"choices": [type("C", (), {"message": msg})], "usage": type("U", (), {"prompt_tokens": 100})})


def test_hosted_errors_are_not_scored_wrong(prepared):
    _, d = prepared
    rows = d["test"][:9]
    p = HostedPredictor(d["labels"], d["shots"], "m", None, None, client=FakeClient())
    s = ev.score("hosted", p.description, rows, p.predict([r["text"] for r in rows]), d["labels"])
    assert s["result"]["errors"] == 3 and s["result"]["n"] == 6


def test_report_roundtrip(prepared, tmp_path):
    _, d = prepared
    p = TfidfPredictor(d["train"])
    ev.save("tfidf", ev.score("tfidf", p.description, d["test"], p.predict([r["text"] for r in d["test"]]), d["labels"]), tmp_path)
    p2 = HostedPredictor(d["labels"], d["shots"], "m", None, None, client=FakeClient())
    rows = d["test"][:12]
    ev.save("hosted", ev.score("hosted", p2.description, rows, p2.predict([r["text"] for r in rows]), d["labels"]), tmp_path)
    md = report.build(tmp_path, tmp_path / "none.json")
    assert "| tfidf |" in md and "Same-subset" in md and "→" in md


def test_lora_roundtrip_through_evaluate(prepared, tiny_lm, tmp_path):
    """Train, save base+tokenizer to disk, then load adapter the way `evaluate --systems lora` does."""
    data_dir, d = prepared
    model, tok = tiny_lm
    base_dir = tmp_path / "base"
    model.save_pretrained(base_dir)
    tok.save_pretrained(base_dir)
    out = tmp_path / "lora"
    train(str(base_dir), out, epochs=1, batch_size=8, max_len=96, lora_r=4, data_dir=data_dir, model=model, tokenizer=tok)
    merged, tok2 = ev.load_lora(str(base_dir), out / "adapter")
    p = HFPredictor("lora", "tiny", merged, tok2, "short", d["labels"], None, batch_size=4, max_new_tokens=4)
    assert len(p.predict([r["text"] for r in d["test"][:5]]).raw) == 5
