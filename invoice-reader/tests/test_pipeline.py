import shutil

import pytest

from src import evaluate as ev
from src import report
from src.extract import RuleRepairExtractor
from src.generate import LEVELS, generate, load_truth

needs_tesseract = pytest.mark.skipif(shutil.which("tesseract") is None, reason="Tesseract is not installed")


def test_generator_is_deterministic_and_consistent(tmp_path):
    a = generate(9, seed=5, out=tmp_path / "a")
    b = generate(9, seed=5, out=tmp_path / "b")
    assert [{k: v for k, v in r.items()} for r in a] == [{k: v for k, v in r.items()} for r in b]
    for r in a:
        assert abs(r["subtotal"] + r["tax"] - r["total"]) < 0.011 and (tmp_path / "a" / r["image"]).exists()
    assert {r["level"] for r in a} == set(LEVELS)
    assert len(load_truth(tmp_path / "a")) == 9


@needs_tesseract
def test_end_to_end_with_real_ocr(tmp_path):
    recs = generate(6, seed=2, out=tmp_path / "inv")
    texts = {r["id"]: ev.get_text(r, tmp_path / "inv", tmp_path / "cache") for r in recs}
    out = ev.run(RuleRepairExtractor(), recs, texts)
    s = ev.summarize("rules_repair", out["rows"], out["seconds"])
    assert s["n"] == 6 and 0 <= s["silent_error_rate"] <= 1 and s["doc_accuracy"] >= 0.5  
    ev.save("rules_repair", s, out["rows"], tmp_path / "res")
    md = report.build(tmp_path / "res")
    assert "Silent errors" in md and "rules_repair" in md and "Limitations" in md


def test_summary_metrics_arithmetic():
    mk = lambda ok, flagged: {"level": "clean", "doc_correct": ok, "flagged": flagged, "llm_called": False, "llm_error": False,  # noqa: E731
                              "correct": {f: ok for f in ev.FIELDS}}
    rows = [mk(True, False), mk(False, True), mk(False, False), mk(True, False)]
    s = ev.summarize("x", rows, 1.0)
    assert s["doc_accuracy"] == 0.5 and s["flagged_rate"] == 0.25 and s["silent_error_rate"] == 0.25
    assert s["accepted_accuracy"] == pytest.approx(2 / 3) and s["error_catch_rate"] == 0.5
