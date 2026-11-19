import argparse
import json

from src.report import build, findings, write_report
from src.study import run


def test_whole_study_runs_and_reports(tmp_path):
    args = argparse.Namespace(dataset="synthetic", epochs=2, subset=0, width=8, student_width=4, lr=0.1, prune=[0.5, 0.9], prune_epochs=1,
                              temperature=4.0, alpha=0.9, calib=64, bench_iters=3, cpu=True, out=str(tmp_path / "outputs"))
    result = run(args, log=lambda s: None)
    names = [r["name"] for r in result["rows"]]
    assert names == ["fp32 baseline", "int8 (static)", "prune 50%", "prune 90%", "student, trained alone", "student, distilled", "distilled + int8"]
    again = run(args, log=lambda s: None)  
    assert [r["name"] for r in again["rows"]] == names
    write_report(result, tmp_path / "res")
    md = (tmp_path / "res" / "REPORT.md").read_text()
    assert (tmp_path / "res" / "chart.png").stat().st_size > 5000 and "Computed findings" in md and "Limitations" in md
    json.dumps(result)  


def test_findings_are_computed_from_rows():
    rows = [{"name": "fp32 baseline", "group": "precision", "gzip_mb": 40.0, "accuracy": 0.9, "cpu_ms_b1": 10.0, "nonzero": 100},
            {"name": "int8 (static)", "group": "precision", "gzip_mb": 10.0, "accuracy": 0.89, "cpu_ms_b1": 5.0, "nonzero": 100}]
    text = "\n".join(findings(rows))
    assert "4.0x smaller" in text and "-1.0 points" in text and "2.00x CPU speed" in text
