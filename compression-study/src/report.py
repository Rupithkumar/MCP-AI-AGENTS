from __future__ import annotations

import json
from pathlib import Path

GROUPS = {"precision": ("Precision (fp32, fp16, bf16, int8)", "#2a78d6"),
          "pruning": ("Pruning (zeroed weights)", "#eb6834"),
          "smaller": ("Smaller model (student, distillation)", "#1baf7a")}
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e7e6e1", "#fcfcfb"


def fmt(v, spec=".1f", dash="-"):
    return dash if v is None else format(v, spec)


def table(rows: list[dict]) -> list[str]:
    out = ["| Variant | Params (M) | Non-zero (M) | Size MB | Size gzip MB | Accuracy | CPU ms/img | CPU img/s | GPU ms/img | GPU img/s |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        out.append(f"| {r['name']} | {r['params'] / 1e6:.2f} | {r['nonzero'] / 1e6:.2f} | {r['size_mb']:.1f} | {r['gzip_mb']:.1f} | "
                   f"{r['accuracy']:.1%} | {fmt(r.get('cpu_ms_b1'))} | {fmt(r.get('cpu_img_per_s'), '.0f')} | "
                   f"{fmt(r.get('cuda_ms_b1'), '.2f')} | {fmt(r.get('cuda_img_per_s'), '.0f')} |")
    return out


def findings(rows: list[dict]) -> list[str]:
    base = next(r for r in rows if r["name"] == "fp32 baseline")
    out = []
    for r in rows:
        if r is base:
            continue
        parts = [f"{base['gzip_mb'] / r['gzip_mb']:.1f}x smaller on disk (gzip)", f"accuracy {100 * (r['accuracy'] - base['accuracy']):+.1f} points"]
        if r.get("cpu_ms_b1") and base.get("cpu_ms_b1"):
            parts.append(f"{base['cpu_ms_b1'] / r['cpu_ms_b1']:.2f}x CPU speed (batch 1)")
        if r.get("cuda_ms_b1") and base.get("cuda_ms_b1"):
            parts.append(f"{base['cuda_ms_b1'] / r['cuda_ms_b1']:.2f}x GPU speed (batch 1)")
        if r["group"] == "pruning":
            parts.append(f"{r['nonzero'] / base['nonzero']:.0%} of weights non-zero")
        out.append(f"- **{r['name']}**: " + ", ".join(parts))
    alone = next((r for r in rows if r["name"] == "student, trained alone"), None)
    kd = next((r for r in rows if r["name"] == "student, distilled"), None)
    if alone and kd:
        out.append(f"- **Distillation effect**: the same small network scores {100 * (kd['accuracy'] - alone['accuracy']):+.1f} points "
                   f"({alone['accuracy']:.1%} to {kd['accuracy']:.1%}) when taught by the teacher instead of trained alone. "
                   "One run each: a difference this small can be seed noise unless it is repeated.")
    return out


def chart(rows: list[dict], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

    fig, axes = plt.subplots(1, 2, figsize=(13, 6.0), facecolor=SURFACE)
    panels = (("gzip_mb", "Size on disk, gzip (MB, log scale)", True), ("cpu_ms_b1", "CPU latency (ms per image, batch 1)", False))
    handles = []
    for ax, (key, xlabel, logx) in zip(axes, panels):
        ax.set_facecolor(SURFACE)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_color(GRID)
        ax.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        pts_all = [r for r in rows if r.get(key)]
        for grp, (label, colour) in GROUPS.items():
            pts = [r for r in pts_all if r["group"] == grp]
            h = ax.scatter([r[key] for r in pts], [100 * r["accuracy"] for r in pts], s=85, color=colour, edgecolor=SURFACE, linewidth=2,
                           label=label, zorder=3)
            if ax is axes[0]:
                handles.append(h)
        if logx:
            ax.set_xscale("log")
            ax.xaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5)))
            ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
            ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_xlabel(xlabel, color=INK2)
        ax.tick_params(colors=INK2)
        ax.margins(x=0.18, y=0.12)
        fig.canvas.draw()
        renderer, placed = fig.canvas.get_renderer(), []
        for r in sorted(pts_all, key=lambda r: -r["accuracy"]):  
            xy = ax.transData.transform((r[key], 100 * r["accuracy"]))
            for dx, dy, ha in ((9, 6, "left"), (9, -15, "left"), (-9, 6, "right"), (-9, -15, "right"), (9, 20, "left"), (-9, 20, "right"), (9, -28, "left"), (-9, -28, "right")):
                t = ax.annotate(r["name"], (r[key], 100 * r["accuracy"]), textcoords="offset points", xytext=(dx, dy), ha=ha,
                                fontsize=8.5, color=INK2, zorder=4)
                box = t.get_window_extent(renderer).expanded(1.05, 1.15)
                markers = [ax.transData.transform((q[key], 100 * q["accuracy"])) for q in pts_all]
                clash = any(box.overlaps(o) for o in placed) or any(box.x0 - 6 < mx < box.x1 + 6 and box.y0 - 6 < my < box.y1 + 6 for mx, my in markers)
                if not clash:
                    placed.append(box)
                    break
                t.remove()
            else:
                t = ax.annotate(r["name"], (r[key], 100 * r["accuracy"]), textcoords="offset points", xytext=(9, 6), fontsize=8.5, color=INK2, zorder=4)
                placed.append(t.get_window_extent(renderer))
    axes[0].set_ylabel("Test accuracy (%)", color=INK2)
    fig.legend(handles=handles, labels=[g[0] for g in GROUPS.values()], loc="lower center", ncol=3, frameon=False, labelcolor=INK2, fontsize=9.5)
    fig.suptitle("Accuracy against size and speed for each compressed variant (up and left is better)", color=INK, fontsize=12, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def build(result: dict) -> str:
    m, rows = result["meta"], result["rows"]
    where = f"{m['gpu']} (training, GPU timing) and {m['cpu_threads']} CPU threads" if m.get("gpu") else f"CPU only, {m['cpu_threads']} threads"
    md = ["# Model compression study", "",
          f"ResNet-18-style network on **{m['dataset']}**: {m['train_images']} training and {m['test_images']} test images, "
          f"{m['epochs']} epochs per training run. Hardware: {where}. PyTorch {m['torch']}.", "",
          "## Results", "", *table(rows), "",
          "Latency is the median of repeated runs (batch 1 for ms/img; a larger batch for img/s). Size is the saved weights; gzip shows what zeros cost.",
          "int8 runs on the CPU only and half precision on the GPU only, so a dash means 'not applicable', not 'not measured'.", "",
          "![Accuracy vs size and speed](chart.png)", "", "## Computed findings (relative to the fp32 baseline)", "", *findings(rows), "",
          "## How to read this", "",
          "- **Unstructured pruning** sets weights to zero but the model stays the same shape, so a normal dense kernel does the same work. "
          "Compare its latency to the baseline's rather than assuming a speed-up; only the gzip size and non-zero count fall.",
          "- **int8** shrinks the weights and speeds up the CPU only if the CPU has fast int8 instructions; measure on your target hardware.",
          "- **A smaller architecture** is the only option here that reduces the work on every device. Distillation tries to recover the accuracy it costs.", "",
          "## Limitations", "",
          "- One dataset, one architecture, one run per variant: no error bars. Differences under about a point are not reliable.",
          "- Latency depends on the machine, thread count and background load. Compare variants within one run only.",
          "- Post-training int8 uses eager-mode quantization, which PyTorch is migrating to the torchao library.",
          "- Structured pruning (removing whole channels) is not covered."]
    return "\n".join(md) + "\n"


def write_report(result: dict, folder: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    chart(result["rows"], folder / "chart.png")
    (folder / "REPORT.md").write_text(build(result), encoding="utf-8")


if __name__ == "__main__":
    import sys

    folder = Path(sys.argv[1] if len(sys.argv) > 1 else "results")
    write_report(json.loads((folder / "results.json").read_text(encoding="utf-8")), folder)
    print(f"wrote {folder / 'REPORT.md'}")
