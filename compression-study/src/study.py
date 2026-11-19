from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import torch

from .benchmark import measure
from .common import ROOT
from .compress import count_nonzero, prune_and_finetune, quantize_static, sparsity, to_half
from .data import GPUData, load_cifar10, synthetic
from .models import ResNetCifar, count_params
from .train import evaluate, train


def get_or_train(tag: str, build, data: GPUData, out: Path, epochs: int, log, **kw) -> tuple[torch.nn.Module, dict]:
    path = out / f"{tag}.pt"
    model = build()
    if path.exists():
        blob = torch.load(path, map_location="cpu", weights_only=False)
        model.load_state_dict(blob["state"])
        log(f"  {tag}: loaded from {path.name} (delete it to retrain)")
        return model.to(data.device).eval(), blob["info"]
    info = train(model, data, epochs, log=log, tag=tag, **kw)
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"state": model.state_dict(), "info": info}, path)
    return model.eval(), info


def run(args: argparse.Namespace, log=print) -> dict:
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    log(f"device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else "  (CPU training is slow; use a GPU for the real run)"))
    raw = synthetic(1500, 400) if args.dataset == "synthetic" else load_cifar10(ROOT / "data")
    data = GPUData(raw, device, args.subset)
    out = Path(args.out)
    key = f"{args.dataset}_e{args.epochs}_n{args.subset or 'all'}"
    gpu = ("cpu", "cuda") if device == "cuda" else ("cpu",)
    iters = args.bench_iters
    rows, curves = [], {}

    def add(row: dict) -> None:
        rows.append(row)
        log(f"  -> {row['name']}: acc {row['accuracy']:.3f}  size {row['gzip_mb']:.2f} MB(gz)  cpu {row['cpu_ms_b1']:.1f} ms/img")
        (out / "partial.json").parent.mkdir(parents=True, exist_ok=True)
        (out / "partial.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")

    log("== teacher")
    teacher, t_info = get_or_train(f"teacher_w{args.width}_{key}", lambda: ResNetCifar(args.width), data, out, args.epochs, log, lr=args.lr)
    base_params = count_params(teacher)
    curves["teacher"] = t_info["history"]
    add(measure("fp32 baseline", "precision", teacher, data, gpu, iters=iters, note=f"width {args.width}"))
    if device == "cuda":
        for dt, label in ((torch.float16, "fp16"), (torch.bfloat16, "bf16")):
            add(measure(label, "precision", to_half(teacher, dt), data, ("cuda",), dtype=dt, params=base_params, iters=iters))
    log("== int8 quantization")
    q = quantize_static(teacher, data.calibration(args.calib))
    add(measure("int8 (static)", "precision", q, data, ("cpu",), params=base_params, nonzero=base_params, iters=iters,
                note="weights and activations in 8 bits; CPU only"))
    log("== pruning")
    for amount in args.prune:
        p = prune_and_finetune(teacher, data, amount, args.prune_epochs, log=log)
        add(measure(f"prune {amount:.0%}", "pruning", p, data, gpu, iters=iters,
                    note=f"{sparsity(p):.0%} of weights are zero; dense storage and dense kernels"))
    log("== smaller student (width %d)" % args.student_width)
    scratch, s_info = get_or_train(f"student_scratch_w{args.student_width}_{key}", lambda: ResNetCifar(args.student_width), data, out, args.epochs, log, lr=args.lr)
    curves["student_scratch"] = s_info["history"]
    add(measure("student, trained alone", "smaller", scratch, data, gpu, iters=iters, note=f"width {args.student_width}"))
    kd, k_info = get_or_train(f"student_kd_w{args.student_width}_{key}_t{args.width}", lambda: ResNetCifar(args.student_width), data, out, args.epochs, log,
                              lr=args.lr, teacher=teacher, T=args.temperature, alpha=args.alpha)
    curves["student_kd"] = k_info["history"]
    add(measure("student, distilled", "smaller", kd, data, gpu, iters=iters, note=f"same student, taught by the width-{args.width} teacher"))
    qk = quantize_static(kd, data.calibration(args.calib))
    add(measure("distilled + int8", "smaller", qk, data, ("cpu",), params=count_params(kd), nonzero=count_params(kd), iters=iters,
                note="smaller architecture and 8-bit weights together"))
    meta = {"dataset": args.dataset, "epochs": args.epochs, "train_images": len(data.xtr), "test_images": len(data.xte), "device": device,
            "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None, "cpu_threads": torch.get_num_threads(),
            "platform": platform.platform(), "torch": torch.__version__, "teacher_width": args.width, "student_width": args.student_width,
            "prune_finetune_epochs": args.prune_epochs, "kd_T": args.temperature, "kd_alpha": args.alpha}
    result = {"meta": meta, "rows": rows, "curves": curves}
    return result


def main() -> None:
    p = argparse.ArgumentParser(description="Model compression study")
    p.add_argument("--dataset", choices=["cifar10", "synthetic"], default="cifar10")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--subset", type=int, default=0, help="train on only the first N images (quick trials)")
    p.add_argument("--width", type=int, default=64, help="teacher width (64 = ResNet-18 sized)")
    p.add_argument("--student-width", type=int, default=32)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--prune", type=float, nargs="+", default=[0.5, 0.8, 0.9])
    p.add_argument("--prune-epochs", type=int, default=5)
    p.add_argument("--temperature", type=float, default=4.0)
    p.add_argument("--alpha", type=float, default=0.9)
    p.add_argument("--calib", type=int, default=512, help="images used to calibrate int8 ranges")
    p.add_argument("--bench-iters", type=int, default=20)
    p.add_argument("--cpu", action="store_true", help="force CPU")
    p.add_argument("--out", default=str(ROOT / "outputs"))
    p.add_argument("--results", default=str(ROOT / "results"))
    a = p.parse_args()
    result = run(a)
    res = Path(a.results)
    res.mkdir(parents=True, exist_ok=True)
    (res / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    from .report import write_report

    write_report(result, res)
    print(f"\nwrote {res / 'REPORT.md'} and {res / 'chart.png'}")


if __name__ == "__main__":
    main()
