from __future__ import annotations

import gzip
import io
import statistics
import time

import torch

from .compress import count_nonzero
from .data import GPUData
from .models import count_params
from .train import evaluate


def model_bytes(model: torch.nn.Module) -> tuple[int, int]:
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    raw = buf.getvalue()
    return len(raw), len(gzip.compress(raw, compresslevel=6))


@torch.no_grad()
def latency(model: torch.nn.Module, device: str, batch: int, dtype: torch.dtype = torch.float32, warmup: int = 5, iters: int = 20) -> float:
    model = model.to(device).eval()
    x = torch.randn(batch, 3, 32, 32, device=device, dtype=dtype)
    times = []
    for i in range(warmup + iters):
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        model(x)
        if device == "cuda":
            torch.cuda.synchronize()
        if i >= warmup:
            times.append(time.perf_counter() - t0)
    return statistics.median(times)


def measure(name: str, group: str, model: torch.nn.Module, data: GPUData, devices: tuple[str, ...] = ("cpu", "cuda"),
            dtype: torch.dtype = torch.float32, params: int | None = None, nonzero: int | None = None, note: str = "",
            thr_batch_cpu: int = 64, thr_batch_gpu: int = 256, iters: int = 20) -> dict:
    raw, gz = model_bytes(model)
    row = {"name": name, "group": group, "params": params if params is not None else count_params(model),
           "nonzero": nonzero if nonzero is not None else count_nonzero(model), "size_mb": raw / 2**20, "gzip_mb": gz / 2**20, "note": note}
    primary = "cuda" if ("cuda" in devices and torch.cuda.is_available()) else "cpu"
    # accuracy on the device the model is meant for (the int8 model is CPU only)
    row["accuracy"] = evaluate(model.to(primary), data, dtype=dtype if dtype != torch.float32 else None, device=primary)
    for dev in devices:
        if dev == "cuda" and not torch.cuda.is_available():
            continue
        b1 = latency(model, dev, 1, dtype, iters=iters)
        bt = thr_batch_gpu if dev == "cuda" else thr_batch_cpu
        row[f"{dev}_ms_b1"] = b1 * 1000
        row[f"{dev}_img_per_s"] = bt / latency(model, dev, bt, dtype, iters=max(5, iters // 2))
    return row
