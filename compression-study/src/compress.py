from __future__ import annotations

import copy
import warnings

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from torch.ao.quantization import convert, get_default_qconfig, prepare

from .data import GPUData
from .train import train


def prunable(model: nn.Module) -> list[tuple[nn.Module, str]]:
    return [(m, "weight") for m in model.modules() if isinstance(m, (nn.Conv2d, nn.Linear))]


def sparsity(model: nn.Module) -> float:
    zeros = total = 0
    for m, _ in prunable(model):
        w = m.weight
        zeros, total = zeros + (w == 0).sum().item(), total + w.numel()
    return zeros / total if total else 0.0


def count_nonzero(model: nn.Module) -> int:
    return sum((p != 0).sum().item() for p in model.parameters())


def make_permanent(model: nn.Module) -> None:
    for m, _ in prunable(model):
        if prune.is_pruned(m):
            prune.remove(m, "weight")


def prune_and_finetune(model: nn.Module, data: GPUData, amount: float, epochs: int, lr: float = 0.02, log=print) -> nn.Module:
    pruned = copy.deepcopy(model)
    prune.global_unstructured(prunable(pruned), pruning_method=prune.L1Unstructured, amount=amount)
    if epochs > 0:
        train(pruned, data, epochs, lr=lr, log=log, tag=f"prune {amount:.0%}")
    make_permanent(pruned)
    return pruned


def pick_engine() -> str:
    engines = torch.backends.quantized.supported_engines
    for name in ("fbgemm", "x86", "qnnpack"):
        if name in engines:
            return name
    raise RuntimeError(f"no int8 backend available: {engines}")


def quantize_static(model: nn.Module, calibration: torch.Tensor, batch: int = 128) -> nn.Module:
    engine = pick_engine()
    torch.backends.quantized.engine = engine
    q = copy.deepcopy(model).cpu().eval()
    q.fuse()
    q.qconfig = get_default_qconfig(engine)
    with warnings.catch_warnings():  
        warnings.simplefilter("ignore")
        prepare(q, inplace=True)
        with torch.no_grad():
            for i in range(0, len(calibration), batch):
                q(calibration[i : i + batch].cpu())
        return convert(q, inplace=True)


def to_half(model: nn.Module, dtype: torch.dtype) -> nn.Module:
    return copy.deepcopy(model).to(dtype).eval()
