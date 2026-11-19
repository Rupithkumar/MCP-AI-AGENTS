from __future__ import annotations

import time

import torch
import torch.nn.functional as F

from .data import GPUData


def distill_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor, target: torch.Tensor,
                 T: float = 4.0, alpha: float = 0.9) -> torch.Tensor:
    soft = F.kl_div(F.log_softmax(student_logits / T, dim=1), F.softmax(teacher_logits / T, dim=1), reduction="batchmean") * T * T
    return alpha * soft + (1 - alpha) * F.cross_entropy(student_logits, target)


@torch.no_grad()
def evaluate(model, data: GPUData, batch: int = 500, dtype: torch.dtype | None = None, device: str | torch.device | None = None) -> float:
    model.eval()
    dev = torch.device(device) if device else data.device
    right = total = 0
    for x, y in data.test_batches(batch):
        x = x.to(dev)
        if dtype is not None:
            x = x.to(dtype)
        right += (model(x).argmax(1).cpu() == y.cpu()).sum().item()
        total += len(y)
    return right / total


def train(model, data: GPUData, epochs: int, lr: float = 0.1, weight_decay: float = 5e-4, batch: int = 128, teacher=None,
          T: float = 4.0, alpha: float = 0.9, seed: int = 0, log=print, tag: str = "") -> dict:
    dev = data.device
    gen = torch.Generator(device=dev).manual_seed(seed)
    model.to(dev).train()
    if teacher is not None:
        teacher.to(dev).eval()
    steps = max(1, epochs * (len(data.xtr) // batch))
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, nesterov=True, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.2)
    use_amp = dev.type == "cuda" and torch.cuda.is_bf16_supported()
    history, started = [], time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        total, n = 0.0, 0
        for x, y in data.train_batches(batch, gen):
            with torch.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=use_amp):
                logits = model(x)
                if teacher is None:
                    loss = F.cross_entropy(logits, y)
                else:
                    with torch.no_grad():
                        t_logits = teacher(x)
                    loss = distill_loss(logits.float(), t_logits.float(), y, T, alpha)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if sched.last_epoch < steps - 1:
                sched.step()
            total, n = total + loss.item(), n + 1
        acc = evaluate(model, data)
        history.append({"epoch": epoch, "loss": total / max(n, 1), "test_acc": acc})
        log(f"  {tag} epoch {epoch}/{epochs}  loss {total / max(n, 1):.3f}  test acc {acc:.3f}")
    return {"history": history, "seconds": time.perf_counter() - started, "final_acc": history[-1]["test_acc"]}
