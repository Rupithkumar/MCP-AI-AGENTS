from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from .data import ROOT, load_prepared
from .prompts import build_messages, render


def encode_example(tokenizer, text: str, label: str, labels: list[str], style: str, max_len: int,
                   shots: list[dict] | None = None) -> dict:
    prompt_ids = tokenizer(render(tokenizer, build_messages(text, labels, style, shots)), add_special_tokens=False)["input_ids"]
    answer_ids = tokenizer(label + tokenizer.eos_token, add_special_tokens=False)["input_ids"]
    room = max_len - len(answer_ids)
    if room < 1:
        raise ValueError(f"max_len={max_len} is too small for the answer")
    prompt_ids = prompt_ids[-room:] 
    return {
        "input_ids": prompt_ids + answer_ids,
        "labels": [-100] * len(prompt_ids) + answer_ids,
    }


class Collator:
    def __init__(self, pad_id: int):
        self.pad_id = pad_id

    def __call__(self, batch: list[dict]) -> dict:
        import torch

        longest = max(len(b["input_ids"]) for b in batch)
        ids, mask, labels = [], [], []
        for b in batch:
            pad = longest - len(b["input_ids"])
            ids.append(b["input_ids"] + [self.pad_id] * pad)
            mask.append([1] * len(b["input_ids"]) + [0] * pad)
            labels.append(b["labels"] + [-100] * pad)
        return {"input_ids": torch.tensor(ids), "attention_mask": torch.tensor(mask), "labels": torch.tensor(labels)}


def pick_dtype():
    import torch

    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float32


def load_base(model_name: str, dtype=None):
    import transformers
    from packaging.version import Version
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    key = "dtype" if Version(transformers.__version__) >= Version("4.56") else "torch_dtype"
    model = AutoModelForCausalLM.from_pretrained(model_name, **{key: dtype or pick_dtype()})
    return model, tokenizer


def train(model_name: str, out_dir: Path, epochs: float = 3, batch_size: int = 16, grad_accum: int = 1,
          lr: float = 2e-4, max_len: int = 192, lora_r: int = 16, lora_alpha: int = 32, lora_dropout: float = 0.05,
          max_steps: int = -1, limit_train: int = 0, grad_ckpt: bool = False, seed: int = 42, data_dir: Path | None = None,
          model=None, tokenizer=None) -> dict:
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import Trainer, TrainingArguments, set_seed

    set_seed(seed)
    data = load_prepared(data_dir) if data_dir else load_prepared()
    labels = data["labels"]
    train_rows = data["train"][:limit_train] if limit_train else data["train"]
    if model is None:
        model, tokenizer = load_base(model_name)

    enc = lambda rows: [encode_example(tokenizer, r["text"], r["label"], labels, "short", max_len) for r in rows]  # noqa: E731
    train_ds, val_ds = enc(train_rows), enc(data["val"])

    lora = LoraConfig(r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, target_modules="all-linear",
                      task_type="CAUSAL_LM")
    model = get_peft_model(model, lora)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"trainable parameters: {trainable:,} of {total:,} ({100 * trainable / total:.2f}%)")
    if grad_ckpt:
        model.enable_input_require_grads()

    on_gpu = torch.cuda.is_available()
    steps_per_epoch = max(1, -(-len(train_ds) // (batch_size * grad_accum)))
    total_steps = max_steps if max_steps > 0 else int(steps_per_epoch * epochs)
    warmup_steps = max(1, int(0.05 * total_steps)) 
    args = TrainingArguments(
        output_dir=str(out_dir / "trainer"), per_device_train_batch_size=batch_size, per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum, num_train_epochs=epochs, max_steps=max_steps, learning_rate=lr,
        lr_scheduler_type="cosine", warmup_steps=warmup_steps, logging_steps=10, eval_strategy="epoch", save_strategy="no",
        bf16=on_gpu and torch.cuda.is_bf16_supported(), use_cpu=not on_gpu, report_to="none", remove_unused_columns=False,
        seed=seed, gradient_checkpointing=grad_ckpt, dataloader_pin_memory=on_gpu,
    )
    trainer = Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds,
                      data_collator=Collator(tokenizer.pad_token_id))
    started = time.perf_counter()
    trainer.train()
    seconds = time.perf_counter() - started

    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir / "adapter")
    history = trainer.state.log_history
    train_losses = [h["loss"] for h in history if "loss" in h]
    val_losses = [h["eval_loss"] for h in history if "eval_loss" in h]
    summary = {
        "base_model": model_name, "labels": labels, "epochs": epochs, "batch_size": batch_size, "grad_accum": grad_accum,
        "lr": lr, "max_len": max_len, "lora_r": lora_r, "lora_alpha": lora_alpha, "seed": seed,
        "n_train": len(train_ds), "n_val": len(val_ds), "trainable_params": trainable, "total_params": total,
        "train_seconds": round(seconds, 1), "first_train_loss": train_losses[0] if train_losses else None,
        "final_train_loss": train_losses[-1] if train_losses else None, "val_losses": val_losses,
        "gpu": torch.cuda.get_device_name(0) if on_gpu else None, "log_history": history,
    }
    (out_dir / "train_log.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"done in {seconds / 60:.1f} min. adapter saved to {out_dir / 'adapter'}")
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description="LoRA fine-tuning")
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--out", type=Path, default=ROOT / "outputs" / "lora")
    p.add_argument("--epochs", type=float, default=3)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--max-len", type=int, default=192)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--max-steps", type=int, default=-1, help="stop after N steps (for quick trials)")
    p.add_argument("--limit-train", type=int, default=0, help="use only the first N training rows")
    p.add_argument("--grad-ckpt", action="store_true", help="save GPU memory at the cost of speed")
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    train(a.model, a.out, a.epochs, a.batch_size, a.grad_accum, a.lr, a.max_len, a.lora_r, a.lora_alpha,
          max_steps=a.max_steps, limit_train=a.limit_train, grad_ckpt=a.grad_ckpt, seed=a.seed)


if __name__ == "__main__":
    main()
