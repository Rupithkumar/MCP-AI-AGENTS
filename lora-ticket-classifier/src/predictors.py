from __future__ import annotations

import time
from dataclasses import dataclass, field

from .prompts import build_messages, render


@dataclass
class Output:
    raw: list[str]
    prompt_tokens: list[float]
    seconds: float
    errors: int = 0
    notes: dict = field(default_factory=dict)


class TfidfPredictor:
    name = "tfidf"
    description = "TF-IDF + logistic regression (classic baseline, CPU)"

    def __init__(self, train_rows: list[dict]):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline

        self.model = make_pipeline(
            TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, lowercase=True),
            LogisticRegression(max_iter=2000, C=10.0),
        )
        started = time.perf_counter()
        self.model.fit([r["text"] for r in train_rows], [r["label"] for r in train_rows])
        self.fit_seconds = time.perf_counter() - started

    def predict(self, texts: list[str]) -> Output:
        started = time.perf_counter()
        raw = [str(x) for x in self.model.predict(texts)]
        return Output(raw, [0.0] * len(texts), time.perf_counter() - started, notes={"fit_seconds": round(self.fit_seconds, 2)})


class HFPredictor:
    def __init__(self, name: str, description: str, model, tokenizer, style: str, labels: list[str],
                 shots: list[dict] | None = None, batch_size: int = 16, max_new_tokens: int = 16):
        self.name, self.description = name, description
        self.model, self.tokenizer, self.style = model, tokenizer, style
        self.labels, self.shots, self.batch_size, self.max_new_tokens = labels, shots, batch_size, max_new_tokens
        self.model.eval()
        self.tokenizer.padding_side = "left"  

    def predict(self, texts: list[str]) -> Output:
        import torch

        prompts = [render(self.tokenizer, build_messages(t, self.labels, self.style, self.shots)) for t in texts]
        lengths = [len(self.tokenizer(p, add_special_tokens=False)["input_ids"]) for p in prompts]
        order = sorted(range(len(prompts)), key=lambda i: -lengths[i])  
        raw = [""] * len(prompts)
        device = next(self.model.parameters()).device
        started = time.perf_counter()
        for start in range(0, len(order), self.batch_size):
            idx = order[start : start + self.batch_size]
            enc = self.tokenizer([prompts[i] for i in idx], return_tensors="pt", padding=True, add_special_tokens=False).to(device)
            with torch.no_grad():
                out = self.model.generate(**enc, max_new_tokens=self.max_new_tokens, do_sample=False,
                                          pad_token_id=self.tokenizer.pad_token_id)
            decoded = self.tokenizer.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
            for i, text in zip(idx, decoded):
                raw[i] = text
        if device.type == "cuda":
            torch.cuda.synchronize()
        return Output(raw, [float(n) for n in lengths], time.perf_counter() - started)


class HostedPredictor:
    name = "hosted"

    def __init__(self, labels: list[str], shots: list[dict], model: str, base_url: str | None, api_key: str | None,
                 delay: float = 0.0, client=None):
        if client is None:
            from openai import OpenAI

            client = OpenAI(base_url=base_url, api_key=api_key or "not-needed", max_retries=2)
        self.client, self.model, self.labels, self.shots, self.delay = client, model, labels, shots, delay
        self.description = f"{model} via API, few-shot (a large hosted model)"

    def predict(self, texts: list[str]) -> Output:
        raw, tokens, errors = [], [], 0
        started = time.perf_counter()
        for text in texts:
            try:
                resp = self.client.chat.completions.create(
                    model=self.model, temperature=0, messages=build_messages(text, self.labels, "few_shot", self.shots))
                raw.append(resp.choices[0].message.content or "")
                tokens.append(float(getattr(resp.usage, "prompt_tokens", 0) or 0))
            except Exception as exc:  
                raw.append("")
                tokens.append(0.0)
                errors += 1
                print(f"  hosted error: {type(exc).__name__}: {str(exc)[:120]}")
            if self.delay:
                time.sleep(self.delay)
        return Output(raw, tokens, time.perf_counter() - started, errors=errors)
