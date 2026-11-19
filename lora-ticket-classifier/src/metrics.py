from __future__ import annotations

import re
from collections import Counter

INVALID = "__invalid__"


def normalize(text: str) -> str:
    text = text.strip().strip("`'\".:;*[]()").lower()
    return re.sub(r"[\s_\-]+", " ", text).strip()


def parse_label(raw: str, labels: list[str]) -> str | None:
    lookup = {normalize(label): label for label in labels}
    lines = [line for line in raw.strip().splitlines() if line.strip()]
    if not lines:
        return None
    line = lines[0].strip()
    line = re.sub(r"^\W*label\s*[:\-]\s*", "", line, flags=re.IGNORECASE)  
    return lookup.get(normalize(line))


def accuracy(y_true: list[str], y_pred: list[str | None]) -> float:
    return sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true) if y_true else 0.0


def macro_f1(y_true: list[str], y_pred: list[str | None], labels: list[str]) -> float:
    scores = []
    for label in labels:
        tp = sum(t == label and p == label for t, p in zip(y_true, y_pred))
        fp = sum(t != label and p == label for t, p in zip(y_true, y_pred))
        fn = sum(t == label and p != label for t, p in zip(y_true, y_pred))
        denom = 2 * tp + fp + fn
        scores.append(2 * tp / denom if denom else 0.0)
    return sum(scores) / len(scores) if scores else 0.0


def top_confusions(y_true: list[str], y_pred: list[str | None], n: int = 8) -> list[tuple[str, str, int]]:
    pairs = Counter((t, p or INVALID) for t, p in zip(y_true, y_pred) if t != p)
    return [(t, p, c) for (t, p), c in pairs.most_common(n)]


def per_label_accuracy(y_true: list[str], y_pred: list[str | None]) -> dict[str, float]:
    total, right = Counter(y_true), Counter(t for t, p in zip(y_true, y_pred) if t == p)
    return {label: right[label] / total[label] for label in total}


def summarize(y_true: list[str], raw: list[str], labels: list[str]) -> dict:
    preds = [parse_label(r, labels) for r in raw]
    return {
        "n": len(y_true),
        "accuracy": accuracy(y_true, preds),
        "macro_f1": macro_f1(y_true, preds, labels),
        "invalid_rate": sum(p is None for p in preds) / len(preds) if preds else 0.0,
        "top_confusions": top_confusions(y_true, preds),
        "preds": preds,
    }
