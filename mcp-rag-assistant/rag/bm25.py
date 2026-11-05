from __future__ import annotations

import math
from collections import Counter


class BM25:
    def __init__(self, corpus: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.n = len(corpus)
        self.lengths = [len(doc) for doc in corpus]
        self.avgdl = (sum(self.lengths) / self.n) if self.n else 0.0
        self.postings: dict[str, list[tuple[int, int]]] = {}
        for i, doc in enumerate(corpus):
            for term, tf in Counter(doc).items():
                self.postings.setdefault(term, []).append((i, tf))

    def idf(self, term: str) -> float:
        df = len(self.postings.get(term, ()))
        return math.log(1.0 + (self.n - df + 0.5) / (df + 0.5))

    def scores(self, query_tokens: list[str]) -> dict[int, float]:
        out: dict[int, float] = {}
        for term in set(query_tokens):
            postings = self.postings.get(term)
            if not postings:
                continue
            idf = self.idf(term)
            for i, tf in postings:
                norm = 1.0 - self.b + self.b * self.lengths[i] / (self.avgdl or 1.0)
                out[i] = out.get(i, 0.0) + idf * tf * (self.k1 + 1.0) / (tf + self.k1 * norm)
        return out
