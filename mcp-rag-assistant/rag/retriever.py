from __future__ import annotations

from dataclasses import dataclass

from .bm25 import BM25
from .embedders import Embedder, normalize
from .store import ChunkRow, Store
from .text import tokenize

RRF_K = 60  
CANDIDATES = 50


@dataclass
class Hit:
    chunk: ChunkRow
    score: float  
    rank: int  


class Retriever:
    def __init__(self, store: Store, embedder: Embedder | None = None, dense_min: float = 0.25):
        self.store, self.embedder, self.dense_min = store, embedder, dense_min
        self.chunks: list[ChunkRow] = []
        self.bm25 = BM25([])
        self.vectors: list = []
        self.reload()

    def reload(self) -> None:
        self.chunks = self.store.all_chunks()
        self.bm25 = BM25([tokenize(c.index_text) for c in self.chunks])
        self.vectors = []
        if self.embedder:
            by_id = self.store.load_vectors(self.embedder.name)
            self.vectors = [by_id.get(c.id) for c in self.chunks]

    @property
    def has_dense(self) -> bool:
        return bool(self.embedder) and any(v is not None for v in self.vectors)

    def resolve_method(self, method: str = "auto") -> str:
        if method == "auto":
            return "hybrid" if self.has_dense else "bm25"
        if method not in {"bm25", "dense", "hybrid"}:
            raise ValueError("method must be auto, bm25, dense or hybrid")
        if method in {"dense", "hybrid"} and not self.has_dense:
            raise ValueError(f"method '{method}' needs embeddings: set EMBEDDER and re-run indexing")
        return method

    def _bm25_ranking(self, query: str) -> list[tuple[int, float]]:
        scores = self.bm25.scores(tokenize(query))
        return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:CANDIDATES]

    def _dense_ranking(self, query: str) -> list[tuple[int, float]]:
        assert self.embedder is not None
        q = normalize(self.embedder.embed([query])[0])
        scored = []
        for i, vec in enumerate(self.vectors):
            if vec is None:
                continue
            sim = sum(a * b for a, b in zip(q, vec))
            if sim >= self.dense_min:
                scored.append((i, sim))
        return sorted(scored, key=lambda kv: kv[1], reverse=True)[:CANDIDATES]

    def search(self, query: str, k: int = 5, method: str = "auto") -> list[Hit]:
        method = self.resolve_method(method)
        if method == "bm25":
            ranking = self._bm25_ranking(query)
        elif method == "dense":
            ranking = self._dense_ranking(query)
        else:  
            fused: dict[int, float] = {}
            for ranker in (self._bm25_ranking, self._dense_ranking):
                for rank, (i, _) in enumerate(ranker(query), start=1):
                    fused[i] = fused.get(i, 0.0) + 1.0 / (RRF_K + rank)
            ranking = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)
        return [Hit(self.chunks[i], score, rank) for rank, (i, score) in enumerate(ranking[:k], start=1)]
