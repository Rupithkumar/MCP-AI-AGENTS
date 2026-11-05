from __future__ import annotations

import hashlib
import math
import os
import re
from typing import Any, Protocol, Sequence


class Embedder(Protocol):
    name: str

    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


def normalize(vec: Sequence[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / norm for x in vec]


class HashingEmbedder:
    def __init__(self, dim: int = 256):
        self.dim, self.name = dim, f"hashing-{dim}"

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        out = []
        for text in texts:
            vec = [0.0] * self.dim
            for token in re.findall(r"[a-z0-9]+", text.lower()):
                digest = hashlib.md5(token.encode()).digest()
                idx = int.from_bytes(digest[:4], "big") % self.dim
                vec[idx] += 1.0 if digest[4] % 2 == 0 else -1.0
            out.append(normalize(vec))
        return out


class SentenceTransformerEmbedder:
    def __init__(self, model: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer  

        self._model = SentenceTransformer(model)
        self.name = f"st:{model}"

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return self._model.encode(list(texts), normalize_embeddings=True).tolist()


class OpenAICompatEmbedder:
    def __init__(self, model: str, base_url: str | None = None, api_key: str | None = None, client: Any = None):
        if client is None:
            from openai import OpenAI

            client = OpenAI(base_url=base_url, api_key=api_key or "not-needed")
        self._client, self._model = client, model
        self.name = f"openai:{model}"

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), 64):
            resp = self._client.embeddings.create(model=self._model, input=list(texts[i : i + 64]))
            out.extend(normalize(d.embedding) for d in resp.data)
        return out


def embedder_from_env() -> Embedder | None:
    spec = os.environ.get("EMBEDDER", "").strip()
    if not spec:
        return None
    if spec == "hashing":
        return HashingEmbedder()
    if spec.startswith("st:"):
        return SentenceTransformerEmbedder(spec[3:])
    if spec.startswith("openai:"):
        return OpenAICompatEmbedder(spec[7:], os.environ.get("EMBED_BASE_URL"), os.environ.get("EMBED_API_KEY"))
    raise ValueError(f"unknown EMBEDDER {spec!r}; use hashing, st:<model> or openai:<model>")
