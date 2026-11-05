from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

from .chunker import chunk_pages
from .embedders import Embedder
from .loaders import SUPPORTED, LoaderError, load_bytes
from .store import Store

MAX_FILE_BYTES = 20 * 1024 * 1024
EMBED_BATCH = 32


@dataclass
class IndexReport:
    added: int = 0
    updated: int = 0
    unchanged: int = 0
    removed: int = 0
    chunks: int = 0
    documents: int = 0
    errors: list[str] = field(default_factory=list)

    def summary(self) -> str:
        text = (
            f"Indexed folder: {self.added} added, {self.updated} updated, {self.unchanged} unchanged, "
            f"{self.removed} removed. Collection now has {self.documents} documents / {self.chunks} chunks."
        )
        if self.errors:
            text += "\nSkipped:\n" + "\n".join(f"  - {e}" for e in self.errors)
        return text


def index_folder(
    store: Store,
    folder: str | Path,
    embedder: Embedder | None = None,
    target: int = 700,
    overlap: int = 1,
) -> IndexReport:
    root = Path(folder).resolve()
    report = IndexReport()

    found: dict[str, Path] = {}
    for path in sorted(root.rglob("*")) if root.is_dir() else []:
        if not path.is_file() or path.suffix.lower() not in SUPPORTED:
            continue
        if not path.resolve().is_relative_to(root):  
            report.errors.append(f"{path.name}: link points outside the documents folder")
            continue
        found[path.relative_to(root).as_posix()] = path

    known = store.doc_hashes()
    for name, path in found.items():
        try:
            if path.stat().st_size > MAX_FILE_BYTES:
                raise LoaderError(f"file larger than {MAX_FILE_BYTES // (1024 * 1024)} MB")
            data = path.read_bytes()
            sha = hashlib.sha256(data).hexdigest()
            if known.get(name) == sha:
                report.unchanged += 1
                continue
            chunks = chunk_pages(load_bytes(path.suffix, data), target=target, overlap=overlap)
            if not chunks:
                raise LoaderError("no text found")
        except (LoaderError, OSError) as exc:
            report.errors.append(f"{name}: {exc}")
            continue
        store.replace_document(name, sha, chunks)
        if name in known:
            report.updated += 1
        else:
            report.added += 1

    for name in set(known) - set(found):
        store.delete_document(name)
        report.removed += 1

    if embedder is not None:  
        missing = store.chunks_missing_vectors(embedder.name)
        for i in range(0, len(missing), EMBED_BATCH):
            batch = missing[i : i + EMBED_BATCH]
            vectors = embedder.embed([c.index_text for c in batch])
            store.put_vectors(embedder.name, [(c.id, v) for c, v in zip(batch, vectors)])

    stats = store.stats()
    report.documents, report.chunks = stats["documents"], stats["chunks"]
    return report
