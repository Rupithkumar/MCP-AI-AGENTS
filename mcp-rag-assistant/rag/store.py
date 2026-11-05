from __future__ import annotations

import sqlite3
from array import array
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

from .chunker import Chunk

SCHEMA = """
CREATE TABLE IF NOT EXISTS docs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE NOT NULL,
    sha256 TEXT NOT NULL,
    n_chunks INTEGER NOT NULL,
    indexed_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id INTEGER NOT NULL REFERENCES docs(id) ON DELETE CASCADE,
    ord INTEGER NOT NULL,
    section TEXT NOT NULL DEFAULT '',
    page INTEGER,
    text TEXT NOT NULL,
    UNIQUE (doc_id, ord)
);
CREATE TABLE IF NOT EXISTS vectors (
    chunk_id INTEGER NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
    model TEXT NOT NULL,
    vec BLOB NOT NULL,
    PRIMARY KEY (chunk_id, model)
);
"""


@dataclass(frozen=True)
class ChunkRow:
    id: int
    doc: str
    ord: int
    section: str
    page: int | None
    text: str

    @property
    def cite(self) -> str:
        return f"{self.doc}#{self.ord}"

    @property
    def index_text(self) -> str:
        return f"{self.section} {self.text}" if self.section else self.text


_CHUNK_SELECT = (
    "SELECT c.id, d.name AS doc, c.ord, c.section, c.page, c.text "
    "FROM chunks c JOIN docs d ON d.id = c.doc_id"
)


def _row(r: sqlite3.Row) -> ChunkRow:
    return ChunkRow(r["id"], r["doc"], r["ord"], r["section"], r["page"], r["text"])


class Store:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as conn:
            conn.executescript(SCHEMA)

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    # ---- documents
    def doc_hashes(self) -> dict[str, str]:
        with self._conn() as conn:
            return {r["name"]: r["sha256"] for r in conn.execute("SELECT name, sha256 FROM docs")}

    def replace_document(self, name: str, sha256: str, chunks: Sequence[Chunk]) -> None:
        with self._conn() as conn:
            conn.execute("DELETE FROM docs WHERE name = ?", (name,))  # cascades to chunks and vectors
            cur = conn.execute(
                "INSERT INTO docs (name, sha256, n_chunks) VALUES (?, ?, ?)", (name, sha256, len(chunks))
            )
            doc_id = cur.lastrowid
            conn.executemany(
                "INSERT INTO chunks (doc_id, ord, section, page, text) VALUES (?, ?, ?, ?, ?)",
                [(doc_id, c.ord, c.section, c.page, c.text) for c in chunks],
            )

    def delete_document(self, name: str) -> None:
        with self._conn() as conn:
            conn.execute("DELETE FROM docs WHERE name = ?", (name,))

    def list_docs(self) -> list[dict]:
        with self._conn() as conn:
            return [dict(r) for r in conn.execute("SELECT name, n_chunks, indexed_at FROM docs ORDER BY name")]

    # ---- chunks
    def all_chunks(self) -> list[ChunkRow]:
        with self._conn() as conn:
            return [_row(r) for r in conn.execute(_CHUNK_SELECT + " ORDER BY d.name, c.ord")]

    def get_chunks(self, doc: str, first: int, last: int) -> list[ChunkRow]:
        with self._conn() as conn:
            rows = conn.execute(
                _CHUNK_SELECT + " WHERE d.name = ? AND c.ord BETWEEN ? AND ? ORDER BY c.ord", (doc, first, last)
            )
            return [_row(r) for r in rows]

    def stats(self) -> dict:
        with self._conn() as conn:
            docs = conn.execute("SELECT COUNT(*) FROM docs").fetchone()[0]
            chunks = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
            models = [r[0] for r in conn.execute("SELECT DISTINCT model FROM vectors")]
        return {"documents": docs, "chunks": chunks, "embedding_models": models}

    # ---- vectors
    def chunks_missing_vectors(self, model: str) -> list[ChunkRow]:
        sql = (
            _CHUNK_SELECT
            + " LEFT JOIN vectors v ON v.chunk_id = c.id AND v.model = ? WHERE v.chunk_id IS NULL ORDER BY c.id"
        )
        with self._conn() as conn:
            return [_row(r) for r in conn.execute(sql, (model,))]

    def put_vectors(self, model: str, items: Sequence[tuple[int, Sequence[float]]]) -> None:
        with self._conn() as conn:
            conn.executemany(
                "INSERT OR REPLACE INTO vectors (chunk_id, model, vec) VALUES (?, ?, ?)",
                [(cid, model, array("f", vec).tobytes()) for cid, vec in items],
            )

    def load_vectors(self, model: str) -> dict[int, array]:
        out: dict[int, array] = {}
        with self._conn() as conn:
            for r in conn.execute("SELECT chunk_id, vec FROM vectors WHERE model = ?", (model,)):
                vec = array("f")
                vec.frombytes(r["vec"])
                out[r["chunk_id"]] = vec
        return out
