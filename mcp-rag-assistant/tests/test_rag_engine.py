from __future__ import annotations

import io
from pathlib import Path
from types import SimpleNamespace

import pytest

from rag import HashingEmbedder, Retriever, Store, chunk_pages, index_folder
from rag.bm25 import BM25
from rag.embedders import OpenAICompatEmbedder
from rag.loaders import LoaderError, Page, load_bytes
from rag.text import stem, tokenize


@pytest.mark.parametrize(
    "family",
    [
        ["embedding", "embeddings", "embedded", "embed"],
        ["retrieve", "retrieved", "retrieves", "retrieving"],
        ["index", "indexes", "indexed", "indexing"],
        ["chunk", "chunks", "chunking", "chunked"],
        ["query", "queries"],
    ],
)
def test_stemmer_collapses_word_families(family):
    assert len({stem(w) for w in family}) == 1


def test_tokenize_drops_stopwords_and_keeps_numbers():
    assert tokenize("What is the value of k1 in BM25?") == ["valu", "k1", "bm25"]


DOC = """# Guide

Intro sentence one. Intro sentence two.

## Setup

Install the package first. Then configure the server. Then start it. Check the logs afterwards.

## Usage

Run the command. Read the output.
"""


def test_chunks_follow_headings_and_never_cross_them():
    chunks = chunk_pages([Page(None, DOC)], target=100)
    sections = {c.section for c in chunks}
    assert sections == {"Guide", "Guide > Setup", "Guide > Usage"}
    setup = [c for c in chunks if c.section == "Guide > Setup"]
    assert all("Run the command" not in c.text for c in setup)
    assert [c.ord for c in chunks] == list(range(len(chunks)))


def test_consecutive_chunks_overlap_by_one_sentence():
    text = " ".join(f"Sentence number {i} is here." for i in range(30))
    chunks = chunk_pages([Page(None, text)], target=150, overlap=1)
    assert len(chunks) > 3
    for prev, nxt in zip(chunks, chunks[1:]):
        assert prev.text.split(". ")[-1].rstrip(".") in nxt.text  
    assert max(len(c.text) for c in chunks) <= 150 * 2


def test_no_overlap_option_and_no_empty_chunks():
    text = " ".join(f"Sentence number {i} is here." for i in range(30))
    chunks = chunk_pages([Page(None, text)], target=150, overlap=0)
    assert all(c.text.strip() for c in chunks)
    joined = " ".join(c.text for c in chunks)
    assert joined.count("Sentence number 7 is here.") == 1


def test_chunks_never_span_pages_so_page_numbers_are_exact():
    chunks = chunk_pages([Page(1, "First page text here."), Page(2, "Second page text here.")], target=100)
    assert [(c.page, c.text) for c in chunks] == [(1, "First page text here."), (2, "Second page text here.")]
    chunks = chunk_pages([Page(1, "A " * 80 + "end."), Page(2, "B " * 80 + "end.")], target=100)
    assert all(("B" in c.text) == (c.page == 2) for c in chunks)  


def test_sentence_without_punctuation_is_split():
    chunks = chunk_pages([Page(None, "word " * 400)], target=200)
    assert len(chunks) > 5 and all(len(c.text) <= 400 for c in chunks)


def test_target_too_small_is_rejected():
    with pytest.raises(ValueError):
        chunk_pages([Page(None, "x")], target=10)


def test_bm25_prefers_documents_with_rare_matching_terms():
    corpus = [tokenize(t) for t in ["cats and dogs", "dogs dogs dogs", "quantum computing basics"]]
    scores = BM25(corpus).scores(tokenize("quantum dogs"))
    assert max(scores, key=scores.get) == 2  
    assert BM25(corpus).scores(tokenize("nothing matches here")) == {}


@pytest.fixture
def folder(tmp_path: Path) -> Path:
    d = tmp_path / "docs"
    d.mkdir()
    (d / "a.md").write_text("# A\n\nAlpha facts live here. They are about apples.")
    (d / "b.txt").write_text("Bravo notes. They discuss bananas and bread.")
    return d


def test_incremental_indexing(folder: Path, tmp_path: Path):
    store = Store(tmp_path / "t.db")
    first = index_folder(store, folder)
    assert (first.added, first.unchanged, first.documents) == (2, 0, 2)

    again = index_folder(store, folder)
    assert (again.added, again.updated, again.unchanged) == (0, 0, 2)

    (folder / "a.md").write_text("# A\n\nAlpha facts changed. Now about avocados.")
    (folder / "c.md").write_text("Charlie is new here.")
    (folder / "b.txt").unlink()
    third = index_folder(store, folder)
    assert (third.added, third.updated, third.removed, third.unchanged) == (1, 1, 1, 0)
    assert {d["name"] for d in store.list_docs()} == {"a.md", "c.md"}
    assert "avocados" in " ".join(c.text for c in store.all_chunks())


def test_bad_files_are_reported_not_fatal(folder: Path, tmp_path: Path):
    (folder / "empty.md").write_text("   \n\n")
    (folder / "ignored.exe").write_bytes(b"MZ")
    report = index_folder(Store(tmp_path / "t.db"), folder)
    assert report.documents == 2
    assert any("empty.md" in e for e in report.errors) and not any("ignored" in e for e in report.errors)


def test_symlink_pointing_outside_is_skipped(folder: Path, tmp_path: Path):
    secret = tmp_path / "secret.txt"
    secret.write_text("top secret")
    try:
        (folder / "link.txt").symlink_to(secret)
    except OSError:
        pytest.skip("cannot create symlinks on this system")
    report = index_folder(Store(tmp_path / "t.db"), folder)
    assert any("outside" in e for e in report.errors)
    assert "link.txt" not in {d["name"] for d in Store(tmp_path / "t.db").list_docs()}


def test_pdf_loading_keeps_page_numbers(folder: Path, tmp_path: Path):
    reportlab = pytest.importorskip("reportlab.pdfgen.canvas")
    buf = io.BytesIO()
    pdf = reportlab.Canvas(buf)
    pdf.drawString(72, 700, "Page one talks about giraffes.")
    pdf.showPage()
    pdf.drawString(72, 700, "Page two talks about zebras.")
    pdf.save()
    pages = load_bytes(".pdf", buf.getvalue())
    assert [p.number for p in pages] == [1, 2] and "zebras" in pages[1].text

    (folder / "animals.pdf").write_bytes(buf.getvalue())
    store = Store(tmp_path / "t.db")
    index_folder(store, folder)
    hit = Retriever(store).search("zebras", k=1)[0].chunk
    assert (hit.doc, hit.page) == ("animals.pdf", 2)


def test_unreadable_pdf_raises_loader_error():
    with pytest.raises(LoaderError):
        load_bytes(".pdf", b"this is not a pdf")


def test_bm25_search_and_empty_result_for_unrelated_query(folder: Path, tmp_path: Path):
    store = Store(tmp_path / "t.db")
    index_folder(store, folder)
    r = Retriever(store)
    assert r.search("bananas", k=3)[0].chunk.doc == "b.txt"
    assert r.search("zzzz qqqq", k=3) == []
    assert r.resolve_method("auto") == "bm25"
    with pytest.raises(ValueError):
        r.search("bananas", method="dense")  


def test_hybrid_uses_both_rankers_and_vectors_survive_reload(folder: Path, tmp_path: Path):
    store = Store(tmp_path / "t.db")
    emb = HashingEmbedder()
    index_folder(store, folder, emb)
    assert store.stats()["embedding_models"] == [emb.name]
    r = Retriever(store, emb, dense_min=0.0)
    assert r.has_dense and r.resolve_method("auto") == "hybrid"
    for method in ("bm25", "dense", "hybrid"):
        assert r.search("apples", k=2, method=method)[0].chunk.doc == "a.md"
    top = r.search("apples", k=2, method="hybrid")[0]
    assert top.score == pytest.approx(2 / 61)  


def test_vectors_are_backfilled_when_an_embedder_is_added_later(folder: Path, tmp_path: Path):
    store = Store(tmp_path / "t.db")
    index_folder(store, folder)
    assert store.stats()["embedding_models"] == []
    index_folder(store, folder, HashingEmbedder())  
    assert len(store.load_vectors(HashingEmbedder().name)) == store.stats()["chunks"]


def test_openai_compat_embedder_batches_and_normalizes():
    calls = []

    class FakeEmbeddings:
        def create(self, model, input):
            calls.append(len(input))
            return SimpleNamespace(data=[SimpleNamespace(embedding=[3.0, 4.0]) for _ in input])

    emb = OpenAICompatEmbedder("m", client=SimpleNamespace(embeddings=FakeEmbeddings()))
    out = emb.embed(["x"] * 70)
    assert calls == [64, 6] and out[0] == pytest.approx([0.6, 0.8])
