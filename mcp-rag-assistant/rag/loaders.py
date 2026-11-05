from __future__ import annotations

import io
from dataclasses import dataclass

SUPPORTED = frozenset({".txt", ".md", ".markdown", ".pdf"})


class LoaderError(Exception):
    """A file could not be turned into text."""


@dataclass
class Page:
    number: int | None  
    text: str


def load_bytes(suffix: str, data: bytes) -> list[Page]:
    suffix = suffix.lower()
    if suffix in {".txt", ".md", ".markdown"}:
        return [Page(None, data.decode("utf-8-sig", errors="replace"))]
    if suffix == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError as exc:  
            raise LoaderError("PDF support needs the 'pypdf' package (pip install pypdf)") from exc
        try:
            reader = PdfReader(io.BytesIO(data))
            pages = [Page(i, (page.extract_text() or "")) for i, page in enumerate(reader.pages, start=1)]
        except Exception as exc:
            raise LoaderError(f"could not read PDF: {exc}") from exc
        if not any(p.text.strip() for p in pages):
            raise LoaderError("PDF has no extractable text (is it a scanned image?)")
        return pages
    raise LoaderError(f"unsupported file type: {suffix}")
