from __future__ import annotations

import re
from dataclasses import dataclass

from .loaders import Page

_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[`])")


@dataclass
class Chunk:
    ord: int
    text: str
    section: str
    page: int | None


def _split_long(sentence: str, limit: int) -> list[str]:
    if len(sentence) <= limit:
        return [sentence]
    pieces: list[str] = []
    current: list[str] = []
    size = 0
    for word in sentence.split():
        if size + len(word) + 1 > limit and current:
            pieces.append(" ".join(current))
            current, size = [], 0
        current.append(word)
        size += len(word) + 1
    if current:
        pieces.append(" ".join(current))
    return pieces


def chunk_pages(pages: list[Page], target: int = 700, overlap: int = 1) -> list[Chunk]:
    if target < 100:
        raise ValueError("target must be at least 100 characters")

    chunks: list[Chunk] = []
    headings: list[tuple[int, str]] = []
    buf: list[str] = []
    buf_len = 0
    new = 0  
    start_page: int | None = None

    def flush(carry: bool) -> None:
        nonlocal buf, buf_len, new, start_page
        if new:
            section = " > ".join(title for _, title in headings)
            chunks.append(Chunk(len(chunks), " ".join(buf).strip(), section, start_page))
        if carry and overlap and new:
            buf = buf[-overlap:]
            buf_len = sum(len(s) + 1 for s in buf)
        else:
            buf, buf_len = [], 0
        new, start_page = 0, None

    def add(sentence: str, page: int | None) -> None:
        nonlocal buf_len, new, start_page
        for piece in _split_long(sentence, target):
            if new and buf_len + len(piece) + 1 > target:
                flush(carry=True)
            if new == 0:
                start_page = page
            buf.append(piece)
            buf_len += len(piece) + 1
            new += 1

    def end_paragraph(lines: list[str], page: int | None) -> None:
        if not lines:
            return
        text = " ".join(lines)
        lines.clear()
        for sentence in _SENTENCE_SPLIT.split(text):
            sentence = sentence.strip()
            if sentence:
                add(sentence, page)

    for page in pages:
        paragraph: list[str] = []
        for raw in page.text.splitlines():
            line = raw.strip()
            match = _HEADING.match(line)
            if match:
                end_paragraph(paragraph, page.number)
                flush(carry=False)
                level = len(match.group(1))
                while headings and headings[-1][0] >= level:
                    headings.pop()
                headings.append((level, match.group(2)))
            elif not line:
                end_paragraph(paragraph, page.number)
            else:
                paragraph.append(line)
        end_paragraph(paragraph, page.number)
        flush(carry=False)  
    return chunks
