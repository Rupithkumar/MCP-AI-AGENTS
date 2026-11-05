"""Pure scoring functions (no I/O, no model calls) so they are easy to unit-test.

Fact checks are deliberately simple: each question lists "fact groups"; a group is satisfied if
ANY of its phrases appears in the text, and the answer passes if ALL groups are satisfied.
Phrases match from a word boundary, so "chunk" also matches "chunking" but not "unchunked".
This is cheap and repeatable but brittle: a correct answer in unexpected words can fail.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


@dataclass(frozen=True)
class Question:
    id: str
    question: str
    sources: list[str]
    facts: list[list[str]]
    answerable: bool
    tier: str = "easy"  # 'hard' = paraphrased, little word overlap with the documents


def load_questions(path: str | Path) -> list[Question]:
    questions = []
    for n, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            d = json.loads(line)
            questions.append(Question(d["id"], d["question"], d["sources"], d["facts"], d["answerable"], d.get("tier", "easy")))
        except (KeyError, json.JSONDecodeError) as exc:
            raise ValueError(f"{path}:{n}: bad dataset line ({exc})") from exc
    return questions


# ----------------------------------------------------------------- text checks
def norm(text: str) -> str:
    text = text.lower().replace("’", "'")
    text = re.sub("[‐‑‒–—―−]", "-", text)  # every dash and hyphen variant
    return re.sub(r"\s+", " ", text)  # \s also covers non-breaking and narrow spaces


def has_phrase(text: str, phrase: str) -> bool:
    return re.search(r"(?<![a-z0-9])" + re.escape(norm(phrase)), norm(text)) is not None


def facts_satisfied(text: str, groups: Sequence[Sequence[str]]) -> bool:
    return all(any(has_phrase(text, p) for p in group) for group in groups)


ABSTAIN = re.compile(
    r"(could(?:n't| not) find|can(?:'t|not) find|unable to find|did(?:n't| not) find"
    r"|not (?:found|covered|mentioned|addressed|specified|contained|available)"
    r"|no (?:relevant |specific )?(?:information|passages?|mention|details?)"
    r"|(?:don't|do not) know|(?:doesn't|does not|do not|don't) (?:contain|cover|mention|include|say|specify|address)"
    r"|(?:isn't|is not|aren't|are not) (?:covered|mentioned|addressed|included|specified))",
    re.IGNORECASE,
)


def is_abstention(answer: str) -> bool:
    return ABSTAIN.search(norm(answer)) is not None


# ------------------------------------------------------------------ citations
def extract_citations(answer: str) -> list[str]:
    """Citations like [file.md#3] or [a.md#1, b.md#2], in order of appearance.

    Some models (gpt-oss, for one) write the same citation in full-width brackets, 【file.md#3】.
    Only the bracket style is tolerated: the id inside must still be exactly file#number.
    """
    found = []
    for bracket in re.finditer(r"[\[【]([^\[\]【】]+)[\]】]", answer):
        for part in re.split(r"[,;]", bracket.group(1)):
            part = part.strip()
            if re.fullmatch(r"[\w./-]+#\d+", part):
                found.append(part)
    return found


SEEN_CITATION = re.compile(r"^\[\d+\] (\S+#\d+) \|", re.MULTILINE)


def seen_citations(tool_result_text: str) -> set[str]:
    """Citation ids that a search tool actually returned (what the model was allowed to cite)."""
    return set(SEEN_CITATION.findall(tool_result_text))


# ------------------------------------------------------------ retrieval ranks
def first_relevant_rank(ranked_docs: Sequence[str], sources: Iterable[str]) -> int | None:
    wanted = set(sources)
    for rank, doc in enumerate(ranked_docs, start=1):
        if doc in wanted:
            return rank
    return None


def hit_at_k(rank: int | None, k: int) -> bool:
    return rank is not None and rank <= k


def reciprocal_rank(rank: int | None) -> float:
    return 1.0 / rank if rank else 0.0


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0