from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, field

from .common import CURRENCIES, FIELDS, parse_amount, parse_date
from .validate import repair_amounts, validate

SYMBOL_TO_CODE = {sym: code for code, sym in CURRENCIES.items()}
AMOUNT = re.compile(r"[€£₹$]?\s?\d[\d.,\s]*\d|\d")
DATE_PATTERNS = [
    r"\d{4}-\d{1,2}-\d{1,2}", r"\d{1,2}[\s\-/.]+[A-Za-z]{3,9}[\s\-/.,]+\d{4}", r"[A-Za-z]{3,9}\.? \d{1,2},? \d{4}", r"\d{1,2}[/.]\d{1,2}[/.]\d{4}",
]
NUMBER_PATTERNS = [
    r"(?:invoice\s*(?:no\.?|number|num|#)|inv\.?\s*(?:no\.?|#))\s*[:.;#]*\s*([A-Z0-9][A-Z0-9\-]{3,})",
    r"\b([A-Z]{1,3}-\d{4,7})\b",
]


@dataclass
class Result:
    record: dict
    llm_called: bool = False
    error: str | None = None
    problems: list[str] = field(default_factory=list)


def last_amount(line: str) -> float | None:
    found = AMOUNT.findall(line)
    for token in reversed(found):
        value = parse_amount(token)
        if value is not None:
            return value
    return None


class RuleExtractor:
    name = "rules"

    def extract(self, text: str) -> Result:
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        rec = {
            "vendor": self._vendor(lines), "invoice_number": self._number(text), "date": self._date(text),
            "currency": self._currency(text),
            "subtotal": self._money(lines, r"sub\s*-?\s*total"),
            "tax": self._money(lines, r"\b(?:tax|vat|gst)\b"),
            "total": self._total(lines),
        }
        rec["_items_sum"] = self._items_sum(lines)  
        return Result(rec)

    @staticmethod
    def _items_sum(lines: list[str]) -> float | None:
        """Sum of the amounts on the line-item rows (between the table header and the subtotal)."""
        start = next((i for i, ln in enumerate(lines) if re.search(r"description", ln, re.I)), None)
        end = next((i for i, ln in enumerate(lines) if re.search(r"sub\s*-?\s*total", ln, re.I)), None)
        if start is None or end is None or end <= start + 1:
            return None
        amounts = [last_amount(ln) for ln in lines[start + 1 : end] if re.search(r"\d[.,]\d{2}\b", ln)]
        amounts = [a for a in amounts if a is not None]
        return round(sum(amounts), 2) if amounts else None

    @staticmethod
    def _vendor(lines: list[str]) -> str | None:
        for ln in lines[:4]:
            cleaned = re.sub(r"\binvoice\b", "", ln, flags=re.I).strip(" -|:")
            if len(re.findall(r"[A-Za-z]", cleaned)) >= 4:
                return cleaned
        return None

    @staticmethod
    def _number(text: str) -> str | None:
        for pat in NUMBER_PATTERNS:
            m = re.search(pat, text, flags=re.I if pat is NUMBER_PATTERNS[0] else 0)
            if m:
                return m.group(1)
        return None

    @staticmethod
    def _date(text: str) -> str | None:
        for pat in DATE_PATTERNS:
            for m in re.finditer(pat, text):
                iso = parse_date(m.group(0))
                if iso:
                    return iso
        return None

    @staticmethod
    def _currency(text: str) -> str | None:
        votes = Counter(SYMBOL_TO_CODE[m.group(1)] for m in re.finditer(r"([€£₹$])\s?\d", text))  
        if votes: 
            return votes.most_common(1)[0][0]
        m = re.search(r"\b(USD|EUR|GBP|INR)\b", text)
        if m:
            return m.group(1)
        return "INR" if re.search(r"\bRs\.?\s?\d", text) else None

    @staticmethod
    def _money(lines: list[str], pattern: str) -> float | None:
        for ln in lines:
            if re.search(pattern, ln, flags=re.I):
                value = last_amount(ln)
                if value is not None:
                    return value
        return None

    @staticmethod
    def _total(lines: list[str]) -> float | None:
        pat = re.compile(r"(grand\s+total|total\s+due|amount\s+due|balance\s+due|\btotal\b)", re.I)
        for ln in reversed(lines):  # the real total is the last such line
            if pat.search(ln) and not re.search(r"sub\s*-?\s*total", ln, re.I):
                value = last_amount(ln)
                if value is not None:
                    return value
        return None


class RuleRepairExtractor:
    name = "rules_repair"

    def __init__(self):
        self.rules = RuleExtractor()

    def extract(self, text: str) -> Result:
        first = self.rules.extract(text)
        fixed, note = repair_amounts(first.record)
        return Result(fixed, problems=[note] if note else [])


PROMPT = """Extract these fields from the invoice text below (OCR output, it may contain small reading errors).
Return ONLY a JSON object with keys: vendor, invoice_number, date (YYYY-MM-DD), currency (USD, EUR, GBP or INR),
subtotal, tax, total (numbers, no currency symbols). Use null for anything you cannot find. Do not guess.

Invoice text:
{text}"""


class LLMExtractor:
    name = "llm"

    def __init__(self, client, model: str):
        self.client, self.model = client, model

    def extract(self, text: str) -> Result:
        try:
            resp = self.client.chat.completions.create(
                model=self.model, temperature=0, messages=[{"role": "user", "content": PROMPT.format(text=text)}])
            raw = resp.choices[0].message.content or ""
            obj = parse_json_object(raw)
            if obj is None:
                return Result({f: None for f in FIELDS}, True, "no JSON in answer")
            return Result({f: obj.get(f) for f in FIELDS}, True)
        except Exception as exc: 
            return Result({f: None for f in FIELDS}, True, f"{type(exc).__name__}: {str(exc)[:100]}")


def parse_json_object(raw: str) -> dict | None:
    raw = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.M).strip()
    start, end = raw.find("{"), raw.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        obj = json.loads(raw[start : end + 1])
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


class Cascade:
    name = "cascade"

    def __init__(self, llm: LLMExtractor):
        self.rules, self.llm = RuleRepairExtractor(), llm

    def extract(self, text: str) -> Result:
        first = self.rules.extract(text)
        problems = validate(first.record)
        if not problems:
            return Result(first.record)
        second = self.llm.extract(text)
        if second.error: 
            return Result(first.record, True, second.error, problems)
        return Result(second.record, True)
