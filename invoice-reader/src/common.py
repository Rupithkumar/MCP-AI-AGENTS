from __future__ import annotations

import re
from datetime import date, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FIELDS = ("vendor", "invoice_number", "date", "currency", "subtotal", "tax", "total")
MONEY_FIELDS = ("subtotal", "tax", "total")
CURRENCIES = {"USD": "$", "EUR": "€", "GBP": "£", "INR": "₹"}
MONTHS = {m: i for i, m in enumerate(["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], 1)}


def parse_amount(text: str | float | int | None) -> float | None:
    """'1,234.50', '1.234,50', '1 234,50', '$99', '₹ 12,345.00' -> float. None if there are no digits."""
    if text is None:
        return None
    if isinstance(text, (int, float)):
        return round(float(text), 2)
    s = re.sub(r"[^\d.,\s-]", "", str(text)).strip().replace(" ", "")
    if not re.search(r"\d", s):
        return None
    neg = s.startswith("-")
    s = s.lstrip("-")
    if "," in s and "." in s:  
        dec = "," if s.rfind(",") > s.rfind(".") else "."
        thou = "." if dec == "," else ","
        s = s.replace(thou, "").replace(dec, ".")
    elif "," in s or "." in s:
        mark = "," if "," in s else "."
        parts = s.split(mark)
        if len(parts) == 2 and len(parts[1]) != 3:  
            s = parts[0] + "." + parts[1]
        else:
            s = "".join(parts)
    try:
        return round(-float(s) if neg else float(s), 2)
    except ValueError:
        return None


def parse_date(text: str | None) -> str | None:
    """Return ISO 'YYYY-MM-DD' or None. Numeric dd/mm/yyyy is read day-first (documented convention)."""
    if not text:
        return None
    s = re.sub(r"\s+", " ", str(text).strip().strip(".,")).lower()
    m = re.search(r"(\d{4})-(\d{1,2})-(\d{1,2})", s)
    if m:
        return _iso(int(m[1]), int(m[2]), int(m[3]))
    m = re.search(r"(\d{1,2})[\s\-/.]+([a-z]{3})[a-z]*[\s\-/.,]+(\d{4})", s)  
    if m and m[2] in MONTHS:
        return _iso(int(m[3]), MONTHS[m[2]], int(m[1]))
    m = re.search(r"([a-z]{3})[a-z]*\.? (\d{1,2}),? (\d{4})", s)  
    if m and m[1] in MONTHS:
        return _iso(int(m[3]), MONTHS[m[1]], int(m[2]))
    m = re.search(r"(\d{1,2})[/.](\d{1,2})[/.](\d{4})", s)
    if m:
        return _iso(int(m[3]), int(m[2]), int(m[1]))
    return None


def _iso(y: int, mo: int, d: int) -> str | None:
    try:
        return date(y, mo, d).isoformat()
    except ValueError:
        return None


def norm_text(s: str | None) -> str:
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


def normalize_record(rec: dict) -> dict:
    out = {
        "vendor": norm_text(rec.get("vendor")),
        "invoice_number": norm_text(rec.get("invoice_number")),
        "date": parse_date(rec.get("date")),
        "currency": (str(rec.get("currency") or "").strip().upper() or None),
    }
    for f in MONEY_FIELDS:
        out[f] = parse_amount(rec.get(f))
    return out


def field_correct(field: str, truth: dict, pred: dict) -> bool:
    t, p = normalize_record(truth)[field], normalize_record(pred)[field]
    if t is None or p is None:
        return False
    return abs(t - p) < 0.005 if field in MONEY_FIELDS else t == p
