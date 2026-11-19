from __future__ import annotations

from datetime import date

from .common import CURRENCIES, FIELDS, MONEY_FIELDS, normalize_record


def validate(rec: dict) -> list[str]:
    """Return a list of problems (empty list = looks consistent)."""
    n = normalize_record(rec)
    problems = [f"missing {f}" for f in FIELDS if n.get(f) in (None, "")]
    sub, tax, tot = n["subtotal"], n["tax"], n["total"]
    if None not in (sub, tax, tot) and abs(sub + tax - tot) > 0.02:
        problems.append(f"subtotal {sub} + tax {tax} != total {tot}")
    for f in MONEY_FIELDS:
        if n[f] is not None and n[f] < 0:
            problems.append(f"negative {f}")
    if n["date"] and not (2000 <= int(n["date"][:4]) <= date.today().year + 1):
        problems.append("date out of range")
    if n["currency"] and n["currency"] not in CURRENCIES:
        problems.append(f"unknown currency {n['currency']}")
    if n["invoice_number"] and len(n["invoice_number"]) < 3:
        problems.append("invoice number too short")
    return problems


def _digits(x: float | None) -> str:
    return "" if x is None else f"{x:.2f}".replace(".", "")


def _edit_distance(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def repair_amounts(rec: dict, max_distance: int = 2) -> tuple[dict, str | None]:
    n = normalize_record(rec)
    vals = {f: n[f] for f in MONEY_FIELDS}
    if None in vals.values() or abs(vals["subtotal"] + vals["tax"] - vals["total"]) <= 0.02:
        return rec, None
    implied = {"subtotal": round(vals["total"] - vals["tax"], 2), "tax": round(vals["total"] - vals["subtotal"], 2),
               "total": round(vals["subtotal"] + vals["tax"], 2)}
    items = rec.get("_items_sum")
    culprit = None
    if items and abs(items + vals["tax"] - vals["total"]) <= 0.02:
        culprit, new = "subtotal", round(items, 2)
    else:
        pool = [f for f in MONEY_FIELDS if implied[f] >= 0]
        if items and abs(vals["subtotal"] - items) <= 0.02:
            pool = [f for f in pool if f != "subtotal"]  
        scored = sorted((_edit_distance(_digits(vals[f]), _digits(implied[f])), f) for f in pool)
        scored = [x for x in scored if x[0] <= max_distance]
        if scored:
            best = [x for x in scored if x[0] == scored[0][0]]
            if len(best) > 1:
                best = [x for x in best if len(_digits(vals[x[1]])) > len(_digits(implied[x[1]]))]
            if len(best) == 1:
                culprit, new = best[0][1], implied[best[0][1]]
    if culprit is None:
        return rec, None
    fixed = dict(rec)
    fixed[culprit] = new
    return fixed, f"repaired {culprit}: {vals[culprit]} -> {new}"
