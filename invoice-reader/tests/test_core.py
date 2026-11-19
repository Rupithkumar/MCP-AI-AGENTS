import json
import pytest

from src.common import field_correct, normalize_record, parse_amount, parse_date
from src.extract import Cascade, LLMExtractor, RuleExtractor, RuleRepairExtractor, parse_json_object
from src.validate import repair_amounts, validate


@pytest.mark.parametrize("text,value", [("1,234.50", 1234.5), ("1.234,50", 1234.5), ("1 234,50", 1234.5), ("₹ 12,345.00", 12345.0),
                                        ("1,234", 1234.0), ("12,50", 12.5), ("$99", 99.0), ("abc", None), (None, None), (12, 12.0)])
def test_parse_amount(text, value):
    assert parse_amount(text) == value


@pytest.mark.parametrize("text,iso", [("12 Mar 2025", "2025-03-12"), ("March 12, 2025", "2025-03-12"), ("12-Mar-2025", "2025-03-12"),
                                      ("2025-03-12", "2025-03-12"), ("31/12/2025", "2025-12-31"), ("32/13/2025", None), ("nonsense", None)])
def test_parse_date(text, iso):
    assert parse_date(text) == iso


def test_comparison_is_forgiving_about_format_only():
    truth = {"vendor": "Acme Ltd.", "invoice_number": "INV-1", "date": "2025-01-02", "currency": "usd", "subtotal": 10, "tax": 1, "total": 11}
    pred = {"vendor": "ACME LTD", "invoice_number": "inv1", "date": "2 Jan 2025", "currency": "USD", "subtotal": "10.00", "tax": "1,00", "total": 11.0}
    assert all(field_correct(f, truth, pred) for f in truth)
    assert not field_correct("total", truth, {**pred, "total": 12})
    assert not field_correct("total", truth, {**pred, "total": None})


def test_validate_flags_arithmetic_and_missing():
    ok = {"vendor": "A B", "invoice_number": "INV-1", "date": "2025-01-02", "currency": "USD", "subtotal": 10, "tax": 1, "total": 11}
    assert validate(ok) == []
    assert any("!=" in p for p in validate({**ok, "total": 99}))
    assert any("missing currency" in p for p in validate({**ok, "currency": None}))
    assert any("unknown currency" in p for p in validate({**ok, "currency": "XYZ"}))


def test_repair_cases():
    fixed, note = repair_amounts({"subtotal": 76083.0, "tax": 729.96, "total": 6812.96})  
    assert fixed["subtotal"] == 6083.0 and "repaired subtotal" in note
    fixed, _ = repair_amounts({"subtotal": 35.23, "tax": 448.23, "total": 4183.46, "_items_sum": 3735.23})  
    assert repair_amounts({"subtotal": 100, "tax": 10, "total": 110}) == ({"subtotal": 100, "tax": 10, "total": 110}, None)
    rec = {"subtotal": 100, "tax": 50, "total": 999} 
    assert repair_amounts(rec) == (rec, None)


INVOICE = """ACME TOOLS INVOICE
Invoice No: AC-12345
Date: March 12, 2025
Description Qty Price Amount
Widget 2 £10.00 £20.00
Gadget 1 £30.50 £30.50
Subtotal £50.50
VAT (20%) £10.10
Total Due £60.60
"""


def test_rules_on_clean_text():
    rec = RuleExtractor().extract(INVOICE).record
    assert rec["vendor"] == "ACME TOOLS" and rec["invoice_number"] == "AC-12345" and rec["date"] == "2025-03-12"
    assert (rec["currency"], rec["subtotal"], rec["tax"], rec["total"]) == ("GBP", 50.5, 10.1, 60.6)
    assert rec["_items_sum"] == 50.5 and validate(rec) == []


def test_currency_is_a_majority_vote_not_first_match():
    text = INVOICE.replace("ACME TOOLS INVOICE", "ACME $ TOOLS INVOICE")
    assert RuleExtractor().extract(text).record["currency"] == "GBP"


class FakeClient:
    def __init__(self, content=None, fail=False):
        self.calls, self.content, self.fail = 0, content, fail
        self.chat = type("Chat", (), {"completions": self})

    def create(self, **kw):
        self.calls += 1
        if self.fail:
            raise RuntimeError("rate limited")
        msg = type("M", (), {"content": self.content})
        return type("R", (), {"choices": [type("C", (), {"message": msg})]})


GOOD = json.dumps({"vendor": "ACME TOOLS", "invoice_number": "AC-12345", "date": "2025-03-12", "currency": "GBP",
                   "subtotal": 50.5, "tax": 10.1, "total": 60.6})


def test_cascade_skips_llm_when_rules_pass():
    client = FakeClient(GOOD)
    res = Cascade(LLMExtractor(client, "m")).extract(INVOICE)
    assert client.calls == 0 and not res.llm_called


def test_cascade_calls_llm_only_when_check_fails():
    broken = INVOICE.replace("Total Due £60.60", "Total Due £99.99")
    client = FakeClient("```json\n" + GOOD + "\n```")
    res = Cascade(LLMExtractor(client, "m")).extract(broken)
    assert client.calls == 1 and res.llm_called and res.record["total"] == 60.6 and validate(res.record) == []


def test_cascade_keeps_rule_answer_and_flags_when_llm_fails():
    broken = INVOICE.replace("Total Due £60.60", "Total Due £99.99")
    res = Cascade(LLMExtractor(FakeClient(fail=True), "m")).extract(broken)
    assert res.error and res.record["total"] == 99.99 and res.problems


def test_json_parsing_is_tolerant():
    assert parse_json_object('text {"a": 1} more') == {"a": 1}
    assert parse_json_object("no json") is None
    assert parse_json_object("{bad json}") is None
