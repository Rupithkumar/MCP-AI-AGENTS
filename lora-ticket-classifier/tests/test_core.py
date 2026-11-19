import json
import pytest
from src.metrics import parse_label, macro_f1, accuracy, summarize
from src.prompts import build_messages
from src.data import split_train_val, pick_shots
from src.train import encode_example, Collator

L = ["card_lost", "refund_status", "change_pin"]


def test_parse_label_variants():
    assert parse_label("card_lost", L) == "card_lost"
    assert parse_label("Card Lost.", L) == "card_lost"
    assert parse_label("label: change_pin\nextra", L) == "change_pin"
    assert parse_label("something else", L) is None
    assert parse_label("", L) is None


def test_metrics_and_invalid():
    s = summarize(["card_lost", "change_pin", "refund_status"], ["card_lost", "nonsense", "refund_status"], L)
    assert s["accuracy"] == pytest.approx(2 / 3)
    assert s["invalid_rate"] == pytest.approx(1 / 3)
    assert macro_f1(["a", "b"], ["a", "b"], ["a", "b"]) == 1.0


def test_prompt_styles():
    shots = [{"text": "x", "label": "card_lost"}]
    assert "card_lost" not in build_messages("hi", L, "short")[-1]["content"] + build_messages("hi", L, "short")[0]["content"]
    assert "card_lost" in json.dumps(build_messages("hi", L, "zero_shot"))
    with pytest.raises(ValueError):
        build_messages("hi", L, "few_shot")
    with pytest.raises(ValueError):
        build_messages("hi", L, "bogus")
    assert len(build_messages("hi", L, "few_shot", shots)) >= 2


def test_split_no_overlap(prepared):
    _, d = prepared
    tr, va, te = ({r["text"] + r["label"] for r in d[k]} for k in ("train", "val", "test"))
    assert len(d["val"]) > 0 and len(d["shots"]) == len(d["labels"])
    assert not any(r in d["train"] for r in d["val"])


def test_loss_masking(tiny_lm):
    _, tok = tiny_lm
    e = encode_example(tok, "lost card", "card_lost", ["card_lost", "change_pin"], "short", 128)
    supervised = [i for i, l in zip(e["input_ids"], e["labels"]) if l != -100]
    assert tok.decode(supervised).startswith("card_lost") and supervised[-1] == tok.eos_token_id
    assert e["labels"][0] == -100
    long = encode_example(tok, "lost " * 500, "card_lost", ["card_lost"], "short", 64)
    assert len(long["input_ids"]) <= 64 and long["labels"][-1] == tok.eos_token_id


def test_collator():
    b = Collator(0)([{"input_ids": [5, 6, 7], "labels": [-100, 6, 7]}, {"input_ids": [5], "labels": [-100]}])
    assert b["input_ids"].shape == (2, 3) and b["attention_mask"][1].tolist() == [1, 0, 0] and b["labels"][1].tolist() == [-100] * 3
