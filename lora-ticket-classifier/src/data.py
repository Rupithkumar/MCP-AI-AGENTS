from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"
DATASETS = ("mteb/banking77", "PolyAI/banking77")  


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")


def load_hf() -> tuple[list[dict], list[dict], str]:
    """Banking77 from the Hugging Face hub. Returns (train, test, dataset_name)."""
    from datasets import load_dataset

    errors = []
    for name in DATASETS:
        try:
            ds = load_dataset(name)
        except Exception as exc:  
            errors.append(f"{name}: {type(exc).__name__}: {str(exc)[:150]}")
            continue
        names = ds["train"].features["label"].names if hasattr(ds["train"].features["label"], "names") else None

        def convert(split) -> list[dict]:
            rows = []
            for r in split:
                label = r.get("label_text") or (names[r["label"]] if names else str(r["label"]))
                rows.append({"text": r["text"], "label": label})
            return rows

        return convert(ds["train"]), convert(ds["test"]), name
    raise SystemExit(
        "Could not load Banking77 from the Hugging Face hub:\n  " + "\n  ".join(errors) +
        "\nCheck your internet connection, or use your own data: python -m src.prepare_data --data-dir <folder with train.csv and test.csv>"
    )


def load_csv_dir(folder: Path) -> tuple[list[dict], list[dict], str]:
    def read(name: str) -> list[dict]:
        path = folder / name
        if not path.exists():
            raise SystemExit(f"{path} not found. Expected train.csv and test.csv with the columns: text,label")
        with path.open(encoding="utf-8", newline="") as f:
            rows = [{"text": r["text"].strip(), "label": r["label"].strip()} for r in csv.DictReader(f)]
        if not rows:
            raise SystemExit(f"{path} has no rows")
        return rows

    return read("train.csv"), read("test.csv"), f"csv:{folder.name}"


def choose_labels(train: list[dict], num_labels: int, seed: int) -> list[str]:
    available = sorted({r["label"] for r in train})
    if num_labels > len(available):
        raise SystemExit(f"Asked for {num_labels} labels but the data has only {len(available)}")
    return sorted(random.Random(seed).sample(available, num_labels))


def stratified_cap(rows: list[dict], per_label: int, seed: int) -> list[dict]:
    if not per_label:
        return rows
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        groups[r["label"]].append(r)
    rng, out = random.Random(seed), []
    for label in sorted(groups):
        g = groups[label][:]
        rng.shuffle(g)
        out += g[:per_label]
    return out


def split_train_val(rows: list[dict], val_fraction: float, seed: int) -> tuple[list[dict], list[dict]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        groups[r["label"]].append(r)
    rng, train, val = random.Random(seed), [], []
    for label in sorted(groups):
        g = groups[label][:]
        rng.shuffle(g)
        k = max(1, round(len(g) * val_fraction)) if len(g) > 1 else 0
        val += g[:k]
        train += g[k:]
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


def pick_shots(train: list[dict], labels: list[str], seed: int) -> list[dict]:
    by_label: dict[str, list[dict]] = defaultdict(list)
    for r in train:
        by_label[r["label"]].append(r)
    rng = random.Random(seed + 1)
    return [rng.choice(by_label[label]) for label in labels]


def prepare(train_raw: list[dict], test_raw: list[dict], source: str, num_labels: int, seed: int = 42,
            val_fraction: float = 0.1, max_per_label: int = 0, test_size: int = 1000, out: Path = PROCESSED) -> dict:
    labels = choose_labels(train_raw, num_labels, seed)
    keep = set(labels)
    train_all = stratified_cap([r for r in train_raw if r["label"] in keep], max_per_label, seed)
    test_all = [r for r in test_raw if r["label"] in keep]
    random.Random(seed).shuffle(test_all)
    test = test_all[:test_size] if test_size else test_all
    train, val = split_train_val(train_all, val_fraction, seed)
    shots = pick_shots(train, labels, seed)
    meta = {"source": source, "seed": seed, "labels": labels, "n_train": len(train), "n_val": len(val), "n_test": len(test)}
    write_jsonl(out / "train.jsonl", train)
    write_jsonl(out / "val.jsonl", val)
    write_jsonl(out / "test.jsonl", test)
    write_jsonl(out / "shots.jsonl", shots)
    (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def load_prepared(folder: Path = PROCESSED) -> dict:
    if not (folder / "meta.json").exists():
        raise SystemExit(f"No prepared data in {folder}. Run: python -m src.prepare_data")
    meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
    return {
        "meta": meta, "labels": meta["labels"],
        "train": read_jsonl(folder / "train.jsonl"), "val": read_jsonl(folder / "val.jsonl"),
        "test": read_jsonl(folder / "test.jsonl"), "shots": read_jsonl(folder / "shots.jsonl"),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Download and split the data")
    p.add_argument("--num-labels", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-per-label", type=int, default=0, help="cap training rows per label (0 = all)")
    p.add_argument("--test-size", type=int, default=1000, help="rows in the test set (0 = all)")
    p.add_argument("--data-dir", type=Path, help="folder with train.csv and test.csv instead of Banking77")
    args = p.parse_args()
    train, test, source = load_csv_dir(args.data_dir) if args.data_dir else load_hf()
    meta = prepare(train, test, source, args.num_labels, args.seed, max_per_label=args.max_per_label, test_size=args.test_size)
    print(f"source={meta['source']}  labels={len(meta['labels'])}  train={meta['n_train']}  val={meta['n_val']}  test={meta['n_test']}")
    print("labels:", ", ".join(meta["labels"]))
    print(f"saved to {PROCESSED}")


if __name__ == "__main__":
    main()
