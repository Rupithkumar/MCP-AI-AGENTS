from __future__ import annotations

import argparse
import json
import random
from datetime import date, timedelta
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from .common import CURRENCIES, ROOT

OUT = ROOT / "data" / "invoices"
LEVELS = ("clean", "mild", "harsh")
VENDORS = ["Northwind Supplies", "Blue Harbor Logistics", "Kestrel Analytics", "Maple & Finch Studio", "Orion Cloud Services",
           "Greenfield Hardware", "Summit Print Works", "Lumen Electric Co", "Atlas Freight Ltd", "Pioneer Office Depot",
           "Redwood Consulting", "Silverline Telecom", "Harbor Point Foods", "Quartz Software", "Cedar Valley Clinics"]
ITEMS = ["Consulting hours", "Cloud hosting", "Printer toner", "Office chairs", "Freight charge", "Software license",
         "Network cable", "Maintenance plan", "Training session", "Laptop stand", "Paper reams", "Support retainer"]
TOTAL_LABELS = ["Total", "Amount Due", "Grand Total", "Total Due", "Balance Due"]
TAX_LABELS = ["Tax", "VAT", "GST", "Sales Tax"]
NUM_LABELS = ["Invoice No", "Invoice #", "Inv No.", "Invoice Number"]
DATE_LABELS = ["Date", "Invoice Date", "Issued", "Date of issue"]
FONT_PATHS = ["C:/Windows/Fonts/arial.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
              "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf", "/Library/Fonts/Arial.ttf"]
BOLD_PATHS = ["C:/Windows/Fonts/arialbd.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
              "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf", "/Library/Fonts/Arial Bold.ttf"]


def font(size: int, bold: bool = False):
    for path in (BOLD_PATHS if bold else FONT_PATHS):
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def money(value: float, style: str) -> str:
    s = f"{value:,.2f}"
    return s.replace(",", "X").replace(".", ",").replace("X", ".") if style == "eu" else s


def fmt_date(d: date, kind: int) -> str:
    return [d.isoformat(), d.strftime("%d %b %Y"), d.strftime("%B %d, %Y").replace(" 0", " "), d.strftime("%d-%b-%Y")][kind]


def make_truth(rng: random.Random, i: int) -> dict:
    cur = rng.choice(list(CURRENCIES))
    items = [(rng.choice(ITEMS), rng.randint(1, 12), round(rng.uniform(8, 480), 2)) for _ in range(rng.randint(2, 5))]
    subtotal = round(sum(q * p for _, q, p in items), 2)
    rate = rng.choice([0.05, 0.08, 0.12, 0.18, 0.20])
    tax = round(subtotal * rate, 2)
    d = date(2023, 1, 1) + timedelta(days=rng.randint(0, 1000))
    return {"id": f"inv_{i:04d}", "vendor": rng.choice(VENDORS), "invoice_number": f"{rng.choice(['INV', 'BL', 'KX', 'A'])}-{rng.randint(10000, 99999)}",
            "date": d.isoformat(), "currency": cur, "subtotal": subtotal, "tax": tax, "total": round(subtotal + tax, 2),
            "_items": items, "_rate": rate}


def rtext(dr: ImageDraw.ImageDraw, right: int, y: int, text: str, fnt, fill="black") -> None:
    """Draw text so that it ENDS at x=right (keeps long dates and big amounts inside the page)."""
    dr.text((right - dr.textlength(text, font=fnt), y), text, font=fnt, fill=fill)


def render(truth: dict, rng: random.Random) -> Image.Image:
    layout = rng.choice(["classic", "modern", "compact"])
    cur_sym, style = CURRENCIES[truth["currency"]], ("eu" if truth["currency"] == "EUR" and rng.random() < 0.6 else "us")
    if truth["currency"] == "INR" and rng.random() < 0.5:
        cur_sym = "Rs. " 
    sym = lambda v: f"{cur_sym}{money(v, style)}"  
    d = date.fromisoformat(truth["date"])
    if d.day > 12 and rng.random() < 0.3:
        date_text = d.strftime("%d/%m/%Y")
    else:
        date_text = fmt_date(d, rng.randint(0, 3))
    labels = {"num": rng.choice(NUM_LABELS), "date": rng.choice(DATE_LABELS), "tax": rng.choice(TAX_LABELS), "total": rng.choice(TOTAL_LABELS)}
    W, H = 850, 1100
    img = Image.new("RGB", (W, H), "white")
    dr = ImageDraw.Draw(img)
    f_s, f_m, f_b, f_xl = font(20), font(24), font(24, True), font(40, True)
    if layout == "modern":
        dr.rectangle([0, 0, W, 130], fill=(30, 50, 90))
        dr.text((40, 40), truth["vendor"], font=font(38, True), fill="white")
        rtext(dr, W - 40, 50, "INVOICE", f_b, "white")
        y = 170
        for k, v in ((labels["num"], truth["invoice_number"]), (labels["date"], date_text)):
            dr.text((40, y), f"{k}: {v}", font=f_m, fill="black")
            y += 40
        y += 30
    elif layout == "classic":
        dr.text((40, 40), truth["vendor"], font=f_xl, fill="black")
        dr.text((40, 95), "12 Market Street, Springfield", font=f_s, fill=(90, 90, 90))
        rtext(dr, W - 40, 45, "INVOICE", f_xl)
        rtext(dr, W - 40, 110, f"{labels['num']}: {truth['invoice_number']}", f_m)
        rtext(dr, W - 40, 150, f"{labels['date']}: {date_text}", f_m)
        y = 260
    else:
        dr.text((40, 40), truth["vendor"].upper(), font=f_b, fill="black")
        dr.text((40, 85), "INVOICE", font=f_m, fill="black")
        dr.text((40, 125), f"{labels['num']} {truth['invoice_number']}   {labels['date']} {date_text}", font=f_s, fill="black")  # short enough
        y = 200
    dr.line([40, y, W - 40, y], fill="black", width=2)
    y += 12
    dr.text((40, y), "Description", font=f_b, fill="black")
    dr.text((440, y), "Qty", font=f_b, fill="black")
    rtext(dr, 680, y, "Price", f_b)
    rtext(dr, W - 40, y, "Amount", f_b)
    y += 44
    for name, qty, price in truth["_items"]:
        dr.text((40, y), name, font=f_m, fill="black")
        dr.text((440, y), str(qty), font=f_m, fill="black")
        rtext(dr, 680, y, sym(price), f_m)
        rtext(dr, W - 40, y, sym(round(qty * price, 2)), f_m)
        y += 42
    y += 20
    dr.line([440, y, W - 40, y], fill="black", width=1)
    y += 18
    for label, value, bold in (("Subtotal", truth["subtotal"], False),
                               (f"{labels['tax']} ({int(round(truth['_rate'] * 100))}%)", truth["tax"], False),
                               (labels["total"], truth["total"], True)):
        dr.text((440, y), label, font=f_b if bold else f_m, fill="black")
        rtext(dr, W - 40, y, sym(value), f_b if bold else f_m)
        y += 48
    dr.text((40, H - 70), "Thank you for your business. Payment due within 30 days.", font=f_s, fill=(90, 90, 90))
    return img


def degrade(img: Image.Image, level: str, rng: random.Random) -> Image.Image:
    if level == "clean":
        return img
    harsh = level == "harsh"
    img = img.rotate(rng.uniform(-3, 3) if harsh else rng.uniform(-1, 1), expand=True, fillcolor="white", resample=Image.BICUBIC)
    img = img.filter(ImageFilter.GaussianBlur(rng.uniform(0.8, 1.4) if harsh else rng.uniform(0.3, 0.7)))
    arr = np.asarray(img).astype(np.float32)
    nrng = np.random.default_rng(rng.randint(0, 2**31))
    arr += nrng.normal(0, 28 if harsh else 10, arr.shape)
    if harsh:  
        arr = 255 - (255 - arr) * rng.uniform(0.55, 0.8)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def generate(n: int, seed: int = 1, out: Path = OUT) -> list[dict]:
    rng = random.Random(seed)
    out.mkdir(parents=True, exist_ok=True)
    records = []
    for i in range(n):
        truth = make_truth(rng, i)
        level = LEVELS[i % len(LEVELS)]
        img = degrade(render(truth, rng), level, rng)
        name = f"{truth['id']}.jpg" if level == "harsh" else f"{truth['id']}.png"
        img.save(out / name, quality=45) if level == "harsh" else img.save(out / name)
        records.append({**{k: v for k, v in truth.items() if not k.startswith("_")}, "image": name, "level": level})
    (out / "truth.jsonl").write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
    return records


def load_truth(folder: Path = OUT) -> list[dict]:
    path = folder / "truth.jsonl"
    if not path.exists():
        raise SystemExit(f"No invoices in {folder}. Run: python -m src.generate --n 150")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    p = argparse.ArgumentParser(description="Generate synthetic invoices with ground truth")
    p.add_argument("--n", type=int, default=150)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--out", type=Path, default=OUT)
    a = p.parse_args()
    recs = generate(a.n, a.seed, a.out)
    print(f"wrote {len(recs)} invoices to {a.out} (levels cycle: {', '.join(LEVELS)})")


if __name__ == "__main__":
    main()
