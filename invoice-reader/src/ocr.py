from __future__ import annotations

import os
from pathlib import Path

from PIL import Image, ImageOps


def _tesseract():
    import pytesseract

    cmd = os.environ.get("TESSERACT_CMD")
    if not cmd:
        for guess in (r"C:\Program Files\Tesseract-OCR\tesseract.exe", r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"):
            if Path(guess).exists():
                cmd = guess
    if cmd:
        pytesseract.pytesseract.tesseract_cmd = cmd
    return pytesseract


def preprocess(img: Image.Image) -> Image.Image:
    g = ImageOps.autocontrast(ImageOps.grayscale(img), cutoff=1)
    if g.width < 1800:
        scale = 1800 / g.width
        g = g.resize((int(g.width * scale), int(g.height * scale)), Image.LANCZOS)
    return g


def read_text(path: str | Path, psm: int = 6) -> str:
    try:
        pt = _tesseract()
        return pt.image_to_string(preprocess(Image.open(path)), config=f"--psm {psm}")
    except Exception as exc:
        if exc.__class__.__name__ == "TesseractNotFoundError":
            raise SystemExit("Tesseract is not installed or not on PATH. See README step 2 (set TESSERACT_CMD if needed).") from exc
        raise
