"""
OCR helper.

Calls the bundled Tesseract binary (third_party/tesseract/tesseract.exe) via
pytesseract.  We only need a *count* of recognized words -- a fast PSM_6
single-block layout is plenty.  No language data beyond English is required
to detect "this image has lots of text" with reasonable accuracy.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Optional

import pytesseract
from PIL import Image

from ..runtime import third_party_dir


_INIT_DONE = False
_WORD_RE = re.compile(r"[A-Za-zÀ-ɏ]{2,}")


def _find_tesseract() -> Optional[Path]:
    """Locate tesseract.exe, in priority order:
       1. third_party/tesseract/tesseract.exe (bundled, portable)
       2. C:/Program Files/Tesseract-OCR/tesseract.exe (UB-Mannheim default)
       3. Whatever pytesseract's defaults can find on PATH
    """
    bundled = third_party_dir() / "tesseract" / "tesseract.exe"
    if bundled.exists():
        return bundled
    win_install = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")
    if win_install.exists():
        return win_install
    return None


def _ensure_init() -> None:
    """Point pytesseract at the chosen binary (once per process)."""
    global _INIT_DONE
    if _INIT_DONE:
        return
    tess = _find_tesseract()
    if tess is not None:
        pytesseract.pytesseract.tesseract_cmd = str(tess)
        tessdata = tess.parent / "tessdata"
        if tessdata.exists():
            os.environ.setdefault("TESSDATA_PREFIX", str(tessdata))
    _INIT_DONE = True


def is_available() -> bool:
    """Return True if Tesseract is reachable in this process."""
    _ensure_init()
    return _find_tesseract() is not None


_MIN_CONF = 60.0   # Tesseract per-word confidence (0..100); below this we
                    # treat the recognition as hallucination (typical when
                    # PSM 6 forces text recognition on textured surfaces
                    # like concrete, brick, fabric, foliage).


def count_words(img: Image.Image) -> int:
    """Return the number of *high-confidence* word-like tokens in the image.

    Uses pytesseract.image_to_data (gives per-word confidence) instead of
    image_to_string.  Only words matching `_WORD_RE` AND with confidence
    >= _MIN_CONF count.  Returns 0 on any error.
    """
    _ensure_init()
    try:
        data = pytesseract.image_to_data(
            img,
            config="--psm 6",
            output_type=pytesseract.Output.DICT,
        )
    except Exception:
        return 0

    texts = data.get("text", []) or []
    confs = data.get("conf", []) or []
    n = 0
    for t, c in zip(texts, confs):
        try:
            conf = float(c)
        except (TypeError, ValueError):
            continue
        if conf < _MIN_CONF:
            continue
        if not t:
            continue
        if _WORD_RE.fullmatch(t.strip()):
            n += 1
    return n
