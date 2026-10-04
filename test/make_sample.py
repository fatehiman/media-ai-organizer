"""
Build a labeled-test sample from a media root.

Picks N random images (stratified by file extension so PNG screenshots,
HEIC photos, JPG messenger images, DNG raws are all present), writes
test/<name>_manifest.csv and numbered contact sheets into
test/sheets/<name>/ so a human (or Claude) can label them quickly.
Images already in another test/*_manifest.csv are skipped, so a second
sample is a clean hold-out set.

Usage:
    python test/make_sample.py F:\rkMob\media 300                # -> sample
    python test/make_sample.py F:\rkMob\media 100 holdout 7      # name, seed
"""
from __future__ import annotations

import csv
import random
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except Exception:
    pass

HERE = Path(__file__).resolve().parent
EXTS = {".jpg", ".jpeg", ".png", ".heic", ".heif", ".webp", ".dng"}
# Share of the sample per extension group (rest of the sample is random).
QUOTA = {".jpg": 0.45, ".heic": 0.25, ".png": 0.22, ".dng": 0.04, ".webp": 0.02, ".jpeg": 0.02}
THUMB = 256
COLS, ROWS = 5, 4


def main() -> None:
    root = Path(sys.argv[1])
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 300
    name = sys.argv[3] if len(sys.argv) > 3 else "sample"
    rng = random.Random(int(sys.argv[4]) if len(sys.argv) > 4 else 42)
    used: set[str] = set()
    for m in HERE.glob("*_manifest.csv"):
        if m.name != f"{name}_manifest.csv":
            with m.open(encoding="utf-8") as f:
                used |= {row["path"] for row in csv.DictReader(f)}
    by_ext: dict[str, list[Path]] = {}
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in EXTS and str(p) not in used:
            ext = ".heic" if p.suffix.lower() == ".heif" else p.suffix.lower()
            by_ext.setdefault(ext, []).append(p)
    picked: list[Path] = []
    for ext, share in QUOTA.items():
        pool = sorted(by_ext.get(ext, []))
        k = min(len(pool), max(1, int(n * share)))
        picked += rng.sample(pool, k)
    rng.shuffle(picked)

    sheets = HERE / "sheets" / name
    sheets.mkdir(parents=True, exist_ok=True)
    with (HERE / f"{name}_manifest.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "path", "label"])
        for i, p in enumerate(picked):
            w.writerow([i, str(p), ""])

    font = ImageFont.load_default(size=28)
    per = COLS * ROWS
    for s in range(0, len(picked), per):
        sheet = Image.new("RGB", (COLS * THUMB, ROWS * THUMB), "gray")
        d = ImageDraw.Draw(sheet)
        for j, p in enumerate(picked[s:s + per]):
            try:
                im = ImageOps.exif_transpose(Image.open(p)).convert("RGB")
                im.thumbnail((THUMB - 4, THUMB - 4))
            except Exception:
                im = Image.new("RGB", (THUMB - 4, THUMB - 4), "red")
            x, y = (j % COLS) * THUMB, (j // COLS) * THUMB
            sheet.paste(im, (x + 2, y + 2))
            d.rectangle([x + 2, y + 2, x + 52, y + 34], fill="yellow")
            d.text((x + 5, y + 2), str(s + j), fill="black", font=font)
        sheet.save(sheets / f"sheet_{s // per:02d}.jpg", quality=85)
    print(f"{len(picked)} images, {(len(picked) + per - 1) // per} sheets")


if __name__ == "__main__":
    main()
