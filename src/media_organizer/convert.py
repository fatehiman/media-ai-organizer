"""
Image conversion for the GUI: resize (aspect ratio kept, never enlarged)
and re-encode to JPG or PNG.

Per file the GUI runs:  open source -> convert in memory -> (categorize the
small converted image) -> write file -> (send source to the Recycle Bin).

EXIF (with Orientation reset to 1, because the pixels are already rotated),
the ICC color profile and the file's modified time are copied to the
output, so photo apps still see the camera, date and colors.
"""

from __future__ import annotations

import io
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

from PIL import Image, ImageOps

# HEIC/HEIF support — register the opener with PIL on import.
try:
    import pillow_heif  # type: ignore
    pillow_heif.register_heif_opener()
except Exception:
    pass

from .classifiers.image import ImageMeta, read_meta


_EXIF_ORIENTATION = 274
# EXIF orientations 5..8 rotate by 90 degrees: width and height swap.
_SWAPS_AXES = {5, 6, 7, 8}


@dataclass
class ConvertOptions:
    resize_mode: str = "none"     # none | percent | width | height
    resize_value: int = 100       # percent, or pixels for width / height
    fmt: str = "jpg"              # jpg | png
    quality: int = 85             # JPG only, 1..100

    @property
    def ext(self) -> str:
        return ".jpg" if self.fmt == "jpg" else ".png"


@dataclass
class Converted:
    image: Image.Image            # resized pixels (RGB or RGBA)
    data: bytes                   # the encoded output file
    meta: ImageMeta               # metadata of the source file
    src_size: Tuple[int, int]     # oriented source width, height
    src_bytes: int


def target_size(w: int, h: int, opts: ConvertOptions) -> Tuple[int, int]:
    """New (w, h) for an oriented w x h image.  Aspect ratio is kept and
    the image is never enlarged."""
    if opts.resize_mode == "percent":
        scale = opts.resize_value / 100.0
    elif opts.resize_mode == "width":
        scale = opts.resize_value / w
    elif opts.resize_mode == "height":
        scale = opts.resize_value / h
    else:
        scale = 1.0
    scale = min(scale, 1.0)
    return max(1, round(w * scale)), max(1, round(h * scale))


def _exif_bytes(img: Image.Image) -> Optional[bytes]:
    try:
        exif = img.getexif()
    except Exception:
        return None
    if not len(exif):
        return None
    exif[_EXIF_ORIENTATION] = 1
    try:
        return exif.tobytes()
    except Exception:
        return None


def _to_output_mode(img: Image.Image, fmt: str) -> Image.Image:
    has_alpha = img.mode in ("RGBA", "LA") or (
        img.mode == "P" and "transparency" in img.info
    )
    if fmt == "png" and has_alpha:
        return img.convert("RGBA")
    if has_alpha:
        # JPG has no alpha: put transparent pixels on white.
        rgba = img.convert("RGBA")
        flat = Image.new("RGB", rgba.size, (255, 255, 255))
        flat.paste(rgba, mask=rgba.getchannel("A"))
        return flat
    return img.convert("RGB")


def convert(path: Path, opts: ConvertOptions) -> Converted:
    src_bytes = path.stat().st_size
    with Image.open(path) as img:
        meta = read_meta(path, img)
        exif = _exif_bytes(img)
        icc = img.info.get("icc_profile")

        w, h = img.size
        orientation = img.getexif().get(_EXIF_ORIENTATION, 1)
        if orientation in _SWAPS_AXES:
            w, h = h, w
        tw, th = target_size(w, h, opts)
        # JPEG only: decode directly at a smaller scale (>= target size).
        img.draft("RGB", (th, tw) if orientation in _SWAPS_AXES else (tw, th))

        img = ImageOps.exif_transpose(img)
        if img.size != (tw, th):
            img = img.resize((tw, th), Image.LANCZOS, reducing_gap=3.0)
        img = _to_output_mode(img, opts.fmt)

    buf = io.BytesIO()
    params = {}
    if exif:
        params["exif"] = exif
    if icc:
        params["icc_profile"] = icc
    if opts.fmt == "jpg":
        img.save(buf, "JPEG", quality=opts.quality, **params)
    else:
        img.save(buf, "PNG", **params)
    return Converted(image=img, data=buf.getvalue(), meta=meta,
                     src_size=(w, h), src_bytes=src_bytes)


def unique_path(folder: Path, stem: str, ext: str) -> Path:
    """folder/stem.ext, or 'stem (2).ext', 'stem (3).ext', ... if taken."""
    dst = folder / f"{stem}{ext}"
    n = 2
    while dst.exists():
        dst = folder / f"{stem} ({n}){ext}"
        n += 1
    return dst


def write_output(dst: Path, data: bytes, source: Path) -> None:
    """Write via a temp file + rename (no half-written outputs), fsync, and
    copy the source's access / modified time."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".part")
    with tmp.open("wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, dst)
    st = source.stat()
    os.utime(dst, (st.st_atime, st.st_mtime))
