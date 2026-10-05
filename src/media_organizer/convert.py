"""
Image conversion for the GUI: resize (aspect ratio kept, never enlarged)
and re-encode to JPG or PNG.

Per file the GUI runs:  open source -> convert in memory -> (categorize the
small converted image) -> write file -> (send source to the Recycle Bin).

Metadata is always copied (there is no option to turn it off), as far as
the target format can hold it:

    what                                  JPG target      PNG target
    EXIF: camera, date taken, GPS, ...    yes             yes (eXIf chunk)
    XMP                                   yes (APP1)      yes (iTXt chunk)
    ICC color profile                     yes             yes
    JPEG comment                          yes             yes (tEXt "Comment")
    IPTC (JPEG APP13)                     yes             - (no standard place)
    PNG text chunks                       -               yes
    file created / modified time          yes             yes

The EXIF Orientation tag is written as 1 ("normal") because the pixels
are physically rotated upright during conversion; keeping the old value
would make viewers rotate the image a second time.

After encoding, `_verify` re-reads the output and checks that every EXIF
tag, the XMP and the ICC profile arrived.  If not, MetadataError is
raised: the file is not written, so the source is never deleted.
"""

from __future__ import annotations

import ctypes
import io
import os
import struct
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from PIL import Image, ImageOps, PngImagePlugin

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
# Sub-IFDs whose tags we copy and verify: Exif, GPS, Interop.
_SUB_IFDS = (0x8769, 0x8825, 0xA005)
_XMP_PNG_KEY = "XML:com.adobe.xmp"


class MetadataError(Exception):
    """The converted file would lose metadata; nothing was written."""


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
class SourceMetadata:
    exif: Optional[bytes] = None              # Orientation already set to 1
    exif_tags: Set[Tuple[int, int]] = field(default_factory=set)  # (ifd, tag)
    xmp: Optional[bytes] = None
    icc: Optional[bytes] = None
    comment: Optional[bytes] = None
    iptc: List[bytes] = field(default_factory=list)   # raw JPEG APP13 payloads
    png_text: Dict[str, str] = field(default_factory=dict)


@dataclass
class Converted:
    image: Image.Image            # resized pixels (RGB or RGBA)
    data: bytes                   # the encoded output file
    meta: ImageMeta               # categorizer signals of the source file
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


# --- metadata ----------------------------------------------------------------

def _exif_tag_set(exif: Image.Exif) -> Set[Tuple[int, int]]:
    tags = {(0, t) for t in exif if t not in _SUB_IFDS}
    for ifd in _SUB_IFDS:
        try:
            tags |= {(ifd, t) for t in exif.get_ifd(ifd)}
        except Exception:
            pass
    return tags


def _as_bytes(v) -> Optional[bytes]:
    if not v:
        return None
    return v.encode("utf-8") if isinstance(v, str) else bytes(v)


def _read_metadata(img: Image.Image) -> SourceMetadata:
    md = SourceMetadata()
    try:
        exif = img.getexif()      # cached object, shared with exif_transpose
        if len(exif):
            original = exif.get(_EXIF_ORIENTATION)
            exif[_EXIF_ORIENTATION] = 1
            # Load the sub-IFDs so tobytes() writes them too.
            md.exif_tags = _exif_tag_set(exif)
            md.exif = exif.tobytes()
            # Restore it: exif_transpose still has to rotate the pixels.
            if original is None:
                del exif[_EXIF_ORIENTATION]
            else:
                exif[_EXIF_ORIENTATION] = original
    except Exception:
        pass
    md.xmp = _as_bytes(img.info.get("xmp") or img.info.get(_XMP_PNG_KEY))
    md.icc = _as_bytes(img.info.get("icc_profile"))
    md.comment = _as_bytes(img.info.get("comment"))
    for marker, payload in getattr(img, "applist", []):
        if marker == "APP13":
            md.iptc.append(payload)
    for key, value in (getattr(img, "text", None) or {}).items():
        if key != _XMP_PNG_KEY:
            md.png_text[key] = str(value)
    if md.comment is None and "Comment" in md.png_text:
        md.comment = md.png_text["Comment"].encode("utf-8")   # PNG -> JPG
    return md


def _save(img: Image.Image, fmt: str, quality: int, md: SourceMetadata) -> bytes:
    buf = io.BytesIO()
    params = {}
    if md.exif:
        params["exif"] = md.exif
    if md.icc:
        params["icc_profile"] = md.icc
    if fmt == "jpg":
        if md.xmp:
            params["xmp"] = md.xmp
        if md.comment:
            params["comment"] = md.comment
        if md.iptc:
            params["extra"] = b"".join(
                b"\xff\xed" + struct.pack(">H", len(p) + 2) + p for p in md.iptc
            )
        img.save(buf, "JPEG", quality=quality, **params)
    else:
        info = PngImagePlugin.PngInfo()
        for key, value in md.png_text.items():
            info.add_itxt(key, value)
        if md.comment and "Comment" not in md.png_text:
            info.add_text("Comment", md.comment.decode("utf-8", "replace"))
        if md.xmp:
            info.add_itxt(_XMP_PNG_KEY, md.xmp.decode("utf-8", "replace"))
        img.save(buf, "PNG", pnginfo=info, **params)
    return buf.getvalue()


def _verify(data: bytes, md: SourceMetadata) -> None:
    """Re-read the encoded output; raise MetadataError if anything the
    target format can hold did not arrive."""
    with Image.open(io.BytesIO(data)) as out:
        missing = []
        if md.exif_tags:
            lost = md.exif_tags - _exif_tag_set(out.getexif())
            if lost:
                missing.append(f"{len(lost)} EXIF tag(s) {sorted(lost)[:5]}")
        if md.xmp and not (out.info.get("xmp") or out.info.get(_XMP_PNG_KEY)):
            missing.append("XMP")
        if md.icc and not out.info.get("icc_profile"):
            missing.append("ICC profile")
    if missing:
        raise MetadataError("metadata would be lost: " + ", ".join(missing))


# --- conversion --------------------------------------------------------------

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
        orientation = img.getexif().get(_EXIF_ORIENTATION, 1)
        md = _read_metadata(img)

        w, h = img.size
        if orientation in _SWAPS_AXES:
            w, h = h, w
        tw, th = target_size(w, h, opts)
        # JPEG only: decode directly at a smaller scale (>= target size).
        img.draft("RGB", (th, tw) if orientation in _SWAPS_AXES else (tw, th))

        img = ImageOps.exif_transpose(img)
        if img.size != (tw, th):
            img = img.resize((tw, th), Image.LANCZOS, reducing_gap=3.0)
        img = _to_output_mode(img, opts.fmt)

    data = _save(img, opts.fmt, opts.quality, md)
    _verify(data, md)
    return Converted(image=img, data=data, meta=meta,
                     src_size=(w, h), src_bytes=src_bytes)


# --- output ------------------------------------------------------------------

def unique_path(folder: Path, stem: str, ext: str) -> Path:
    """folder/stem.ext, or 'stem (2).ext', 'stem (3).ext', ... if taken."""
    dst = folder / f"{stem}{ext}"
    n = 2
    while dst.exists():
        dst = folder / f"{stem} ({n}){ext}"
        n += 1
    return dst


def _copy_creation_time(source: Path, dst: Path) -> None:
    """Windows only: give dst the creation time of source."""
    if sys.platform != "win32":
        return
    from ctypes import wintypes
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.CreateFileW.restype = wintypes.HANDLE
    GENERIC_READ, GENERIC_WRITE = 0x80000000, 0x40000000
    OPEN_EXISTING, SHARE_ALL = 3, 0x7
    created = wintypes.FILETIME()
    h = k32.CreateFileW(str(source), GENERIC_READ, SHARE_ALL, None,
                        OPEN_EXISTING, 0, None)
    if h in (None, wintypes.HANDLE(-1).value):
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        if not k32.GetFileTime(h, ctypes.byref(created), None, None):
            raise ctypes.WinError(ctypes.get_last_error())
    finally:
        k32.CloseHandle(h)
    h = k32.CreateFileW(str(dst), GENERIC_WRITE, SHARE_ALL, None,
                        OPEN_EXISTING, 0, None)
    if h in (None, wintypes.HANDLE(-1).value):
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        if not k32.SetFileTime(h, ctypes.byref(created), None, None):
            raise ctypes.WinError(ctypes.get_last_error())
    finally:
        k32.CloseHandle(h)


def write_output(dst: Path, data: bytes, source: Path) -> None:
    """Write via a temp file + rename (no half-written outputs), fsync, and
    copy the source's created / accessed / modified times."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".part")
    with tmp.open("wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, dst)
    st = source.stat()
    os.utime(dst, (st.st_atime, st.st_mtime))
    _copy_creation_time(source, dst)
