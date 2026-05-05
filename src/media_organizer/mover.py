"""
Safe file movement: collision-resolution + sidecars.

Filename collisions are resolved by appending a numeric suffix:
    IMG_0001.jpg -> IMG_0001 (2).jpg -> IMG_0001 (3).jpg ...
The numbering format is configurable via collision-suffix (default '({n})').
Sidecars are renamed in lockstep to preserve their stem<->primary link.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import List, Tuple


def _format_suffix(template: str, n: int) -> str:
    # Expects '{n}' placeholder; falls back to ' (n)' if absent.
    # Always rendered with a leading space so we get
    # "IMG_0001 (2).jpg" (Windows convention) rather than
    # "IMG_0001(2).jpg".
    if "{n}" in template:
        return " " + template.replace("{n}", str(n))
    return f" ({n})"


def resolve_collision(dst: Path, suffix_template: str) -> Path:
    """Return a non-existing path derived from dst by adding a numeric suffix."""
    if not dst.exists():
        return dst
    stem, ext = dst.stem, dst.suffix
    parent = dst.parent
    n = 2
    while True:
        candidate = parent / f"{stem}{_format_suffix(suffix_template, n)}{ext}"
        if not candidate.exists():
            return candidate
        n += 1
        if n > 9999:
            raise RuntimeError(f"Too many collisions for {dst}")


def _ensure_dir(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)


def _move(src: Path, dst: Path) -> None:
    _ensure_dir(dst)
    # No-op when source and destination resolve to the same file.  Happens
    # when the user re-classifies an already-organized tree (source ==
    # target) and a file's classification didn't change.
    try:
        same = src.resolve() == dst.resolve()
    except OSError:
        same = False
    if same:
        return
    # shutil.move handles cross-volume moves transparently (copy + remove).
    # On Windows, if dst exists this would fail; collision resolution should
    # have produced a fresh path, but we double-check defensively.
    if dst.exists():
        raise FileExistsError(f"Destination already exists: {dst}")
    shutil.move(str(src), str(dst))


def move_with_sidecars(
    src: Path,
    dst: Path,
    sidecars: List[Path],
    suffix_template: str,
) -> Tuple[Path, List[Tuple[Path, Path]]]:
    """Move src to dst (resolving any collision), then move each sidecar to
    a parallel destination (same stem in the same target directory, original
    sidecar extension).  Returns (final_dst, [(sidecar_src, sidecar_dst), ...]).

    If a sidecar's target also collides, it gets its own numeric suffix --
    independent of the primary's suffix -- to preserve uniqueness without
    forcing the primary to renumber.
    """
    final_dst = resolve_collision(dst, suffix_template)
    _move(src, final_dst)

    sidecar_pairs: List[Tuple[Path, Path]] = []
    for sc in sidecars:
        if not sc.exists():
            # Sidecar may have already been moved / removed by the user; skip.
            continue
        sc_dst = final_dst.parent / (final_dst.stem + sc.suffix.lower())
        sc_dst = resolve_collision(sc_dst, suffix_template)
        try:
            _move(sc, sc_dst)
            sidecar_pairs.append((sc, sc_dst))
        except OSError:
            # Don't abort the whole run because one sidecar misbehaved.
            sidecar_pairs.append((sc, sc))

    return final_dst, sidecar_pairs
