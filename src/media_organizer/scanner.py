"""
Source-tree scanner.

Walks the source folder, classifies each file by extension into
image / video / audio / unknown, and groups sidecar files (same stem,
sidecar extension) with their parent media file.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List

from .config import Config


@dataclass
class MediaItem:
    path: Path                    # absolute path to the primary media file
    kind: str                     # 'image' | 'video' | 'audio' | 'unknown'
    sidecars: List[Path] = field(default_factory=list)


def _walk_files(root: Path) -> Iterable[Path]:
    # rglob('*') yields directories too; filter to files only.
    for p in root.rglob("*"):
        if p.is_file():
            yield p


def scan(cfg: Config) -> List[MediaItem]:
    """Return one MediaItem per primary media (or unknown) file in source.

    Sidecar files are NOT returned as their own items; they are attached to
    the matching primary file when one exists in the same directory with the
    same stem.  Sidecars without a matching primary become 'unknown' items.
    """
    if not cfg.source.exists():
        raise FileNotFoundError(f"Source folder does not exist: {cfg.source}")

    sidecar_exts = set(cfg.sidecar_extensions) if cfg.move_sidecars else set()
    # The move log lives in the target folder; when source == target (i.e.
    # re-classifying an already-organized tree) it would otherwise be picked
    # up as 'unknown' and shuffled around mid-run.  Skip it.
    move_log_path = (cfg.target / cfg.move_log).resolve()

    # First pass: collect every file, separated into primaries and sidecars.
    primaries: Dict[Path, MediaItem] = {}
    sidecar_files: List[Path] = []

    for p in _walk_files(cfg.source):
        try:
            if p.resolve() == move_log_path:
                continue
        except OSError:
            pass
        ext = p.suffix.lower()
        if ext in sidecar_exts:
            sidecar_files.append(p)
            continue
        kind = cfg.media_kind(ext)
        primaries[p] = MediaItem(path=p, kind=kind)

    # Second pass: attach sidecars to primaries by (parent dir, stem).
    if sidecar_exts:
        index: Dict[tuple, MediaItem] = {
            (item.path.parent, item.path.stem): item
            for item in primaries.values()
        }
        for sc in sidecar_files:
            key = (sc.parent, sc.stem)
            host = index.get(key)
            if host is not None:
                host.sidecars.append(sc)
            else:
                # Orphan sidecar: treat as unknown so it gets moved somewhere
                # rather than silently left behind.
                primaries[sc] = MediaItem(path=sc, kind="unknown")

    # Stable order: sorted by path so the move log is deterministic.
    return sorted(primaries.values(), key=lambda it: str(it.path).lower())


def summarize(items: List[MediaItem]) -> Dict[str, int]:
    counts = {"image": 0, "video": 0, "audio": 0, "unknown": 0}
    for it in items:
        counts[it.kind] = counts.get(it.kind, 0) + 1
    return counts
