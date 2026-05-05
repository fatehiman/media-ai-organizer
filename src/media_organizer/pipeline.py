"""
End-to-end pipeline: scan -> classify -> plan -> confirm -> apply.

The pipeline is split into small functions so the CLI can drive each step
and pause for user confirmation between planning and execution.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

from tqdm import tqdm

from .config import Config
from .movelog import MoveLogWriter, PlanEntry, read_with_confirms
from .mover import move_with_sidecars
from .scanner import MediaItem, scan, summarize
from .workers import Classification, classify_all


# --- target path resolution --------------------------------------------------

def _target_for(cfg: Config, item: MediaItem, folder: str) -> Path:
    """Map (kind, classifier folder) -> absolute destination path.

    Layout:
        target/images/<folder>/<filename>
        target/video/<folder>/<filename>
        target/audio/talk/<filename>
        target/audio/music/<band>/<filename>
        target/audio/unknown/<filename>
        target/unknown/<filename>          (any unknown-kind / failure)
    """
    name = item.path.name

    if item.kind == "image":
        sub = folder if folder in cfg.image_folders else "unknown"
        return cfg.target / "images" / sub / name

    if item.kind == "video":
        sub = folder if folder in cfg.video_folders else "unknown"
        return cfg.target / "video" / sub / name

    if item.kind == "audio":
        # folder is one of: 'talk', 'music/<band>', 'music/unknown-rythm', 'unknown'
        if folder == "talk":
            return cfg.target / "audio" / "talk" / name
        if folder.startswith("music/"):
            band = folder.split("/", 1)[1]
            return cfg.target / "audio" / "music" / band / name
        return cfg.target / "audio" / "unknown" / name

    return cfg.target / "unknown" / name


def _category_label(cfg: Config, item: MediaItem, folder: str) -> str:
    """Human-readable label written into the move log + summary."""
    if item.kind == "image":
        sub = folder if folder in cfg.image_folders else "unknown"
        return f"images/{sub}"
    if item.kind == "video":
        sub = folder if folder in cfg.video_folders else "unknown"
        return f"video/{sub}"
    if item.kind == "audio":
        if folder == "talk":
            return "audio/talk"
        if folder.startswith("music/"):
            return f"audio/{folder}"
        return "audio/unknown"
    return "unknown"


# --- plan ------------------------------------------------------------------

@dataclass
class Plan:
    entries: List[PlanEntry]
    summary: Dict[str, int]


def build_plan(cfg: Config, classifications: List[Classification]) -> Plan:
    entries: List[PlanEntry] = []
    counts: Counter = Counter()

    for c in classifications:
        item = c.item
        folder = c.folder
        # Errored items always go to the top-level unknown folder.
        if c.error and item.kind != "unknown":
            label = "unknown"
            dst = cfg.target / "unknown" / item.path.name
        else:
            label = _category_label(cfg, item, folder)
            dst = _target_for(cfg, item, folder)

        entries.append(PlanEntry(
            src=item.path,
            dst=dst,
            category=label,
            confidence=c.confidence,
            sidecars=item.sidecars,
            tags=c.tags,
        ))
        counts[label] += 1

    return Plan(entries=entries, summary=dict(counts))


# --- apply (move + log) ----------------------------------------------------

def apply_plan(
    cfg: Config,
    plan: Plan,
    log_path: Path,
    *,
    resume: bool = False,
    show_progress: bool = True,
) -> Tuple[int, int]:
    """Execute moves; return (moved, failed)."""
    writer = MoveLogWriter(log_path)

    if resume:
        writer.open_resume()
        existing = read_with_confirms(log_path)
        already_done = set()
        if existing is not None:
            for entry, ok in zip(existing.plan, existing.ok_flags):
                if ok:
                    already_done.add(str(entry.src))
    else:
        # Plan was already persisted to disk by the CLI just before the
        # confirmation prompt, so we only append from here on.
        writer.open_append()
        already_done = set()

    moved = 0
    failed = 0
    iterator = enumerate(plan.entries, 1)
    if show_progress:
        iterator = tqdm(
            list(iterator),
            total=len(plan.entries),
            desc="Moving",
            unit="file",
        )

    try:
        for line_no, entry in iterator:
            if str(entry.src) in already_done:
                continue
            if not entry.src.exists():
                # Could be a partial-resume case: source already moved earlier.
                if entry.dst.exists():
                    writer.mark_ok(line_no)
                    moved += 1
                else:
                    failed += 1
                continue
            try:
                move_with_sidecars(
                    src=entry.src,
                    dst=entry.dst,
                    sidecars=entry.sidecars,
                    suffix_template=cfg.collision_suffix,
                )
                writer.mark_ok(line_no)
                moved += 1
            except Exception:
                failed += 1
        writer.write_completed()
    finally:
        writer.close()

    return moved, failed


# --- end-to-end (used by CLI for fresh runs) -------------------------------

def run_full(cfg: Config) -> Plan:
    """Scan + classify + build plan.  Caller decides whether to apply."""
    items = scan(cfg)
    classifications = classify_all(items, cfg)
    plan = build_plan(cfg, classifications)
    return plan


def kind_summary(cfg: Config) -> Dict[str, int]:
    return summarize(scan(cfg))
