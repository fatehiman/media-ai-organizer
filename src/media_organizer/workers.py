"""
Process pool that classifies files in parallel.

We use multiprocessing because:
  * The image / audio / video classifiers are mostly CPU-bound numpy + ONNX
    work, plus subprocess calls (ffmpeg, tesseract).  GIL contention would
    cap thread-based parallelism.
  * Each worker creates its own ONNX session lazily on first use.  Multiple
    sessions sharing one GPU works fine for small models (MobileNetV3 ~22MB,
    silero-vad ~2MB).

The dispatcher routes by media kind so each call lands in the right
classifier.  Failures are caught and surfaced as classification results
with `error` populated (the file then routes to `unknown/`).
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional

from tqdm import tqdm

from .config import Config
from .scanner import MediaItem


@dataclass
class Classification:
    item: MediaItem
    folder: str         # final sub-folder under <kind>/  (e.g. 'personal',
                        # 'music/medium', 'unknown')
    confidence: float
    tags: List[str] = field(default_factory=list)   # explanatory reasons
    error: Optional[str] = None


# Module-global config used inside workers; populated by `_init_worker`.
# Keeping it here (vs pickling per-task) avoids re-shipping the full config
# tree thousands of times.
_WORKER_CFG: Optional[Config] = None


def _init_worker(cfg: Config) -> None:
    global _WORKER_CFG
    _WORKER_CFG = cfg


def _classify_one(item: MediaItem) -> Classification:
    cfg = _WORKER_CFG
    assert cfg is not None, "worker not initialized"
    try:
        if item.kind == "image":
            from .classifiers import image as image_clf
            r = image_clf.classify(item.path, cfg)
            return Classification(item, r.folder, r.confidence,
                                  tags=r.tags, error=r.error)
        if item.kind == "video":
            from .classifiers import video as video_clf
            r = video_clf.classify(item.path, cfg)
            return Classification(item, r.folder, r.confidence,
                                  tags=r.tags, error=r.error)
        if item.kind == "audio":
            from .classifiers import audio as audio_clf
            r = audio_clf.classify(item.path, cfg)
            return Classification(item, r.folder, r.confidence,
                                  tags=getattr(r, "tags", []), error=r.error)
        return Classification(item, "unknown", 0.0)
    except Exception as e:
        return Classification(item, "unknown", 0.0, error=str(e))


def classify_all(
    items: List[MediaItem],
    cfg: Config,
    *,
    show_progress: bool = True,
) -> List[Classification]:
    """Run all classification tasks in parallel; return one Classification
    per input MediaItem (preserving input order)."""

    workers = cfg.cpu_workers if cfg.cpu_workers > 0 else (os.cpu_count() or 4)

    # Items with kind=='unknown' don't need a worker at all -- classify them
    # synchronously to avoid IPC overhead for thousands of trivial calls.
    quick: List[Classification] = []
    todo: List[MediaItem] = []
    for it in items:
        if it.kind == "unknown":
            quick.append(Classification(it, "unknown", 0.0))
        else:
            todo.append(it)

    results_by_path = {c.item.path: c for c in quick}

    if todo:
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_init_worker,
            initargs=(cfg,),
        ) as pool:
            futures = {pool.submit(_classify_one, it): it for it in todo}
            iterator: Iterable = as_completed(futures)
            if show_progress:
                iterator = tqdm(
                    iterator, total=len(futures), desc="Classifying", unit="file"
                )
            for fut in iterator:
                c = fut.result()
                results_by_path[c.item.path] = c

    # Reassemble in original order.
    return [results_by_path[it.path] for it in items]
