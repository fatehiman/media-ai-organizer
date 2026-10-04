"""
Process pool that classifies files in parallel.

We use multiprocessing because:
  * The image / audio / video classifiers are mostly CPU-bound numpy + ONNX
    work, plus subprocess calls (ffmpeg, tesseract).  GIL contention would
    cap thread-based parallelism.
  * Each worker creates its own ONNX session lazily on first use.  The CLIP
    vision model costs ~350 MB of RAM per worker, so with `cpu-workers =
    auto` the pool size is capped by free RAM (see `_auto_workers`) and
    each worker gets several ONNX threads to keep all cores busy.
  * The CLIP text embeddings are computed once here, in the main process,
    and handed to every worker.

The dispatcher routes by media kind so each call lands in the right
classifier.  Failures are caught and surfaced as classification results
with `error` populated (the file then routes to `unknown/`).
"""

from __future__ import annotations

import ctypes
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

from . import runtime
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


def _init_worker(
    cfg: Config,
    text_embeddings: Optional[Tuple[List[str], np.ndarray]],
    onnx_threads: int,
) -> None:
    global _WORKER_CFG
    _WORKER_CFG = cfg
    runtime.set_intra_op_threads(onnx_threads)
    if text_embeddings is not None:
        from .classifiers import clip
        clip.set_text_embeddings(text_embeddings)


# RAM one worker needs: CLIP vision session (~350 MB) + decoded image and
# YuNet / OCR buffers.  Measured peak is ~550 MB; keep some margin.
_WORKER_RAM_BYTES = 700 * 1024 * 1024


def _available_ram() -> Optional[int]:
    """Free physical RAM in bytes, or None if it can't be read."""
    try:
        if sys.platform == "win32":
            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]
            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
                return int(stat.ullAvailPhys)
            return None
        return os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    except (AttributeError, OSError, ValueError):
        return None


def _auto_workers() -> int:
    cpus = os.cpu_count() or 4
    ram = _available_ram()
    if ram is None:
        return cpus
    return max(1, min(cpus, ram // _WORKER_RAM_BYTES))


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

    workers = cfg.cpu_workers if cfg.cpu_workers > 0 else _auto_workers()
    # Spread the cores over the workers (1 thread each when not RAM-capped).
    onnx_threads = max(1, (os.cpu_count() or 4) // workers)

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
        text_embeddings = None
        if any(it.kind in ("image", "video") for it in todo):
            from .classifiers import clip
            text_embeddings = clip.compute_text_embeddings(cfg)
        if show_progress:
            print(f"Workers       : {workers} x {onnx_threads} ONNX thread(s)")
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_init_worker,
            initargs=(cfg, text_embeddings, onnx_threads),
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
