"""
Video classifier.

Strategy: extract a small handful of frames evenly distributed through the
video, classify each frame with the image model, then vote.

We use OpenCV (cv2) for frame extraction.  cv2 wheels ship with their own
ffmpeg-derived backend, so there is no external ffmpeg dependency at
runtime.  Falls back to bundled ffmpeg / ffprobe if cv2 fails to open the
file (rare exotic codec).
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

from ..config import Config
from ..runtime import third_party_dir
from . import image as image_classifier


_FRAME_COUNT = 5          # how many frames to sample per video
_EDGE_TRIM = 0.05         # skip first/last 5% (intros, fades)


@dataclass
class VideoResult:
    path: Path
    folder: str
    confidence: float
    tags: List[str] = field(default_factory=list)
    error: Optional[str] = None


def _bundled_bin(name: str) -> Optional[Path]:
    """Return path to the bundled binary, or None if missing."""
    p = third_party_dir() / "ffmpeg" / "bin" / f"{name}.exe"
    return p if p.exists() else None


def _probe_duration_ffprobe(path: Path) -> float:
    """Optional ffprobe-based duration probe (used as fallback)."""
    ffprobe = _bundled_bin("ffprobe")
    if ffprobe is None:
        return 0.0
    cmd = [
        str(ffprobe),
        "-v", "error",
        "-print_format", "json",
        "-show_format",
        str(path),
    ]
    try:
        out = subprocess.run(
            cmd, capture_output=True, check=True, timeout=30
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
        return 0.0
    try:
        data = json.loads(out.stdout.decode("utf-8", errors="ignore"))
        return float(data.get("format", {}).get("duration", 0.0) or 0.0)
    except (ValueError, KeyError):
        return 0.0


def _frames_via_cv2(
    path: Path, count: int
) -> Optional[List[Image.Image]]:
    """Sample `count` frames evenly from a video using OpenCV.

    Returns a list of PIL RGB images, or None if the file couldn't be opened.
    """
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        if total <= 0 or fps <= 0:
            return None

        lo = int(total * _EDGE_TRIM)
        hi = max(lo + 1, int(total * (1.0 - _EDGE_TRIM)))
        if count == 1:
            indices = [(lo + hi) // 2]
        else:
            step = max(1, (hi - lo) // (count - 1))
            indices = [min(hi, lo + step * i) for i in range(count)]

        frames: List[Image.Image] = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, float(idx))
            ok, bgr = cap.read()
            if not ok or bgr is None:
                continue
            # OpenCV decodes as BGR; convert to RGB for PIL.
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            # Downscale early; image preprocessor will crop to 224 anyway.
            h, w = rgb.shape[:2]
            if w > 640:
                new_h = int(round(h * (640.0 / w)))
                rgb = cv2.resize(rgb, (640, new_h), interpolation=cv2.INTER_AREA)
            frames.append(Image.fromarray(rgb))
        return frames or None
    finally:
        cap.release()


def _extract_frame_ffmpeg(path: Path, ts_seconds: float) -> Optional[Image.Image]:
    """Fallback: bundled ffmpeg pipe one frame as PNG bytes."""
    ffmpeg = _bundled_bin("ffmpeg")
    if ffmpeg is None:
        return None
    cmd = [
        str(ffmpeg),
        "-loglevel", "error",
        "-ss", f"{ts_seconds:.3f}",
        "-i", str(path),
        "-frames:v", "1",
        "-vf", "scale=640:-1",
        "-f", "image2pipe",
        "-vcodec", "png",
        "-",
    ]
    try:
        out = subprocess.run(cmd, capture_output=True, check=True, timeout=30)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
        return None
    if not out.stdout:
        return None
    try:
        return Image.open(BytesIO(out.stdout)).convert("RGB")
    except Exception:
        return None


def _sample_timestamps(duration: float, n: int) -> List[float]:
    if duration <= 0:
        return [0.0]
    if duration < 1.0:
        return [duration / 2.0]
    lo = duration * _EDGE_TRIM
    hi = duration * (1.0 - _EDGE_TRIM)
    if n == 1:
        return [(lo + hi) / 2.0]
    step = (hi - lo) / (n - 1)
    return [lo + step * i for i in range(n)]


def _sample_frames(path: Path) -> List[Image.Image]:
    # Prefer cv2 (no external deps).  Fall back to bundled ffmpeg if cv2
    # can't open the container.
    frames = _frames_via_cv2(path, _FRAME_COUNT)
    if frames:
        return frames
    duration = _probe_duration_ffprobe(path)
    timestamps = _sample_timestamps(duration, _FRAME_COUNT)
    out: List[Image.Image] = []
    for ts in timestamps:
        f = _extract_frame_ffmpeg(path, ts)
        if f is not None:
            out.append(f)
    return out


def classify(path: Path, cfg: Config) -> VideoResult:
    """Pick the strongest single-frame classification across N samples.

    Averaging confidences across frames was wrong: a video where one
    frame clearly shows a face (conf=1.0) and four frames are dim/random
    (conf=0.05) used to average to 0.21 — below threshold — even though a
    person was clearly present.  Max-pool wins instead, with a small bias
    toward folders that fired in multiple frames (confidence sum).
    """
    frames = _sample_frames(path)

    if not frames:
        return VideoResult(
            path=path, folder="unknown", confidence=0.0,
            tags=["no-frames"], error="no decodable frames",
        )

    # Per-folder: best single-frame score AND total across frames.
    best_per_folder: Dict[str, float] = {}
    sum_per_folder: Dict[str, float] = {}
    frame_tags: List[str] = []
    frames_classified = 0

    for i, frame in enumerate(frames):
        try:
            folder, conf, _scores, tags = image_classifier.classify_pil(frame, cfg)
        except Exception:
            continue
        frames_classified += 1
        if conf > best_per_folder.get(folder, 0.0):
            best_per_folder[folder] = conf
        sum_per_folder[folder] = sum_per_folder.get(folder, 0.0) + conf
        # Keep one short tag per frame for the move log.
        frame_tags.append(f"f{i}={folder}({conf:.2f})")

    if frames_classified == 0:
        return VideoResult(
            path=path, folder="unknown", confidence=0.0,
            tags=["no-frames"], error="no decodable frames",
        )

    # Score each folder: strongest single frame breaks ties; otherwise the
    # folder that hit in more frames wins.
    def folder_score(name: str) -> Tuple[float, float]:
        return (best_per_folder.get(name, 0.0), sum_per_folder.get(name, 0.0))

    best_folder = max(best_per_folder.keys(), key=folder_score)
    best_score = best_per_folder[best_folder]

    tags = [f"frames={frames_classified}"] + frame_tags

    if best_score < cfg.unknown_threshold:
        return VideoResult(
            path=path, folder="unknown", confidence=float(best_score),
            tags=tags + ["below-threshold"],
        )

    if best_folder not in cfg.video_folders:
        return VideoResult(
            path=path, folder="unknown", confidence=float(best_score),
            tags=tags + [f"not-a-video-folder:{best_folder}"],
        )

    return VideoResult(
        path=path, folder=best_folder, confidence=float(best_score), tags=tags,
    )
