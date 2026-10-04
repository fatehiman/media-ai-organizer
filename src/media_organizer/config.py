"""
Configuration loader for media-organizer.conf.

The file is plain `key = value` (one per line, '#' or ';' starts a comment).
Lists are CSV.  Prefix-keyed groups (e.g. `image-personal`, `prompts-person`,
`rhythm-slow`) are exposed as dicts so the CLI can iterate them dynamically.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple


def _strip_comment(line: str) -> str:
    # Honor ';' and '#' as comment starters, but only outside any value content.
    # We split on the first occurrence and discard the rest.
    out = []
    for ch in line:
        if ch in ("#", ";"):
            break
        out.append(ch)
    return "".join(out)


def _parse_csv(value: str) -> List[str]:
    return [v.strip() for v in value.split(",") if v.strip()]


def _parse_bool(value: str) -> bool:
    return value.strip().lower() in ("1", "true", "yes", "on", "y")


def _parse_bpm_range(value: str) -> Tuple[int, int]:
    lo, hi = value.split("-", 1)
    return int(lo.strip()), int(hi.strip())


@dataclass
class Config:
    # Paths
    source: Path
    target: Path

    # Behavior
    dry_run: bool = True
    require_confirmation: bool = True
    move_log: str = "media-organizer-move.log"
    unknown_threshold: float = 0.40
    fallback_folder: str = "other"      # images/videos below unknown-threshold
    collision_suffix: str = "({n})"
    move_sidecars: bool = True
    sidecar_extensions: List[str] = field(default_factory=list)

    # Performance
    use_gpu: str = "auto"               # auto | yes | no
    cpu_workers: int = 0                # 0 == auto
    gpu_batch_size: int = 16

    # Extension sets (lowercase, with leading dot)
    ext_image: List[str] = field(default_factory=list)
    ext_video: List[str] = field(default_factory=list)
    ext_audio: List[str] = field(default_factory=list)

    # Folder definitions: {folder_name: [content_type, ...]}
    image_folders: Dict[str, List[str]] = field(default_factory=dict)
    video_folders: Dict[str, List[str]] = field(default_factory=dict)

    # Audio thresholds
    audio_talk_min_speech_ratio: float = 0.60
    audio_music_max_speech_ratio: float = 0.20

    # Face detection
    face_detection_enabled: bool = True
    face_min_score: float = 0.88
    face_boost: float = 0.50
    face_target_content_type: str = "person"

    # Screenshot detection from file metadata
    screenshot_content_type: str = "screenshot"
    screenshot_metadata: bool = True
    screenshot_tall_boost: float = 0.50

    # Optional Tesseract OCR boost (0 = OCR not run at all)
    ocr_boost: float = 0.0
    ocr_content_type: str = "document"

    # Rhythm bands: {band_name: (lo_bpm, hi_bpm)}
    rhythm_bands: Dict[str, Tuple[int, int]] = field(default_factory=dict)

    # CLIP prompts: {content_type: [sentence, ...]}
    prompts: Dict[str, List[str]] = field(default_factory=dict)

    # Raw key/value bag for any keys we did not recognize (forward-compat).
    raw: Dict[str, str] = field(default_factory=dict)

    @property
    def all_extensions(self) -> List[str]:
        return self.ext_image + self.ext_video + self.ext_audio

    def media_kind(self, ext: str) -> str:
        ext = ext.lower()
        if ext in self.ext_image:
            return "image"
        if ext in self.ext_video:
            return "video"
        if ext in self.ext_audio:
            return "audio"
        return "unknown"


def load(config_path: Path | str) -> Config:
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    raw: Dict[str, str] = {}
    with config_path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            stripped = _strip_comment(line).strip()
            if not stripped:
                continue
            if "=" not in stripped:
                raise ValueError(
                    f"{config_path}:{lineno}: missing '=' in line: {line!r}"
                )
            key, value = stripped.split("=", 1)
            raw[key.strip().lower()] = value.strip()

    def get(key: str, default: str | None = None) -> str | None:
        return raw.get(key, default)

    # -- Required paths ----------------------------------------------------
    source = get("source")
    target = get("target")
    if not source or not target:
        raise ValueError("Config must define both 'source' and 'target' paths")

    cfg = Config(
        source=Path(source).expanduser(),
        target=Path(target).expanduser(),
    )

    # -- Behavior ----------------------------------------------------------
    if (v := get("dry-run")) is not None:
        cfg.dry_run = _parse_bool(v)
    if (v := get("require-confirmation")) is not None:
        cfg.require_confirmation = _parse_bool(v)
    if (v := get("move-log")) is not None:
        cfg.move_log = v
    if (v := get("unknown-threshold")) is not None:
        cfg.unknown_threshold = float(v)
    if (v := get("fallback-folder")) is not None:
        cfg.fallback_folder = v.strip()
    if (v := get("collision-suffix")) is not None:
        cfg.collision_suffix = v
    if (v := get("move-sidecars")) is not None:
        cfg.move_sidecars = _parse_bool(v)
    if (v := get("sidecar-extensions")) is not None:
        cfg.sidecar_extensions = [e.lower() for e in _parse_csv(v)]

    # -- Performance -------------------------------------------------------
    if (v := get("use-gpu")) is not None:
        cfg.use_gpu = v.strip().lower()
    if (v := get("cpu-workers")) is not None:
        cfg.cpu_workers = 0 if v.strip().lower() == "auto" else int(v)
    if (v := get("gpu-batch-size")) is not None:
        cfg.gpu_batch_size = int(v)

    # -- Extensions --------------------------------------------------------
    cfg.ext_image = [e.lower() for e in _parse_csv(get("ext-image", ""))]
    cfg.ext_video = [e.lower() for e in _parse_csv(get("ext-video", ""))]
    cfg.ext_audio = [e.lower() for e in _parse_csv(get("ext-audio", ""))]

    # -- Audio thresholds --------------------------------------------------
    if (v := get("audio-talk-min-speech-ratio")) is not None:
        cfg.audio_talk_min_speech_ratio = float(v)
    if (v := get("audio-music-max-speech-ratio")) is not None:
        cfg.audio_music_max_speech_ratio = float(v)

    # -- Face detection ---------------------------------------------------
    if (v := get("face-detection")) is not None:
        cfg.face_detection_enabled = _parse_bool(v)
    if (v := get("face-min-score")) is not None:
        cfg.face_min_score = float(v)
    if (v := get("face-boost")) is not None:
        cfg.face_boost = float(v)
    if (v := get("face-target-content-type")) is not None:
        cfg.face_target_content_type = v.strip().lower()

    # -- Screenshot / OCR signals ------------------------------------------
    if (v := get("screenshot-content-type")) is not None:
        cfg.screenshot_content_type = v.strip().lower()
    if (v := get("screenshot-metadata")) is not None:
        cfg.screenshot_metadata = _parse_bool(v)
    if (v := get("screenshot-tall-boost")) is not None:
        cfg.screenshot_tall_boost = float(v)
    if (v := get("ocr-boost")) is not None:
        cfg.ocr_boost = float(v)
    if (v := get("ocr-content-type")) is not None:
        cfg.ocr_content_type = v.strip().lower()

    # -- Prefix-keyed groups ----------------------------------------------
    for key, value in raw.items():
        if key.startswith("image-"):
            folder = key[len("image-"):]
            if folder in ("personal", "docs", "nonsense") or True:
                # any image-<folder> is taken; reserved buckets are still fine
                cfg.image_folders[folder] = _parse_csv(value)
        elif key.startswith("video-"):
            folder = key[len("video-"):]
            cfg.video_folders[folder] = _parse_csv(value)
        elif key.startswith("rhythm-"):
            band = key[len("rhythm-"):]
            cfg.rhythm_bands[band] = _parse_bpm_range(value)
        elif key.startswith("prompts-"):
            ctype = key[len("prompts-"):]
            cfg.prompts[ctype] = _parse_csv(value)

    cfg.raw = raw
    return cfg
