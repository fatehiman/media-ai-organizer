"""
Image classifier.

Pipeline per file:
  1. Decode (HEIC/HEIF supported via pillow_heif) and read metadata.
  2. Screenshot metadata: iOS writes EXIF UserComment "Screenshot", Android
     names files "Screenshot_...".  A hit routes straight to the screenshot
     content type; no model runs.
  3. CLIP ViT-B/32 zero-shot -> probability per content type, from the
     `prompts-<type>` lines in the config (see clip.py).
  4. Metadata adjustments:
       - camera EXIF (Make) present -> it is a camera photo, so the
         screenshot score is zeroed;
       - no camera EXIF and phone-shaped (>= 1.9:1) -> screenshot boosted.
  5. Face detection (YuNet).  A face scoring >= face-min-score adds
     face-boost to the face-target content type.  Cartoon faces on toys and
     cookies score lower than real faces, so the threshold filters them.
  6. Optional OCR boost for the document type (ocr-boost > 0).
  7. Aggregate per folder (sum of its content-type scores).  Winner = max;
     if max < unknown-threshold -> fallback folder.

Returns an ImageResult with chosen folder + confidence + a `tags` list of
human-readable reasons that explain the decision.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageOps

# HEIC/HEIF support — register the opener with PIL on import.
try:
    import pillow_heif  # type: ignore
    pillow_heif.register_heif_opener()
except Exception:
    pass

from ..config import Config
from ..runtime import models_dir
from . import clip
from . import ocr as ocr_module


# --- Face detection ----------------------------------------------------------
# CLIP alone often calls a selfie in a dark room "a dark photo", or a group
# far away "a street".  A real detected face is a strong extra signal.
#
# Primary detector: YuNet (DNN, 2022) — bundled in OpenCV >= 4.5.4 via
# cv2.FaceDetectorYN_create.  ~232 KB ONNX, far better precision than Haar
# on documents / logos / patterns.
# Fallback: Haar cascades — only used if YuNet's ONNX is missing.

_YUNET_DETECTOR = None
_YUNET_FAILED = False
_FACE_CASCADES: Optional[List[cv2.CascadeClassifier]] = None


def _yunet_detector(score_threshold: float):
    """Lazily build the YuNet detector; cache one per process.  Returns None
    if the model file is missing or YuNet isn't available in this cv2 build.
    """
    global _YUNET_DETECTOR, _YUNET_FAILED
    if _YUNET_FAILED:
        return None
    if _YUNET_DETECTOR is not None:
        return _YUNET_DETECTOR
    model = models_dir() / "yunet_face.onnx"
    if not model.exists() or not hasattr(cv2, "FaceDetectorYN_create"):
        _YUNET_FAILED = True
        return None
    try:
        det = cv2.FaceDetectorYN_create(
            str(model),
            "",
            (320, 320),                   # placeholder; set per image
            score_threshold=score_threshold,
            nms_threshold=0.3,
            top_k=5000,
        )
    except Exception:
        _YUNET_FAILED = True
        return None
    _YUNET_DETECTOR = det
    return det


def _face_cascades() -> List[cv2.CascadeClassifier]:
    """Lazily load Haar cascades.  Used only as YuNet fallback."""
    global _FACE_CASCADES
    if _FACE_CASCADES is None:
        base = Path(cv2.data.haarcascades)
        xmls = [
            base / "haarcascade_frontalface_default.xml",
            base / "haarcascade_frontalface_alt2.xml",
            base / "haarcascade_profileface.xml",
        ]
        cascades = []
        for xml in xmls:
            if xml.exists():
                c = cv2.CascadeClassifier(str(xml))
                if not c.empty():
                    cascades.append(c)
        _FACE_CASCADES = cascades
    return _FACE_CASCADES


def _detect_faces_yunet(img: Image.Image, score_threshold: float) -> int:
    det = _yunet_detector(score_threshold)
    if det is None:
        return -1   # signal: YuNet unavailable, caller should fall back
    try:
        # Cap input size to ~640px on the long edge for speed; YuNet handles
        # arbitrary sizes but bigger images take longer for marginal gain.
        w, h = img.size
        if max(w, h) > 640:
            scale = 640.0 / max(w, h)
            img = img.resize((int(w * scale), int(h * scale)), Image.BILINEAR)
            w, h = img.size
        bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        det.setInputSize((w, h))
        _retval, faces = det.detect(bgr)
        if faces is None:
            return 0
        return int(len(faces))
    except Exception:
        return 0


def _detect_faces_haar(img: Image.Image) -> int:
    try:
        w, h = img.size
        if max(w, h) > 800:
            scale = 800.0 / max(w, h)
            img = img.resize((int(w * scale), int(h * scale)), Image.BILINEAR)
        gray = np.array(img.convert("L"))
        gray_flipped = cv2.flip(gray, 1)
        seen = 0
        for cascade in _face_cascades():
            for arr in (gray, gray_flipped):
                faces = cascade.detectMultiScale(
                    arr,
                    scaleFactor=1.15,
                    minNeighbors=8,
                    minSize=(60, 60),
                )
                seen += len(faces)
                if seen > 0:
                    return seen
        return 0
    except Exception:
        return 0


def _detect_faces(img: Image.Image, score_threshold: float) -> int:
    """Return number of faces scoring >= score_threshold.  Tries YuNet
    first, falls back to Haar."""
    n = _detect_faces_yunet(img, score_threshold)
    if n >= 0:
        return n
    return _detect_faces_haar(img)


# --- Metadata ----------------------------------------------------------------

_EXIF_MAKE = 271
_EXIF_IFD = 0x8769
_EXIF_USER_COMMENT = 37510
_TALL_RATIO = 1.9            # phone screens are 19.5:9 (2.17); photos <= 16:9


@dataclass
class ImageMeta:
    screenshot: bool          # metadata says "this is a screenshot"
    camera: bool              # has a camera Make -> taken by a camera
    tall: bool                # phone-screen aspect ratio


def _read_meta(path: Path, img: Image.Image) -> ImageMeta:
    camera = False
    comment = ""
    try:
        exif = img.getexif()
        camera = bool(str(exif.get(_EXIF_MAKE, "")).strip())
        raw = exif.get_ifd(_EXIF_IFD).get(_EXIF_USER_COMMENT, b"")
        comment = raw.decode("latin-1") if isinstance(raw, bytes) else str(raw)
    except Exception:
        pass
    xmp = img.info.get("xmp") or img.info.get("XML:com.adobe.xmp") or ""
    if isinstance(xmp, bytes):
        xmp = xmp.decode("utf-8", errors="ignore")
    name = path.name.lower()
    screenshot = (
        "screenshot" in comment.lower()
        or "<exif:UserComment>Screenshot" in xmp
        or "screenshot" in name
        or "screen shot" in name
    )
    w, h = img.size
    tall = max(w, h) >= _TALL_RATIO * min(w, h)
    return ImageMeta(screenshot=screenshot, camera=camera, tall=tall)


# --- decoding ----------------------------------------------------------------

_MAX_SIDE = 1024   # CLIP uses 224 px, YuNet 640 px; no need to keep more


def _load_image(path: Path) -> Tuple[Image.Image, ImageMeta]:
    img = Image.open(path)
    # JPEG only: decode directly at reduced scale (much faster for 12 MP).
    img.draft("RGB", (_MAX_SIDE, _MAX_SIDE))
    meta = _read_meta(path, img)
    img = ImageOps.exif_transpose(img)
    if img.mode != "RGB":
        img = img.convert("RGB")
    img.thumbnail((_MAX_SIDE, _MAX_SIDE), Image.BILINEAR)
    return img, meta


# --- scoring -----------------------------------------------------------------

def _folder_scores(
    content_scores: Dict[str, float],
    folders: Dict[str, List[str]],
) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for folder, ctypes in folders.items():
        out[folder] = sum(content_scores.get(c, 0.0) for c in ctypes)
    return out


def _folder_of(ctype: str, folders: Dict[str, List[str]]) -> Optional[str]:
    for folder, ctypes in folders.items():
        if ctype in ctypes:
            return folder
    return None


# --- result type -------------------------------------------------------------

@dataclass
class ImageResult:
    path: Path
    folder: str               # e.g. 'personal', 'screenshots', 'other'
    confidence: float         # 0..1+ (sum of contributing scores)
    tags: List[str] = field(default_factory=list)
    error: Optional[str] = None


# --- core classification used by both still-image and video paths -----------

def _classify_core(
    img: Image.Image,
    cfg: Config,
    *,
    meta: Optional[ImageMeta],
    folders: Dict[str, List[str]],
) -> Tuple[str, float, Dict[str, float], List[str]]:
    """Return (best_folder, best_score, content_scores, tags)."""
    tags: List[str] = []

    content_scores = clip.content_scores(img, cfg)
    top = max(content_scores, key=content_scores.get)
    tags.append(f"clip={top}({content_scores[top]:.2f})")

    shot = cfg.screenshot_content_type
    if meta is not None and shot in content_scores:
        if meta.camera:
            tags.append("camera-exif")
            content_scores[shot] = 0.0
        elif meta.tall:
            tags.append("tall-no-camera")
            content_scores[shot] += cfg.screenshot_tall_boost

    if cfg.face_detection_enabled:
        n_faces = _detect_faces(img, cfg.face_min_score)
        if n_faces > 0:
            tags.append(f"faces={n_faces}")
            target = cfg.face_target_content_type
            content_scores[target] = content_scores.get(target, 0.0) + cfg.face_boost

    if cfg.ocr_boost > 0:
        words = ocr_module.count_words(img)
        tags.append(f"ocr={words}")
        target = cfg.ocr_content_type
        content_scores[target] = (content_scores.get(target, 0.0)
                                  + cfg.ocr_boost * min(1.0, words / 30.0))

    folder_scores = _folder_scores(content_scores, folders)
    if not folder_scores:
        return cfg.fallback_folder, 0.0, content_scores, tags

    best_folder, best_score = max(folder_scores.items(), key=lambda kv: kv[1])
    return best_folder, float(best_score), content_scores, tags


# --- entry point used by workers --------------------------------------------

def classify(path: Path, cfg: Config) -> ImageResult:
    try:
        img, meta = _load_image(path)
    except Exception as e:
        return ImageResult(path=path, folder="unknown", confidence=0.0,
                           tags=["decode_error"], error=f"decode: {e}")
    return classify_loaded(path, img, meta, cfg)


def read_meta(path: Path, img: Image.Image) -> ImageMeta:
    """Metadata signals of a freshly opened (not yet transposed or
    converted) image.  Read it from the source file: conversion may lose
    the EXIF / XMP fields it looks at."""
    return _read_meta(path, img)


def classify_loaded(
    path: Path, img: Image.Image, meta: ImageMeta, cfg: Config
) -> ImageResult:
    """Classify an image that is already decoded (RGB, EXIF-oriented).
    `meta` should come from the original file (see read_meta)."""
    if cfg.screenshot_metadata and meta.screenshot:
        folder = _folder_of(cfg.screenshot_content_type, cfg.image_folders)
        if folder is not None:
            return ImageResult(path=path, folder=folder, confidence=1.0,
                               tags=["screenshot-meta"])

    try:
        best_folder, best_score, _scores, tags = _classify_core(
            img, cfg, meta=meta, folders=cfg.image_folders,
        )
    except Exception as e:
        return ImageResult(path=path, folder="unknown", confidence=0.0,
                           tags=["infer_error"], error=f"infer: {e}")

    if best_score < cfg.unknown_threshold:
        return ImageResult(
            path=path, folder=cfg.fallback_folder, confidence=float(best_score),
            tags=tags + ["below-threshold"],
        )

    return ImageResult(
        path=path, folder=best_folder, confidence=float(best_score), tags=tags,
    )


# --- video helper -----------------------------------------------------------

def classify_pil(
    img: Image.Image, cfg: Config
) -> Tuple[str, float, Dict[str, float], List[str]]:
    """Classify a single video frame.  Returns folder, score, content_scores, tags.
    No file metadata and no OCR for frames."""
    if img.mode != "RGB":
        img = img.convert("RGB")
    return _classify_core(img, cfg, meta=None, folders=cfg.video_folders)
