"""
Image classifier.

Pipeline per file:
  1. Decode (HEIC/HEIF supported via pillow_heif).
  2. EXIF-orient + center-crop + resize to 224x224.
  3. ONNX MobileNetV3-Large -> 1000-class softmax probabilities.
  4. OCR pass via Tesseract (always on).
  5. Face detection (Haar cascades, frontal + profile + mirror).
  6. Aggregate per content type:
       - keyword-driven types: sum probs of classes whose name contains any
         configured keyword (substring, case-insensitive).
       - 'paper': max(OCR-driven score, blank-paper-heuristic score).
       - 'human': boosted to face-boost if a face was found.
  7. Aggregate per folder (sum of its content-type scores).
  8. Winner = max-scoring folder; if max < unknown_threshold -> 'unknown'.

Returns an ImageResult with chosen folder + confidence + a `tags` list of
human-readable reasons that explain the decision (face detected, top
ImageNet class, OCR word count, blank-paper score, etc.).
"""

from __future__ import annotations

import json
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
from ..runtime import get_session, models_dir
from . import ocr as ocr_module


# --- Face detection ----------------------------------------------------------
# ImageNet has no "person" class — when shown a photo of a human the model
# fires on clothing / accessories / scene objects (drumstick, neck brace,
# bow tie, academic gown, wig, ...).  So we run a face detector in parallel;
# a positive face hit forces the configured face-target content type to a
# high score regardless of what ImageNet picked.
#
# Primary detector: YuNet (DNN, 2022) — bundled in OpenCV >= 4.5.4 via
# cv2.FaceDetectorYN_create.  ~232 KB ONNX, far better precision than Haar
# on documents / logos / patterns.
# Fallback: Haar cascades — only used if YuNet's ONNX is missing.

_YUNET_DETECTOR = None
_YUNET_FAILED = False
_FACE_CASCADES: Optional[List[cv2.CascadeClassifier]] = None
_YUNET_SCORE_THRESHOLD = 0.7


def _yunet_detector():
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
            score_threshold=_YUNET_SCORE_THRESHOLD,
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


def _detect_faces_yunet(img: Image.Image) -> int:
    det = _yunet_detector()
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


def _detect_faces(img: Image.Image) -> int:
    """Return number of detected faces.  Tries YuNet first, falls back to Haar."""
    n = _detect_faces_yunet(img)
    if n >= 0:
        return n
    return _detect_faces_haar(img)


# --- Blank / rectangular paper heuristic -------------------------------------
# Detects pages, bills, receipts and similar mostly-text-on-white-paper
# images that may not OCR cleanly (e.g. handwriting, low contrast, foreign
# script).  Signal:
#     * High mean luminance (mostly bright pixels)
#     * Low color saturation (mostly grey/black/white)
#     * Smooth interior (low pixel variance), to exclude bright nature shots
# All thresholds picked to avoid firing on snow scenes, blue sky photos,
# overexposed selfies, etc.

def _blank_paper_score(img: Image.Image) -> float:
    try:
        w, h = img.size
        if max(w, h) > 600:
            scale = 600.0 / max(w, h)
            img_small = img.resize((int(w * scale), int(h * scale)), Image.BILINEAR)
        else:
            img_small = img
        rgb = np.asarray(img_small, dtype=np.float32)
        if rgb.ndim != 3 or rgb.shape[2] < 3:
            return 0.0
        # HSV via cv2 (expects BGR)
        bgr = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2BGR)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        sat = hsv[:, :, 1].astype(np.float32) / 255.0
        val = hsv[:, :, 2].astype(np.float32) / 255.0

        mean_val = float(val.mean())
        mean_sat = float(sat.mean())
        std_val = float(val.std())

        # Three sub-scores in [0, 1].  Multiply for the final score so all
        # three must agree before paper fires.
        bright = max(0.0, (mean_val - 0.55) / 0.35)         # 0.55 -> 0,  0.90 -> 1
        bright = min(bright, 1.0)
        unsat = max(0.0, (0.30 - mean_sat) / 0.30)          # 0.30 -> 0, 0.00 -> 1
        unsat = min(unsat, 1.0)
        flat = max(0.0, (0.25 - std_val) / 0.25)            # noisy -> 0, very flat -> 1
        flat = min(flat, 1.0)

        return bright * unsat * flat
    except Exception:
        return 0.0


# --- ImageNet model + class list (lazy, per-process) ------------------------

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
_INPUT_SIZE = 224

_CLASS_NAMES: Optional[List[str]] = None


def _load_class_names() -> List[str]:
    global _CLASS_NAMES
    if _CLASS_NAMES is None:
        path = models_dir() / "imagenet_classes.json"
        if not path.exists():
            raise FileNotFoundError(
                f"Class-name file missing: {path}\n"
                f"Run scripts/bootstrap_models.py to fetch it."
            )
        with path.open("r", encoding="utf-8") as f:
            _CLASS_NAMES = [name.lower() for name in json.load(f)]
    return _CLASS_NAMES


def _model_path() -> Path:
    return models_dir() / "mobilenetv3.onnx"


# --- preprocessing -----------------------------------------------------------

def _load_image(path: Path) -> Image.Image:
    img = Image.open(path)
    img = ImageOps.exif_transpose(img)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


def _preprocess(img: Image.Image) -> np.ndarray:
    w, h = img.size
    short = min(w, h)
    scale = _INPUT_SIZE / short
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    img = img.resize((new_w, new_h), Image.BILINEAR)
    left = (new_w - _INPUT_SIZE) // 2
    top = (new_h - _INPUT_SIZE) // 2
    img = img.crop((left, top, left + _INPUT_SIZE, top + _INPUT_SIZE))

    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - _IMAGENET_MEAN) / _IMAGENET_STD
    arr = np.transpose(arr, (2, 0, 1))                 # HWC -> CHW
    arr = np.expand_dims(arr, 0).astype(np.float32)    # NCHW
    return arr


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


# --- scoring -----------------------------------------------------------------

def _content_type_scores(
    probs: np.ndarray,
    classes: List[str],
    keywords: Dict[str, List[str]],
) -> Dict[str, float]:
    scores: Dict[str, float] = {}
    for ctype, kws in keywords.items():
        if not kws:
            scores[ctype] = 0.0
            continue
        total = 0.0
        for i, name in enumerate(classes):
            if any(kw in name for kw in kws):
                total += float(probs[i])
        scores[ctype] = total
    return scores


def _folder_scores(
    content_scores: Dict[str, float],
    folders: Dict[str, List[str]],
) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for folder, ctypes in folders.items():
        out[folder] = sum(content_scores.get(c, 0.0) for c in ctypes)
    return out


# --- result type -------------------------------------------------------------

@dataclass
class ImageResult:
    path: Path
    folder: str               # e.g. 'personal', 'docs', 'unknown'
    confidence: float         # 0..1+ (sum of contributing scores)
    tags: List[str] = field(default_factory=list)
    error: Optional[str] = None


# --- core classification used by both still-image and video paths -----------

def _classify_core(
    img: Image.Image,
    cfg: Config,
    *,
    run_ocr: bool,
    folders: Dict[str, List[str]],
) -> Tuple[str, float, Dict[str, float], List[str]]:
    """Return (best_folder, best_score, content_scores, tags)."""
    tags: List[str] = []

    session = get_session(_model_path(), cfg.use_gpu)
    x = _preprocess(img)
    input_name = session.get_inputs()[0].name
    out = session.run(None, {input_name: x})[0][0]
    probs = _softmax(out)
    classes = _load_class_names()

    # Top class for the explanatory tag.
    top_idx = int(np.argmax(probs))
    tags.append(f"top={classes[top_idx]}({probs[top_idx]:.2f})")

    content_scores = _content_type_scores(probs, classes, cfg.keywords)

    # Paper signal: OCR words OR blank-paper appearance heuristic.
    paper_score = 0.0
    ocr_words = 0
    if run_ocr:
        try:
            ocr_words = ocr_module.count_words(img)
        except Exception:
            ocr_words = 0
        if ocr_words > 0:
            paper_score = max(paper_score, min(1.0, ocr_words / 30.0))
            tags.append(f"ocr={ocr_words}")
        else:
            tags.append("ocr=0")

    blank_score = _blank_paper_score(img)
    if blank_score > 0.05:
        tags.append(f"blank={blank_score:.2f}")
    paper_score = max(paper_score, blank_score)

    # If the user defined `keywords-paper`, let those classes also contribute
    # to the paper signal.  E.g. envelope, menu, book_jacket all strongly
    # suggest a document even when OCR misses the text.
    keyword_paper_score = content_scores.get("paper", 0.0)
    if keyword_paper_score > 0:
        tags.append(f"paper-kw={keyword_paper_score:.2f}")
    paper_score = max(paper_score, keyword_paper_score)

    # Face detection.  YuNet is precise enough that we trust any face hit:
    # if a real face is in the image, it's a personal photo, full stop —
    # even if there's also a lot of OCR text (people in restaurants, group
    # shots in front of signs, selfies with menus).  Paper signal is
    # zeroed out when faces are present.
    n_faces = 0
    if cfg.face_detection_enabled:
        n_faces = _detect_faces(img)
        if n_faces > 0:
            tags.append(f"faces={n_faces}")
            target = cfg.face_target_content_type
            boost = cfg.face_boost
            content_scores[target] = max(content_scores.get(target, 0.0), boost)
            if paper_score > 0:
                tags.append("paper-suppressed-by-face")
                paper_score = 0.0

    content_scores["paper"] = paper_score

    folder_scores = _folder_scores(content_scores, folders)
    if not folder_scores:
        return "unknown", 0.0, content_scores, tags

    best_folder, best_score = max(folder_scores.items(), key=lambda kv: kv[1])
    return best_folder, float(best_score), content_scores, tags


# --- entry point used by workers --------------------------------------------

def classify(path: Path, cfg: Config) -> ImageResult:
    try:
        img = _load_image(path)
    except Exception as e:
        return ImageResult(path=path, folder="unknown", confidence=0.0,
                           tags=[f"decode_error"], error=f"decode: {e}")

    try:
        best_folder, best_score, _scores, tags = _classify_core(
            img, cfg, run_ocr=True, folders=cfg.image_folders,
        )
    except Exception as e:
        return ImageResult(path=path, folder="unknown", confidence=0.0,
                           tags=["infer_error"], error=f"infer: {e}")

    if best_score < cfg.unknown_threshold:
        return ImageResult(
            path=path, folder="unknown", confidence=float(best_score),
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
    Skips OCR (rarely useful inside a video, expensive at 5x per file)."""
    if img.mode != "RGB":
        img = img.convert("RGB")
    return _classify_core(img, cfg, run_ocr=False, folders=cfg.video_folders)
