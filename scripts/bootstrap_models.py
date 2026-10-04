"""
One-time model bootstrap.

Run this once on a machine with internet access to populate the `models/`
directory.  After it succeeds, the app is fully offline.

Produces:
    models/clip/vision_model.onnx  (CLIP ViT-B/32 image encoder, ~335 MB)
    models/clip/text_model.onnx    (CLIP ViT-B/32 text encoder, ~242 MB)
    models/clip/tokenizer.json     (CLIP BPE tokenizer)
    models/silero_vad.onnx         (voice activity detector, ~2 MB)
    models/yunet_face.onnx         (face detector, ~232 KB)

The CLIP files are larger than GitHub's 100 MB limit, so they are not in
git (see .gitignore).  Files that already exist with the right size are
skipped, and interrupted downloads resume.

Usage:
    python scripts/bootstrap_models.py
"""

from __future__ import annotations

import sys
import time
import urllib.request
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = REPO_ROOT / "models"

# OpenAI CLIP ViT-B/32, ONNX export by Xenova (fp32).  License: MIT.
# fp32 on purpose: the int8 quantized vision model scored ~3 points lower on
# test/holdout, and the fp16 one does not load in onnxruntime 1.23.
CLIP_BASE = "https://huggingface.co/Xenova/clip-vit-base-patch32/resolve/main/"
CLIP_FILES = {
    "onnx/vision_model.onnx": "clip/vision_model.onnx",
    "onnx/text_model.onnx": "clip/text_model.onnx",
    "tokenizer.json": "clip/tokenizer.json",
}

SILERO_VAD_URL = (
    "https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/"
    "silero_vad.onnx"
)

# YuNet — DNN face detector, ~232 KB.  License: MIT.
YUNET_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/"
    "face_detection_yunet_2023mar.onnx"
)


def _remote_size(url: str) -> int:
    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req) as response:
        return int(response.headers.get("Content-Length", 0))


def _download(url: str, dst: Path, retries: int = 20) -> None:
    """Download `url` to `dst`, resuming with HTTP Range after a dropped
    connection, until the local size matches the remote size."""
    print(f"  downloading {url}")
    print(f"        ->   {dst}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    want = _remote_size(url)
    for attempt in range(retries):
        have = dst.stat().st_size if dst.exists() else 0
        if want and have == want:
            return
        if have > want:
            dst.unlink()
            have = 0
        req = urllib.request.Request(url)
        if have:
            req.add_header("Range", f"bytes={have}-")
        try:
            with urllib.request.urlopen(req, timeout=60) as response, \
                    dst.open("ab" if have else "wb") as out:
                while chunk := response.read(1 << 20):
                    out.write(chunk)
        except OSError as e:
            print(f"        connection dropped ({e}); retry {attempt + 1}")
            time.sleep(2)
        if not want:
            return
    have = dst.stat().st_size if dst.exists() else 0
    if have != want:
        sys.exit(f"Download failed: {url} ({have} of {want} bytes)")


def main() -> int:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    print("[1/3] Fetching CLIP ViT-B/32")
    for remote, local in CLIP_FILES.items():
        _download(CLIP_BASE + remote, MODELS_DIR / local)
    print("[2/3] Fetching silero-vad")
    _download(SILERO_VAD_URL, MODELS_DIR / "silero_vad.onnx")
    print("[3/3] Fetching YuNet face detector")
    _download(YUNET_URL, MODELS_DIR / "yunet_face.onnx")
    print("\nDone.  Models ready in", MODELS_DIR)
    return 0


if __name__ == "__main__":
    sys.exit(main())
