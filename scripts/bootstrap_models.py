"""
One-time model bootstrap.

Run this once on a machine with internet access to populate the `models/`
directory.  After it succeeds, the app is fully offline.

Produces:
    models/mobilenetv3.onnx        (image classifier, ~22 MB)
    models/imagenet_classes.json   (1000 class names in model order)
    models/silero_vad.onnx         (voice activity detector, ~2 MB)

Usage:
    python scripts/bootstrap_models.py
"""

from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = REPO_ROOT / "models"

SILERO_VAD_URL = (
    "https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/"
    "silero_vad.onnx"
)

# YuNet — DNN face detector, ~232 KB.  License: MIT.
YUNET_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/"
    "face_detection_yunet_2023mar.onnx"
)


def _download(url: str, dst: Path) -> None:
    print(f"  downloading {url}")
    print(f"        ->   {dst}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as response, dst.open("wb") as out:
        out.write(response.read())


def export_mobilenetv3(dst: Path, classes_dst: Path) -> None:
    """Export torchvision's MobileNetV3-Large to ONNX (FP32, opset 17)."""
    try:
        import torch
        from torchvision.models import (
            MobileNet_V3_Large_Weights,
            mobilenet_v3_large,
        )
    except ImportError:
        sys.exit(
            "Bootstrap requires torch + torchvision (build-time only).\n"
            "Install with:  pip install torch torchvision"
        )

    print("[1/3] Exporting MobileNetV3-Large -> ONNX")
    weights = MobileNet_V3_Large_Weights.IMAGENET1K_V2
    model = mobilenet_v3_large(weights=weights)
    model.eval()

    dst.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.randn(1, 3, 224, 224)
    torch.onnx.export(
        model,
        dummy,
        str(dst),
        input_names=["input"],
        output_names=["logits"],
        opset_version=17,
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
    )
    print(f"        wrote {dst}")

    print("[2/3] Writing ImageNet class names")
    categories = weights.meta["categories"]
    classes_dst.parent.mkdir(parents=True, exist_ok=True)
    with classes_dst.open("w", encoding="utf-8") as f:
        json.dump(categories, f, ensure_ascii=False, indent=2)
    print(f"        wrote {classes_dst} ({len(categories)} classes)")


def fetch_silero_vad(dst: Path) -> None:
    print("[3/4] Fetching silero-vad")
    _download(SILERO_VAD_URL, dst)


def fetch_yunet(dst: Path) -> None:
    print("[4/4] Fetching YuNet face detector")
    _download(YUNET_URL, dst)


def main() -> int:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    export_mobilenetv3(
        dst=MODELS_DIR / "mobilenetv3.onnx",
        classes_dst=MODELS_DIR / "imagenet_classes.json",
    )
    fetch_silero_vad(MODELS_DIR / "silero_vad.onnx")
    fetch_yunet(MODELS_DIR / "yunet_face.onnx")
    print("\nDone.  Models ready in", MODELS_DIR)
    return 0


if __name__ == "__main__":
    sys.exit(main())
