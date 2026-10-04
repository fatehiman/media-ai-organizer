"""
CLIP zero-shot scorer (OpenAI CLIP ViT-B/32, ONNX export by Xenova).

CLIP maps an image and a sentence into the same vector space; the cosine
similarity between them says how well the sentence describes the image.
So instead of a fixed list of 1000 ImageNet classes we can ask about our
own categories with plain-English prompts from the config:

    prompts-screenshot = a screenshot of a phone, a screenshot of a chat app
    prompts-document   = a photo of a document, a receipt, ...

Score of a content type = sum of softmax(100 * cosine) over its prompts
(100 is CLIP's trained logit scale).  The scores of all types sum to 1.

Text embeddings are computed once (in the main process, see
workers.classify_all) and handed to the workers, so the 250 MB text model
is never loaded more than once per run.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from ..config import Config
from ..runtime import get_session, make_session, models_dir


_INPUT_SIZE = 224
_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
_STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
_LOGIT_SCALE = 100.0

# (content_type per prompt, prompt embeddings [n_prompts, 512]); per process.
_TEXT: Optional[Tuple[List[str], np.ndarray]] = None


def clip_dir() -> Path:
    return models_dir() / "clip"


def vision_model_path() -> Path:
    return clip_dir() / "vision_model.onnx"


def _normalize(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def compute_text_embeddings(cfg: Config) -> Tuple[List[str], np.ndarray]:
    """Embed every `prompts-<type>` sentence.  Loads the text model + the
    tokenizer, so call this once and pass the result to the workers."""
    from tokenizers import Tokenizer

    tok_path = clip_dir() / "tokenizer.json"
    if not tok_path.exists():
        raise FileNotFoundError(
            f"CLIP tokenizer missing: {tok_path}\n"
            f"Run scripts/bootstrap_models.py to fetch it."
        )
    tokenizer = Tokenizer.from_file(str(tok_path))
    # Text model runs a few dozen times per run: CPU is plenty.
    session = make_session(clip_dir() / "text_model.onnx", "no")

    types: List[str] = []
    embeds: List[np.ndarray] = []
    for ctype, prompts in cfg.prompts.items():
        for prompt in prompts:
            ids = np.array([tokenizer.encode(prompt).ids], dtype=np.int64)
            types.append(ctype)
            embeds.append(session.run(None, {"input_ids": ids})[0][0])
    if not embeds:
        raise ValueError("Config defines no `prompts-<type>` lines")
    return types, _normalize(np.stack(embeds).astype(np.float32))


def set_text_embeddings(text: Tuple[List[str], np.ndarray]) -> None:
    global _TEXT
    _TEXT = text


def _text_embeddings(cfg: Config) -> Tuple[List[str], np.ndarray]:
    global _TEXT
    if _TEXT is None:
        _TEXT = compute_text_embeddings(cfg)
    return _TEXT


def _preprocess(img: Image.Image) -> np.ndarray:
    """Pad to a square (grey bars) then resize to 224x224.

    Padding instead of CLIP's usual center-crop keeps the whole frame:
    tall phone screenshots lose their status bar / app chrome when cropped,
    and that chrome is exactly what makes them recognizable.
    """
    w, h = img.size
    side = max(w, h)
    square = Image.new("RGB", (side, side), (128, 128, 128))
    square.paste(img, ((side - w) // 2, (side - h) // 2))
    square = square.resize((_INPUT_SIZE, _INPUT_SIZE), Image.BICUBIC)
    arr = (np.asarray(square, dtype=np.float32) / 255.0 - _MEAN) / _STD
    return np.transpose(arr, (2, 0, 1))[None].astype(np.float32)   # NCHW


def content_scores(img: Image.Image, cfg: Config) -> Dict[str, float]:
    """Return {content_type: probability}; probabilities sum to 1."""
    types, text = _text_embeddings(cfg)
    session = get_session(vision_model_path(), cfg.use_gpu)
    image = session.run(None, {"pixel_values": _preprocess(img)})[0][0]
    sims = text @ _normalize(image.astype(np.float32))
    z = np.exp(_LOGIT_SCALE * (sims - sims.max()))
    z /= z.sum()
    scores: Dict[str, float] = {}
    for ctype, p in zip(types, z):
        scores[ctype] = scores.get(ctype, 0.0) + float(p)
    return scores
