"""
Audio classifier.

Stage 1 — Voice detection (silero-vad ONNX):
    Run VAD over the entire audio at 16 kHz mono.  Compute a per-window
    speech probability and derive `speech_ratio = #frames(prob>0.5) / total`.

Stage 2 — Routing:
    speech_ratio >= talk_min                    -> audio/talk
    speech_ratio <= music_max:
        - librosa.beat.beat_track for tempo (BPM) and beat strength
        - if beat is too weak                    -> audio/music/unknown-rythm
        - else map BPM to configured rhythm band -> audio/music/<band>
    everything in between (mixed)                -> audio/unknown
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from ..config import Config
from ..runtime import get_session, models_dir


_TARGET_SR = 16000
_VAD_WINDOW = 512          # silero-vad expects 512 samples @ 16k (32 ms)
_VAD_THRESHOLD = 0.5
_BEAT_STRENGTH_MIN = 0.05  # below this we call it 'no clear rhythm'


@dataclass
class AudioResult:
    path: Path
    folder: str               # 'talk' | 'music/<band>' | 'music/unknown-rythm' | 'unknown'
    confidence: float
    error: Optional[str] = None


def _silero_path() -> Path:
    return models_dir() / "silero_vad.onnx"


def _load_audio(path: Path) -> np.ndarray:
    """Return mono float32 audio at 16 kHz, range [-1, 1]."""
    # librosa handles a wide format set via soundfile + audioread, and will
    # transparently fall back to ffmpeg when present on PATH.
    import librosa  # local import: keeps cold start cheap when audio absent
    y, _sr = librosa.load(str(path), sr=_TARGET_SR, mono=True)
    return y.astype(np.float32, copy=False)


def _vad_speech_ratio(y: np.ndarray, cfg: Config) -> float:
    """Return the fraction of windows where speech-prob > threshold.

    Uses silero-vad's stateful ONNX graph.  We pad to a multiple of window
    size and feed windows one at a time; this is fast enough on CPU.
    """
    sess = get_session(_silero_path(), cfg.use_gpu)
    # silero-vad ONNX inputs: input(1xWindow), state(2x1x128), sr(int64)
    state = np.zeros((2, 1, 128), dtype=np.float32)
    sr = np.array(_TARGET_SR, dtype=np.int64)

    n = len(y)
    if n < _VAD_WINDOW:
        return 0.0
    pad = (-n) % _VAD_WINDOW
    if pad:
        y = np.pad(y, (0, pad))
    windows = y.reshape(-1, _VAD_WINDOW)

    speech = 0
    for w in windows:
        inp = w.astype(np.float32, copy=False).reshape(1, _VAD_WINDOW)
        probs, state = sess.run(None, {"input": inp, "state": state, "sr": sr})
        if float(probs[0][0]) > _VAD_THRESHOLD:
            speech += 1
    return speech / float(len(windows))


def _tempo(y: np.ndarray) -> tuple[float, float]:
    """Return (bpm, beat_strength).  beat_strength is the mean onset
    envelope over beat frames -- a rough proxy for 'how rhythmic is this'.
    """
    import librosa
    onset_env = librosa.onset.onset_strength(y=y, sr=_TARGET_SR)
    if onset_env.size == 0:
        return 0.0, 0.0
    tempo, beats = librosa.beat.beat_track(
        onset_envelope=onset_env, sr=_TARGET_SR
    )
    if len(beats) == 0:
        beat_strength = float(onset_env.mean())
    else:
        beat_strength = float(onset_env[beats].mean())
    # librosa returns tempo as a 0-d array on newer versions; coerce to float.
    return float(np.atleast_1d(tempo)[0]), beat_strength


def _bpm_band(bpm: float, cfg: Config) -> Optional[str]:
    for band, (lo, hi) in cfg.rhythm_bands.items():
        if lo <= bpm <= hi:
            return band
    return None


def classify(path: Path, cfg: Config) -> AudioResult:
    try:
        y = _load_audio(path)
    except Exception as e:
        return AudioResult(path=path, folder="unknown", confidence=0.0, error=f"decode: {e}")

    if y.size < _TARGET_SR:   # less than 1 second of audio
        return AudioResult(path=path, folder="unknown", confidence=0.0, error="too short")

    try:
        speech_ratio = _vad_speech_ratio(y, cfg)
    except Exception as e:
        return AudioResult(path=path, folder="unknown", confidence=0.0, error=f"vad: {e}")

    if speech_ratio >= cfg.audio_talk_min_speech_ratio:
        return AudioResult(path=path, folder="talk", confidence=float(speech_ratio))

    if speech_ratio <= cfg.audio_music_max_speech_ratio:
        try:
            bpm, beat_strength = _tempo(y)
        except Exception:
            bpm, beat_strength = 0.0, 0.0

        if beat_strength < _BEAT_STRENGTH_MIN or bpm <= 0:
            return AudioResult(
                path=path,
                folder="music/unknown-rythm",
                confidence=float(beat_strength),
            )
        band = _bpm_band(bpm, cfg)
        if band is None:
            return AudioResult(
                path=path,
                folder="music/unknown-rythm",
                confidence=float(beat_strength),
            )
        return AudioResult(
            path=path,
            folder=f"music/{band}",
            confidence=float(beat_strength),
        )

    # Mixed (some speech, some not) -> unknown bucket.
    return AudioResult(path=path, folder="unknown", confidence=float(speech_ratio))
