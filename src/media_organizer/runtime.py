"""
ONNX Runtime session helpers.

Centralized so every classifier picks the right execution providers (GPU
first, CPU fallback) and we don't reload models per worker.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional

import onnxruntime as ort

# Silence the global ORT logger (which fires before we ever build a session,
# e.g. provider-DLL-not-found warnings).  3 = ERROR.
try:
    ort.set_default_logger_severity(3)
except AttributeError:
    pass


_DEFAULT_OPTS = ort.SessionOptions()
_DEFAULT_OPTS.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
# Let ORT pick a sensible thread count; we already saturate the box with
# multiprocessing, so over-subscription threads here would just thrash.
_DEFAULT_OPTS.intra_op_num_threads = 1
_DEFAULT_OPTS.inter_op_num_threads = 1
# 3 = ERROR.  Silences the noisy CUDA-provider-not-loaded warnings that
# trigger when onnxruntime-gpu is installed but cuDNN/CUDA are missing.
_DEFAULT_OPTS.log_severity_level = 3


def app_root() -> Path:
    """Return the folder that contains the app's bundled resources.

    When frozen by PyInstaller (--onedir), `sys._MEIPASS` is set; we want the
    folder that holds media-organizer.exe so models/ sits next to it.
    """
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    # Source layout: <repo>/src/media_organizer/runtime.py -> <repo>
    return Path(__file__).resolve().parents[2]


def models_dir() -> Path:
    return app_root() / "models"


def third_party_dir() -> Path:
    return app_root() / "third_party"


def _cuda_dlls_loadable() -> bool:
    """Best-effort check that the CUDA + cuDNN DLLs ORT needs are present.

    ort's CUDA provider depends on cublasLt64_12.dll (CUDA 12 runtime) and
    cudnn64_9.dll (cuDNN 9).  When either is missing on Windows, the C++
    side prints a long red error to stderr that we can't easily suppress,
    so we'd rather skip the provider entirely.
    """
    if sys.platform != "win32":
        return True   # don't second-guess on Linux/macOS
    import ctypes
    for dll in ("cublasLt64_12.dll", "cudnn64_9.dll"):
        try:
            ctypes.WinDLL(dll)
        except OSError:
            return False
    return True


def _providers_for(use_gpu: str) -> List[str]:
    """Return the ORT provider list given the user's preference.

    use_gpu ∈ {'auto', 'yes', 'no'}.  'auto' tries CUDA when its DLLs are
    loadable, falls back to CPU otherwise; 'yes' is currently identical to
    'auto' (we don't crash); 'no' forces CPU.
    """
    available = set(ort.get_available_providers())
    cpu = "CPUExecutionProvider"
    cuda = "CUDAExecutionProvider"
    dml = "DmlExecutionProvider"   # Windows DirectML, useful on AMD/Intel GPUs

    if use_gpu == "no":
        return [cpu]

    chosen: List[str] = []
    if cuda in available and _cuda_dlls_loadable():
        chosen.append(cuda)
    if dml in available and dml not in chosen:
        chosen.append(dml)
    chosen.append(cpu)
    return chosen


def make_session(model_path: Path, use_gpu: str) -> ort.InferenceSession:
    if not model_path.exists():
        raise FileNotFoundError(
            f"ONNX model not found: {model_path}\n"
            f"Run scripts/bootstrap_models.py once with internet access to "
            f"download it (or copy it manually into the models/ folder)."
        )
    providers = _providers_for(use_gpu)
    return ort.InferenceSession(
        str(model_path), sess_options=_DEFAULT_OPTS, providers=providers
    )


def active_provider(session: ort.InferenceSession) -> str:
    p = session.get_providers()
    return p[0] if p else "unknown"


# --- per-process session cache (workers reuse one session per model) --------

_SESSIONS: dict = {}


def get_session(model_path: Path, use_gpu: str) -> ort.InferenceSession:
    key = (str(model_path), use_gpu)
    sess = _SESSIONS.get(key)
    if sess is None:
        sess = make_session(model_path, use_gpu)
        _SESSIONS[key] = sess
    return sess
