"""
ONNX Runtime session helpers.

Centralized so every classifier picks the right execution providers (GPU
first, CPU fallback) and we don't reload models per worker.

GPU: the app ships onnxruntime-directml.  DirectML runs on any DirectX 12
GPU (NVIDIA, AMD, Intel) with no CUDA / cuDNN install.  CUDA is still used
first when an onnxruntime build with CUDA and its DLLs are present.  If
the GPU session can't be created, the session falls back to CPU.
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


_CPU = "CPUExecutionProvider"
_CUDA = "CUDAExecutionProvider"
_DML = "DmlExecutionProvider"

# CPU threads per session.  We already saturate the box with
# multiprocessing, so more threads here would just thrash.
_INTRA_OP_THREADS = 1


def set_intra_op_threads(n: int) -> None:
    """Threads per ONNX session in this process.  Workers call this when
    RAM limits the pool to fewer processes than CPU cores, so the cores
    are still used.  Must run before the first session is created."""
    global _INTRA_OP_THREADS
    _INTRA_OP_THREADS = max(1, n)


def _session_options(directml: bool) -> ort.SessionOptions:
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.intra_op_num_threads = _INTRA_OP_THREADS
    opts.inter_op_num_threads = 1
    # 3 = ERROR.  Silences provider-not-loaded warnings.
    opts.log_severity_level = 3
    if directml:
        # Required by the DirectML provider.
        opts.enable_mem_pattern = False
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    return opts


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

    use_gpu ∈ {'auto', 'yes', 'no'}.  'auto' / 'yes': GPU first (CUDA when
    its DLLs are loadable, else DirectML), CPU after it; 'no' forces CPU.
    """
    if use_gpu == "no":
        return [_CPU]
    available = set(ort.get_available_providers())
    chosen: List[str] = []
    if _CUDA in available and _cuda_dlls_loadable():
        chosen.append(_CUDA)
    elif _DML in available:
        chosen.append(_DML)
    chosen.append(_CPU)
    return chosen


def make_session(model_path: Path, use_gpu: str) -> ort.InferenceSession:
    if not model_path.exists():
        raise FileNotFoundError(
            f"ONNX model not found: {model_path}\n"
            f"Run scripts/bootstrap_models.py once with internet access to "
            f"download it (or copy it manually into the models/ folder)."
        )
    providers = _providers_for(use_gpu)
    if providers != [_CPU]:
        try:
            return ort.InferenceSession(
                str(model_path), sess_options=_session_options(_DML in providers),
                providers=providers,
            )
        except Exception:
            pass        # no usable GPU (e.g. no DirectX 12 device): use the CPU
    return ort.InferenceSession(
        str(model_path), sess_options=_session_options(False), providers=[_CPU]
    )


def active_provider(session: ort.InferenceSession) -> str:
    p = session.get_providers()
    return p[0] if p else "unknown"


def describe_provider(provider: str) -> str:
    """Human-readable name of an execution provider."""
    return {
        _CUDA: "GPU (CUDA)",
        _DML: "GPU (DirectML)",
        _CPU: "CPU",
    }.get(provider, provider)


# --- per-process session cache (workers reuse one session per model) --------

_SESSIONS: dict = {}


def get_session(model_path: Path, use_gpu: str) -> ort.InferenceSession:
    key = (str(model_path), use_gpu)
    sess = _SESSIONS.get(key)
    if sess is None:
        sess = make_session(model_path, use_gpu)
        _SESSIONS[key] = sess
    return sess
