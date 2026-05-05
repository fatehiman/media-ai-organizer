# Tech Reference — Media Organizer

Internal-facing notes: setup, design decisions, library quirks, build
pitfalls, and links worth keeping. Companion to README.MD (which is
user-facing).

---

## Setup from a clean machine

If you cloned the repo onto a fresh Windows 11 box, this is the full
end-to-end recipe. Aimed at being followable by an AI agent or by a
human with a terminal.

### 0. Prerequisites

| Tool | Version | URL |
|---|---|---|
| Python | 3.10.x | <https://www.python.org/downloads/release/python-31011/> |
| Git | any | <https://git-scm.com/download/win> |

Optional, only if you want CUDA acceleration on an NVIDIA GPU:

| Tool | Version | URL |
|---|---|---|
| CUDA Toolkit | 12.x | <https://developer.nvidia.com/cuda-12-4-0-download-archive> |
| cuDNN | 9.x | <https://developer.nvidia.com/cudnn-downloads> |

Without CUDA the app runs on CPU automatically — no errors, just slower.

### 1. Clone + venv

```powershell
git clone https://github.com/<your-user>/media-ai-organizer.git
cd media-ai-organizer
python -m venv .venv
.\.venv\Scripts\Activate.ps1   # or: powershell -ExecutionPolicy Bypass
.\.venv\Scripts\python.exe -m pip install --upgrade pip
```

### 2. Install Python runtime deps

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

`onnxruntime-gpu` is large (~1 GB). If you don't need CUDA, swap it for
`onnxruntime` (CPU-only, ~150 MB) by editing `requirements.txt`.

### 3. (Optional, build-time only) Install torch + torchvision

Only needed if you want to **re-export** MobileNetV3 ONNX from scratch via
`scripts/bootstrap_models.py`. The committed `models/mobilenetv3.onnx`
is already exported — skip this step unless you specifically need to
regenerate it.

```powershell
.\.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install onnxscript
$env:PYTHONIOENCODING = "utf-8"   # avoids cp1252 emoji crash from torch.onnx
.\.venv\Scripts\python.exe scripts\bootstrap_models.py
```

The committed `models/` folder (24 MB) contains:

| File | Size | Source |
|---|---|---|
| `mobilenetv3.onnx` | 347 KB graph | torchvision `MobileNet_V3_Large_Weights.IMAGENET1K_V2` |
| `mobilenetv3.onnx.data` | 22 MB weights | (external-data sidecar) |
| `imagenet_classes.json` | 16 KB | torchvision `WEIGHTS.meta["categories"]` |
| `silero_vad.onnx` | 2.3 MB | <https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx> |
| `yunet_face.onnx` | 232 KB | <https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx> |

### 4. Install bundled binaries (Tesseract)

Tesseract is excluded from the repo (239 MB total; one DLL is 97 MB).
Install once and either copy the install dir into `third_party/tesseract/`
or just leave it system-wide — the app finds it at
`C:\Program Files\Tesseract-OCR\tesseract.exe` by default.

```powershell
winget install UB-Mannheim.TesseractOCR --accept-source-agreements --accept-package-agreements --silent
```

For a portable build (the .exe must work on machines without Tesseract
installed), copy the install directory into the repo:

```powershell
Copy-Item -Recurse "C:\Program Files\Tesseract-OCR\*" "third_party\tesseract\"
```

`third_party/tesseract/tesseract.exe` is then preferred over the
system install at runtime.

### 5. (Optional) ffmpeg

Not required at runtime — OpenCV's bundled decoder handles MP4, MOV,
AVI, MKV, WebM. Only add ffmpeg if you encounter an exotic codec cv2
can't open.

```
Source: https://www.gyan.dev/ffmpeg/builds/   (essentials LGPL build)
Drop into:  third_party\ffmpeg\bin\ffmpeg.exe
            third_party\ffmpeg\bin\ffprobe.exe
```

### 6. Configure paths and run

Edit `media-organizer.conf`:

```
source = E:\path\to\unsorted
target = E:\path\to\organized
```

Run from source:

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe -m media_organizer
.\.venv\Scripts\python.exe -m media_organizer --apply --yes   # to actually move
```

### 7. Build the portable .exe

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\build.ps1
```

Output: `dist\MediaOrganizer\` (~1.4 GB unzipped). Copy that folder
anywhere — fully portable.

---

## Models

### MobileNetV3-Large (image classifier)

- **Source**: torchvision `MobileNet_V3_Large_Weights.IMAGENET1K_V2`.
- **Export**: `scripts/bootstrap_models.py` → ONNX opset 17, FP32, with
  `dynamic_axes={"input": {0: "batch"}}`.
- **Input**: NCHW, 3×224×224, normalized with ImageNet mean/std.
- **Output**: 1000-logit vector. We softmax it ourselves.
- **External data**: torch's exporter automatically writes an
  `mobilenetv3.onnx.data` sidecar holding the 22 MB of weights. Both
  files must travel together; ORT loads the data file by name.
- **Class names**: from `weights.meta["categories"]`, frozen in
  `models/imagenet_classes.json`. The order MUST match the model's
  output ordering — never re-derive these from a different source.

### YuNet (face detector)

- **Source**: <https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx>. License: MIT.
- **API**: `cv2.FaceDetectorYN_create` (bundled in `opencv-python` ≥ 4.5.4).
- **Input size**: arbitrary; we cap to 640px on the long edge for speed.
- **Score threshold**: 0.7 (set in `image.py`).
- **Why over Haar cascades**: Haar produces many false positives on
  documents, logos, and patterned objects. YuNet is a small DNN
  trained on actual face data and almost never fires on non-face
  patterns. Haar is kept as a fallback.

### silero-vad (voice activity detection)

- **Source**: <https://github.com/snakers4/silero-vad>. License: MIT.
- **URL**: `src/silero_vad/data/silero_vad.onnx` on `master`. If upstream
  restructures, update the URL in `scripts/bootstrap_models.py`.
- **API contract**: inputs `input` (1×512 float32), `state` (2×1×128
  float32), `sr` (int64 scalar). Outputs `probs` (1×1) and a new `state`.
  Stateful — must thread the state across consecutive windows.
- **Window size**: 512 samples at 16 kHz = 32 ms. Other sizes are
  unsupported.
- **Speech threshold**: 0.5; tweakable in `audio.py`.

### Tesseract OCR

- **Mode**: `--psm 6` ("single uniform block of text") — faster than
  PSM 3 (auto-segment) and adequate for the binary "is this a document?"
  question.
- **Word counting**: uses `pytesseract.image_to_data(...)` rather than
  `image_to_string()`. Per-word `conf` is checked; words below
  `_MIN_CONF = 60` (in `ocr.py`) are dropped. This kills hallucinated
  pseudo-words on textured surfaces (concrete, fabric, tree bark).
  Tokens still must match the regex `[A-Za-zÀ-ɏ]{2,}`.
- **Threshold to fire `paper`**: `min(1.0, words / 30.0)` per file.
- **Tessdata**: only `eng.traineddata` is needed for the count
  heuristic. Adding more languages is fine but slows startup.
- **Bundle locations searched, in order**:
  1. `third_party/tesseract/tesseract.exe`
  2. `C:\Program Files\Tesseract-OCR\tesseract.exe`
  3. system PATH

---

## ONNX Runtime: provider selection

- We use `onnxruntime-gpu` (the wheel includes CPU). Provider list is
  built dynamically at `make_session()` time:
  `[CUDAExecutionProvider?, DmlExecutionProvider?, CPUExecutionProvider]`.
- **DLL probe**: before adding `CUDAExecutionProvider`, we try to
  `ctypes.WinDLL("cudnn64_9.dll")` and `cublasLt64_12.dll`. If either
  fails to load, CUDA is silently dropped. This avoids the noisy
  red-text warnings ORT prints when the DLLs are missing.
- **Per-process sessions**: each multiprocessing worker creates its
  own session lazily. Sharing one session across processes is
  impossible (CUDA contexts can't be forked safely on Windows). Multiple
  sessions on the same GPU are fine for small models.
- **Thread settings**: `intra_op_num_threads = 1` and
  `inter_op_num_threads = 1`. We saturate the box with multiprocessing,
  so ORT-internal threads would just thrash.
- **Logger**: `ort.set_default_logger_severity(3)` at module load
  silences info/warnings; only errors print.

---

## Image classification scoring

For each image:

1. Run MobileNetV3 → softmax → 1000-class probabilities.
2. For each `keywords-<type>` list in conf, sum probabilities of classes
   whose names contain any keyword (substring, case-insensitive).
3. **Paper signal**: `max(keyword_paper_score, ocr_words/30, blank_score)`.
4. **Face boost**: if YuNet finds ≥1 face, set
   `content_scores[face_target] = max(current, face_boost)` and zero out
   `paper`. A real face beats text-in-background.
5. **Folder scoring**: for each `image-<folder>` in conf, sum its
   content-type scores.
6. Best folder wins; if best score < `unknown-threshold` → `images/unknown/`.

The blank-paper heuristic (in `image.py::_blank_paper_score`) checks:

- Mean luminance > 0.55 (bright)
- Mean saturation < 0.30 (low color)
- Std of value < 0.25 (smooth surface)

All three must agree (multiplicative); thresholds picked to avoid firing
on snow scenes, blue sky, overexposed selfies.

---

## Multiprocessing on Windows

- We use `concurrent.futures.ProcessPoolExecutor` with an `initializer`
  that stashes the parsed `Config` in a module global of each worker.
- **Spawn**, not fork: Windows has no fork. The entry point must be
  guarded by `if __name__ == "__main__":` and call
  `multiprocessing.freeze_support()` for PyInstaller compatibility — we
  do both in `__main__.py` and in the `run_media_organizer.py`
  PyInstaller launcher.
- **Pickling**: dataclasses are picklable; `Path` objects are picklable;
  the `Config` dataclass is picklable. Don't add un-picklable state
  (open file handles, DB connections) to it.

---

## Move log invariants

- **Append-only**. Mid-file rewrites are unsafe across crashes. The
  CONFIRM record pattern (`CONFIRM<TAB><line_idx><TAB>OK`) lets us
  acknowledge a previously written PLAN line by appending a small
  overlay record. `read_with_confirms()` reconstructs the OK flags by
  overlaying CONFIRM records onto the PLAN entries.
- **Line numbering is 1-based** within the data section (PLAN lines
  only, excluding `#` headers). Be careful when refactoring — the
  CONFIRM record's index must match the order PLAN lines were written.
- **Fsync on every commit**. We `flush()` then `os.fsync(fd)` after the
  header, after the plan block, and after every CONFIRM record. fsync
  may fail on network filesystems; we swallow that error.
- **TAGS column** is `|`-separated within one cell; the field
  separator is `<TAB>`. A tag that contains `|` would be corrupted —
  we strip it on write.

---

## Sidecar handling

- A sidecar file is paired to a primary by `(parent_dir, stem)`. So
  `IMG_0001.jpg` + `IMG_0001.json` in the same folder are paired, but
  `IMG_0001.json` from a different folder is not.
- When the primary gets a collision suffix `(2)`, sidecars take a
  parallel suffix derived from the **final** primary stem; if the
  sidecar's own destination still collides (rare), it gets its own
  independent numeric suffix so we never lose a file.
- Sidecars whose primary doesn't exist become standalone unknown
  items — better to move them somewhere visible than leave them stranded.
- The scanner skips the move-log file itself (so re-running with
  source==target doesn't try to "organize" the log).

---

## File layout when frozen

- PyInstaller `--onedir` produces `dist/<name>/<name>.exe` plus
  `dist/<name>/_internal/`. `sys.frozen` is True at runtime and
  `sys.executable` points at the .exe.
- `runtime.app_root()` returns either `Path(sys.executable).parent`
  (frozen) or the repo root (dev). We use this to locate `models/`
  and `third_party/` so the same code path works in both.
- The build script renames `dist/media-organizer/` → `dist/MediaOrganizer/`
  and copies `models/`, `third_party/`, `media-organizer.conf`, and
  `README.MD` next to the exe.
- The PyInstaller entry point is `src/run_media_organizer.py` (a
  thin launcher), NOT `src/media_organizer/__main__.py`. PyInstaller
  doesn't preserve the parent package when bundling a `__main__.py`,
  which breaks relative imports.

---

## Library notes / pitfalls

- **pillow_heif**: `register_heif_opener()` mutates PIL's global
  registry. The import is wrapped in try/except so a missing wheel
  degrades gracefully (HEIC files become "unknown").
- **librosa**: imports lazily — first call costs ~1.5 s (numba JIT).
- **librosa.beat.beat_track**: returns `tempo` as a 0-d ndarray on
  newer versions and as a float on older ones. We coerce with
  `float(np.atleast_1d(tempo)[0])`.
- **soundfile / audioread**: librosa's `load()` first tries soundfile
  (libsndfile), falls back to audioread (which calls ffmpeg). Bundled
  ffmpeg matters here, even for audio-only inputs.
- **EXIF orientation**: `ImageOps.exif_transpose(img)` MUST run before
  resize, or rotated phone photos classify as gibberish.
- **opencv-python**: bundles ffmpeg statically for VideoCapture; we use
  `cv2.VideoCapture` for video frame extraction so we don't need an
  external ffmpeg binary at runtime.
- **PyInstaller + onnxruntime**: needs `--collect-binaries onnxruntime`
  and `--collect-data onnxruntime` to ship the CUDA DLLs alongside the
  .exe. Without them, the runtime DLL probe always returns "not
  loadable" and we silently fall back to CPU.
- **PyInstaller + librosa/numba**: needs `--collect-submodules librosa`
  and `--collect-data librosa` to include the numba JIT cache.

---

## Build details

`scripts/build.ps1`:

1. Preflight-checks `models/` are present.
2. Calls **the venv's** `pyinstaller.exe`, not whatever `pyinstaller` is
   on PATH (system Python finds nothing because deps aren't there).
3. Entry point is `src/run_media_organizer.py`.
4. PyInstaller flags include hidden-import declarations for
   `onnxruntime`, `pytesseract`, `pillow_heif`, `cv2` and `--collect-*`
   flags for librosa, soundfile, pillow_heif, onnxruntime.
5. Renames `dist/media-organizer/` → `dist/MediaOrganizer/` and copies
   conf + models + third_party + README next to the exe.

Approximate sizes:

| Artifact | Size |
|---|---|
| `media-organizer.exe` | 33 MB |
| `_internal/` | 1.1 GB (mostly onnxruntime CUDA + torch_cpu DLLs) |
| `models/` | 24 MB |
| `third_party/tesseract/` | 239 MB |
| **Total** | ~1.4 GB |

To shrink to ~150 MB, swap `onnxruntime-gpu` → `onnxruntime` in
`requirements.txt` (loses GPU support).

---

## Performance ballparks (CPU-only, 8-core laptop)

- Image classify (full pipeline): ~150–400 ms per file.
- Video classify (5 frames, 1080p): ~1–3 s.
- Audio classify (3 min MP3): ~1–2 s.
- Throughput at 8-way parallelism: ~25 images/s, 4 videos/s, 6 audios/s.
- 564-file phone DCIM (~14 GB): ~90 minutes end-to-end on CPU.

GPU adds ~3–5× to the image path; OCR + ffmpeg + librosa stay on CPU.

---

## Useful URLs

- ImageNet 1000-class names (must match torchvision order!):
  `MobileNet_V3_Large_Weights.IMAGENET1K_V2.meta["categories"]`
- silero-vad: <https://github.com/snakers4/silero-vad>
- YuNet: <https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet>
- Tesseract Windows builds: <https://github.com/UB-Mannheim/tesseract/wiki>
- ffmpeg Windows builds (LGPL): <https://www.gyan.dev/ffmpeg/builds/>
- ONNX Runtime providers: <https://onnxruntime.ai/docs/execution-providers/>
- PyInstaller hooks for librosa / numba:
  <https://github.com/librosa/librosa/issues/1480>
- CUDA 12 download: <https://developer.nvidia.com/cuda-12-4-0-download-archive>
- cuDNN 9 download: <https://developer.nvidia.com/cudnn-downloads>

---

## Future ideas (NOT implemented; just notes)

- Replace the OCR-everywhere strategy with a tiny "text-vs-photo"
  classifier to skip Tesseract on most images (3–5× speed-up).
- Music genre via `MusicGenreClassifier` ONNX (50 MB, GTZAN-trained)
  for an optional `audio/music/<genre>/` second axis.
- Per-folder confidence overrides in the conf so e.g.
  `unknown-threshold-docs` can be tighter than the default.
- Batch ONNX inference inside a dedicated GPU worker (a queue + a
  single CUDA session) to amortize launch latency on huge runs.
- Write tags into image / video file metadata (IPTC keywords, XMP
  dc:subject) so they're searchable from Windows Explorer / Lightroom.
