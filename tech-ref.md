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

### 3. Download the models (one-time, online)

```powershell
.\.venv\Scripts\python.exe scripts\bootstrap_models.py
```

The CLIP files are too big for git (GitHub's limit is 100 MB per file),
so `models/clip/` is git-ignored and this step is required. The script
resumes broken downloads and skips files that are already complete.

| File | Size | Source |
|---|---|---|
| `clip/vision_model.onnx` | 335 MB | <https://huggingface.co/Xenova/clip-vit-base-patch32> (`onnx/vision_model.onnx`) |
| `clip/text_model.onnx` | 242 MB | same repo (`onnx/text_model.onnx`) |
| `clip/tokenizer.json` | 2 MB | same repo |
| `silero_vad.onnx` | 2.3 MB | <https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx> (committed) |
| `yunet_face.onnx` | 232 KB | <https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx> (committed) |

### 4. (Optional) Install bundled binaries (Tesseract)

Only needed when `ocr-boost > 0` in the conf (off by default).
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

Output: `dist\MediaOrganizer\` (~1.6 GB unzipped with Tesseract). Copy
that folder anywhere — fully portable.

---

## Models

### CLIP ViT-B/32 (image classifier)

- **Source**: OpenAI CLIP ViT-B/32, ONNX export by Xenova
  (<https://huggingface.co/Xenova/clip-vit-base-patch32>). License: MIT.
- **Why CLIP**: zero-shot. Categories are plain-English sentences in the
  conf (`prompts-<type>`), so "screenshot", "document", "object photo"
  work although ImageNet has no such classes. On `test/holdout`:
  CLIP 94.9 %, old MobileNetV3/ImageNet pipeline 30.6 %.
- **Models compared** (zero-shot only, same prompts, `test/sample`):

  | Model | center-crop | pad to square |
  |---|---|---|
  | CLIP ViT-B/32 | 83.8 % | **88.9 %** |
  | CLIP ViT-B/16 | 81.5 % | 85.2 % |
  | SigLIP base-16-224 | 80.1 % | 84.2 % |

  Pad-to-square wins because tall screenshots lose their status bar and
  app chrome in a center crop.
- **Variants**: fp32 is used. The int8 `vision_model_quantized.onnx`
  (84 MB) scored ~3 points lower in the full pipeline; the int8 text model
  lowered it ~4 points; `vision_model_fp16.onnx` fails to load in ORT 1.23
  (graph-optimizer bug: `InsertedPrecisionFreeCast_...`).
- **Vision input**: `pixel_values` NCHW 1×3×224×224, CLIP mean/std,
  bicubic resize after grey padding. Output `image_embeds` (512).
- **Text input**: `input_ids` (int64, 1×n, no padding), from the HF
  `tokenizers` library with `clip/tokenizer.json`. Output `text_embeds`.
- **Scoring**: cosine similarity × 100 (CLIP's logit scale) → softmax
  over all prompts → summed per content type.
- **Text embeddings** are computed once in the main process
  (`clip.compute_text_embeddings`) and passed to the workers through the
  pool initializer, so the text model is loaded only once per run.
- **Speed / RAM** (CPU, 1 thread): ~87 ms per image, +352 MB RSS per
  session. 4 threads: ~30 ms.

### YuNet (face detector)

- **Source**: <https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx>. License: MIT.
- **API**: `cv2.FaceDetectorYN_create` (bundled in `opencv-python` ≥ 4.5.4).
- **Input size**: arbitrary; we cap to 640px on the long edge for speed.
- **Score threshold**: `face-min-score` in the conf (default 0.88). On
  `test/sample`, real faces scored 0.88–0.95; cartoon faces on cookies and
  toys scored 0.5–0.87. A hit adds `face-boost` (0.5) to `person`.
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

### Tesseract OCR (optional)

- **Status**: off by default (`ocr-boost = 0`). On `test/sample`, an OCR
  document boost of 0.1–0.2 changed nothing and 0.3+ made results worse;
  OCR was also the slowest step and caused false "documents" on textured
  photos in the old pipeline.
- **Mode**: `--psm 6` ("single uniform block of text") — faster than
  PSM 3 (auto-segment) and adequate for the binary "is this a document?"
  question.
- **Word counting**: uses `pytesseract.image_to_data(...)` rather than
  `image_to_string()`. Per-word `conf` is checked; words below
  `_MIN_CONF = 60` (in `ocr.py`) are dropped. This kills hallucinated
  pseudo-words on textured surfaces (concrete, fabric, tree bark).
  Tokens still must match the regex `[A-Za-zÀ-ɏ]{2,}`.
- **Boost**: `ocr-boost * min(1.0, words / 30.0)` added to `ocr-content-type`.
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

For each image (`classifiers/image.py`):

1. Decode at reduced size (`Image.draft` for JPEG, then max 1024 px) and
   read metadata: camera `Make`, EXIF/XMP `UserComment`, file name,
   aspect ratio.
2. `UserComment` contains `Screenshot` (iOS) or the name contains
   `screenshot` → screenshot folder immediately (`screenshot-meta` tag).
3. CLIP → one probability per content type (sum = 1).
4. Camera `Make` present → screenshot score = 0. No camera `Make` and
   aspect ≥ 1.9 → `+ screenshot-tall-boost` (messenger re-sends strip EXIF
   but keep the screen shape).
5. YuNet face ≥ `face-min-score` → `+ face-boost` to `person`.
6. Optional OCR boost.
7. Folder score = sum of its content types; best wins; below
   `unknown-threshold` → `fallback-folder`.

Video frames use steps 3, 5, 6 (no metadata).

### How the numbers were chosen

`test/sample` (297 hand-labeled images from a real phone dump) was used
for every choice above; `test/holdout` (98 other images) was only used
for the final check. Simulation on `test/sample`:

| Pipeline | Accuracy |
|---|---|
| CLIP only | 91.2 % |
| + face boost 0.5 at score ≥ 0.88 | 93.6 % |
| + screenshot metadata / camera EXIF / tall boost | 94.6 % |
| + `unknown-threshold` 0.40 → other | 96.3 % |

Face score thresholds 0.85–0.90 gave the same result; boost 0.3 or 1.0
were worse than 0.5.

---

## Multiprocessing on Windows

- We use `concurrent.futures.ProcessPoolExecutor` with an `initializer`
  that stashes the parsed `Config` (plus the CLIP text embeddings and the
  ONNX thread count) in module globals of each worker.
- **Worker count**: `cpu-workers = auto` means
  `min(cpu_count, free_RAM / 700 MB)` (`workers._auto_workers`), because
  each worker holds its own CLIP session (~350 MB). Each worker then gets
  `cpu_count // workers` ONNX intra-op threads, so all cores stay busy.
- **GPU note**: with CUDA, every worker creates its own CUDA session of
  CLIP. On an 8 GB card, set `cpu-workers` to a small number (e.g. 4).
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
- The PyInstaller entry points are `src/run_media_organizer.py` (CLI)
  and `src/run_media_organizer_gui.py` (GUI) — thin launchers, NOT
  `src/media_organizer/__main__.py`. PyInstaller doesn't preserve the
  parent package when bundling a `__main__.py`, which breaks relative
  imports.
- Both exes live in the same folder and share `_internal/` (one
  `COLLECT` with two `EXE`s in `scripts/media-organizer.spec`), so the
  GUI adds only a few MB instead of a second ~800 MB copy.

---

## GUI (`gui.py`, `convert.py`, `trash.py`)

- **Toolkit**: Tkinter (ships with Python, ~10 MB in the bundle).
- **Threading**: all work (decode, convert, CLIP, file writes) runs in
  one worker thread; it talks to the UI only through a `queue.Queue`
  polled with `after(100)`. Tk variables must never be read from the
  worker — e.g. the ticked categories are copied into a `set` in the UI
  thread before the worker starts ("main thread is not in main loop"
  otherwise).
- **Low disk IO** (mass conversion of big USB / HDD trees):
  - Folders view (the start view, not saved) never lists files. At start
    it reads one directory (the source's top level). Each folder gets a
    placeholder child (`<iid>|dummy`; `|` can't be in a Windows file
    name) and is read on `<<TreeviewOpen>>`.
  - "Scan folders" walks the whole tree once (counts + sizes per folder;
    the tree is rebuilt fully loaded, open folders stay open).
  - Convert in Folders view walks only folders that are ticked or have a
    ticked descendant (`want_dir`), then converts.
  - Files view reads files when its tab is opened (non-recursive unless
    *Include subfolders*). The target list is read only on Refresh and
    after a Files-view run. After a Folders-view run the counts are only
    cleared, not re-read.
- **Walker** (`_walk`): explicit stack + `os.scandir`; on Windows
  `DirEntry.stat()` / `is_dir()` come from the directory listing, so no
  extra disk access per file. Folder keys are posix paths relative to the
  source (`""` = root). An `OSError` on one folder is recorded and the
  walk goes on. Background scans have a global generation number and a
  cancel `Event`; results of older scans are ignored. Convert / Preview
  are disabled while the source is scanning. 30,000 images in 300
  folders: Scan folders ~0.4 s, longest UI pause ~60 ms.
- **Tick states**: `states` holds only explicit states; a folder without
  one follows its nearest ancestor (default off). Ticking a folder sets
  its state and drops all descendant states, so unloaded subfolders
  follow too. States are saved with the source path they belong to
  (`folder_states_source`) and reset when the source changes.
- **Output path**: `<target>/<source-relative dir>/[<category>/]<stem>.<ext>`
  in both views. A target equal to or inside the source is refused
  (otherwise outputs would be listed and converted again).
- **Window placement**: `_center()` sizes the window (clipped to the
  screen) and centers it; used for the main and the preview window.
- **Don't name methods like Tk internals**: a method called `_options`
  on the `tk.Tk` subclass broke `columnconfigure` (Tk calls
  `self._options(cnf, kw)` internally).
- **Models**: CLIP is loaded lazily on the first categorize / preview with
  *Auto categorize* on. ONNX gets `cpu_count` intra-op threads, because
  the GUI uses one process.
- **Conversion** (`convert.py`):
  - target size: percent / width / height, aspect kept, `scale <= 1`
    (never enlarged);
  - JPEG decode uses `Image.draft` at the target size (much faster for
    big photos); the draft size is given in stored orientation, so width
    and height swap for EXIF orientations 5–8;
  - metadata is always copied (no option): EXIF incl. the Exif, GPS and
    Interop sub-IFDs, XMP (JPEG: APP1 via `xmp=`; PNG: `iTXt`
    `XML:com.adobe.xmp`, because Pillow's PNG writer ignores `xmp=`),
    ICC profile (iPhone HEIC is Display P3), JPEG comment, IPTC (JPEG
    APP13 payloads from `img.applist`, re-emitted via the JPEG `extra=`
    bytes), PNG text chunks; the Orientation tag is written as 1 because
    the pixels are rotated;
  - `img.getexif()` is a cached object that `exif_transpose` reads too:
    set Orientation 1 only while taking the bytes, then restore it, or
    the pixels are never rotated;
  - `_verify` re-opens the encoded bytes and compares the full set of
    (IFD, tag) pairs plus XMP / ICC presence; on a loss it raises
    `MetadataError` before anything is written (so the source is never
    recycled);
  - file times: atime / mtime with `os.utime`, creation time with
    `GetFileTime` / `SetFileTime` (ctypes);
  - transparent images: kept as RGBA in PNG, flattened on white for JPG;
  - written to `<name>.part` then `os.replace` + fsync, so a crash never
    leaves a half file under the real name.
- **Categorize after convert**: the classifier runs on the converted
  pixels (thumbnailed to 1024 px), but with `ImageMeta` read from the
  **source** file, because the screenshot / camera signals live in EXIF
  and XMP that a format change may drop.
- **Recycle Bin** (`trash.py`): `SHFileOperationW` with
  `FOF_ALLOWUNDO | FOF_NOCONFIRMATION | FOF_SILENT | FOF_NOERRORUI` via
  ctypes; `pFrom` must end with a double NUL. On a volume without a
  Recycle Bin Windows deletes permanently. The source is recycled only
  after its output was written successfully.
- **Settings**: `%APPDATA%\MediaOrganizer\gui.json`. Every Tk variable
  has a write trace that saves 0.5 s after the last change (and on
  close), so settings survive a crash or a killed process. The view is
  deliberately not saved.
- **Config**: the GUI loads `media-organizer.conf` with
  `require_paths=False` (it picks its own folders). Category checkboxes
  are the `image-<folder>` names; unticked / low-confidence →
  `fallback-folder`.

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
3. Builds from `scripts/media-organizer.spec` (committed; the
   `.gitignore` has an exception for it). The spec has two `Analysis`
   blocks (CLI, GUI) with the same hidden imports (`onnxruntime`,
   `pytesseract`, `pillow_heif`, `tokenizers`, `cv2`) and collected data
   for librosa, soundfile, pillow_heif, onnxruntime. The GUI exe is
   built with `console=False`.
4. Renames `dist/media-organizer/` → `dist/MediaOrganizer/` and copies
   conf + models + third_party + README next to the exe.

Approximate sizes:

| Artifact | Size |
|---|---|
| `media-organizer.exe` + `media-organizer-gui.exe` | 18 MB each |
| `_internal/` | ~800 MB (mostly onnxruntime CUDA DLLs; torch excluded in the spec) |
| `models/` | ~600 MB (CLIP 580 MB) |
| `third_party/tesseract/` | 239 MB (optional) |
| **Total** | ~1.6 GB |

To shrink to ~150 MB, swap `onnxruntime-gpu` → `onnxruntime` in
`requirements.txt` (loses GPU support).

---

## Performance ballparks (CPU-only)

Measured on a 20-thread laptop with ~2 GB free RAM (so `auto` = 2
workers × 10 threads):

- Image classify (decode + CLIP + YuNet): ~350 ms wall per image
  (old MobileNet + OCR pipeline: ~560 ms with 8 workers).
- Screenshots with metadata skip the model: decode time only.
- Video classify (5 frames): ~1–2 s.
- Audio classify (3 min MP3): ~1–2 s.

More free RAM → more workers → faster.

---

## Useful URLs

- CLIP ONNX models: <https://huggingface.co/Xenova/clip-vit-base-patch32>
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

- Batch CLIP inference (several images per `session.run`) for better
  CPU/GPU use.
- A small linear head trained on CLIP embeddings of labeled images
  (few-shot) if prompt tuning stops improving `test/holdout`.
- Music genre via `MusicGenreClassifier` ONNX (50 MB, GTZAN-trained)
  for an optional `audio/music/<genre>/` second axis.
- Per-folder confidence overrides in the conf so e.g.
  `unknown-threshold-docs` can be tighter than the default.
- Batch ONNX inference inside a dedicated GPU worker (a queue + a
  single CUDA session) to amortize launch latency on huge runs.
- Write tags into image / video file metadata (IPTC keywords, XMP
  dc:subject) so they're searchable from Windows Explorer / Lightroom.
