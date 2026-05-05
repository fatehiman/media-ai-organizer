# Media Organizer

Offline tool that scans a source folder for images, videos and audio, detects
their content using local AI models, and **moves** them into a tidy folder
structure under a target directory. Fully offline, GPU-accelerated when a
CUDA GPU is present, falls back to CPU automatically.

```
target/
├── images/
│   ├── personal/      photos of people, nature, animals, buildings, museums
│   ├── docs/          papers, receipts, screenshots, menus, envelopes
│   ├── painting/      framed paintings, artwork
│   ├── nonsense/      object-only shots that aren't memories or documents
│   └── unknown/       anything below the confidence threshold
├── video/
│   ├── personal/      same buckets, voted across sampled frames
│   ├── nonsense/
│   └── unknown/
├── audio/
│   ├── talk/          phone calls, meetings, voice notes (speech-dominant)
│   ├── music/
│   │   ├── slow/      0–90 BPM
│   │   ├── medium/    90–130 BPM
│   │   ├── high/      130–300 BPM
│   │   └── unknown-rythm/   music with weak / mixed beat
│   └── unknown/       mixed speech + music
└── unknown/           files with unknown extensions or that failed to decode
```

## Highlights

- **Offline**. All inference runs locally with ONNX Runtime + bundled
  Tesseract. No network calls at runtime.
- **GPU + CPU saturation**. ONNX Runtime uses CUDA when `cudnn64_9.dll` and
  `cublasLt64_12.dll` are findable; otherwise transparent CPU fallback.
  Classification work is fanned out across `os.cpu_count()` worker
  processes either way.
- **Dry-run by default**. The first pass writes a full plan with explanatory
  tags to a move log file and prompts for confirmation before any file is
  moved.
- **Crash-safe resume**. Every successful move is fsync'd to the log; if the
  process is killed mid-run, the next invocation picks up where it left off.
- **YuNet face detection** (DNN, 232 KB). ImageNet has no "person" class —
  YuNet finds people in photos that ImageNet would otherwise mis-tag based
  on clothing or accessories. Far fewer false positives than Haar cascades.
- **Confidence-filtered OCR**. Tesseract's per-word confidence is checked;
  hallucinated "words" on textured surfaces (concrete, fabric, foliage)
  don't count toward the document signal.
- **Sidecars travel with their parent**. `.json`, `.xmp`, `.aae`, `.thm`,
  `.srt`, `.nfo`, `.txt`, `.vtt` are moved alongside their primary file so
  Google Takeout metadata, subtitles, and edit-list markers stay paired.
- **No renames**. Original filenames are preserved; collisions get a numeric
  suffix (`IMG_0001.jpg` → `IMG_0001 (2).jpg`).
- **Dynamic config**. Add or remove `image-<folder>` / `video-<folder>` /
  `keywords-<type>` lines in `media-organizer.conf` without touching code.

## Quick start

For users running the prebuilt portable `.exe`, see
[Run the portable .exe](#run-the-portable-exe) below.

For developers building from source, see
[tech-ref.md → Setup from a clean machine](tech-ref.md#setup-from-a-clean-machine).

### Run the portable .exe

1. Download or build `MediaOrganizer/` (a portable folder, ~1.4 GB).
2. Edit `MediaOrganizer\media-organizer.conf`:
   ```
   source = D:\Photos\Unsorted
   target = D:\Photos\Organized
   ```
3. Open a terminal in `MediaOrganizer\` (so the console doesn't auto-close):
   ```
   cd /d D:\path\to\MediaOrganizer
   .\media-organizer.exe              # dry-run, prompts before moving
   .\media-organizer.exe --apply      # apply moves (still prompts)
   .\media-organizer.exe --apply --yes  # apply, no prompts
   ```
4. The full plan is written to `<target>\media-organizer-move.log` before
   any move; open it to review classifications and tags.

If the run is killed mid-move (power loss, Ctrl-C, BSOD), launch the app
again — it will offer to resume from where it stopped.

## CLI flags

```
media-organizer.exe                 # full run (dry-run if config says so)
media-organizer.exe --apply         # force apply, ignoring dry-run = true
media-organizer.exe --dry-run       # plan only, never move
media-organizer.exe --yes           # skip the confirmation prompt
media-organizer.exe --config foo    # use an alternate config file
media-organizer.exe --version
```

## Configuration (`media-organizer.conf`)

Plain `key = value` text file with `#` comments. Lists are CSV.

The most important keys:

| Key                               | Meaning                                                                 |
|-----------------------------------|-------------------------------------------------------------------------|
| `source` / `target`               | Required. Absolute paths.                                               |
| `dry-run`                         | If `true`, plan only.                                                   |
| `unknown-threshold`               | Minimum classifier confidence (0–1). Below this, → `unknown/`.          |
| `collision-suffix`                | Format for the numeric suffix. Default `({n})` → ` (2)`, ` (3)`, …      |
| `move-sidecars` / `sidecar-extensions` | Whether and which sidecar files to move with their parent.         |
| `use-gpu`                         | `auto` / `yes` / `no`.                                                  |
| `cpu-workers`                     | `auto` (= all cores) or an integer.                                     |
| `ext-image` / `ext-video` / `ext-audio` | Which extensions count as each kind.                              |
| `image-<folder>`                  | Comma-separated list of content types that map into `images/<folder>/`. |
| `video-<folder>`                  | Same for video.                                                         |
| `rhythm-<band>`                   | `lo-hi` BPM range; defines `audio/music/<band>/`.                       |
| `keywords-<type>`                 | ImageNet class-name substrings that count toward this content type.     |
| `keywords-paper`                  | Document-class names (envelope, menu, book_jacket, …).                  |
| `face-detection`                  | `true` / `false`. Enables YuNet face-detection boost.                   |
| `face-boost`                      | Score (0–1) added to `face-target-content-type` when a face is found.   |
| `face-target-content-type`        | Content type that face hits feed into (default `human`).                |
| `audio-talk-min-speech-ratio`     | Threshold for routing into `audio/talk/`.                               |
| `audio-music-max-speech-ratio`    | Threshold for routing into `audio/music/`.                              |

You can add new image folders simply by adding a line — the app picks them up:

```
image-art = paintings, drawings, sculptures
keywords-paintings  = oil_painting
keywords-drawings   = sketch, comic_book
keywords-sculptures = statue
```

## How classification works

### Images

For each file the pipeline runs in parallel:

1. **MobileNetV3-Large** (ImageNet, 1000 classes) → softmax probabilities.
2. **YuNet** face detector → number of faces.
3. **Tesseract OCR** → list of words with per-word confidence.
4. **Blank-paper heuristic** → light/unsaturated/flat-surface score.

Then it aggregates:

- For each `keywords-<type>` list (e.g. `keywords-human`, `keywords-paper`),
  sum the ImageNet probabilities of classes whose names contain any
  configured keyword (substring, case-insensitive).
- `paper` content type combines: keyword score + (high-confidence-OCR
  word count / 30) + blank-paper heuristic. Highest wins.
- If a face is detected, `face-target-content-type` (default `human`) is
  forced to at least `face-boost` (default 0.95). The paper signal is
  zeroed out — a real face beats whatever text is in the background.
- For each `image-<folder>` (e.g. `image-personal = human, nature, …`),
  sum the scores of its content types.
- Best folder wins; if the best score is below `unknown-threshold`, the
  file goes to `images/unknown/`.

### Videos

`OpenCV` (cv2) extracts 5 frames evenly distributed through the video.
Each frame goes through the image pipeline (no OCR — too slow at 5×).
The folder with the highest single-frame score wins. Ties broken by total
score across all frames. Falls back to bundled `ffmpeg` if cv2 can't
open the container.

### Audio

`librosa` loads the file at 16 kHz mono. **Silero-VAD** ONNX gives a
per-window speech probability. If `>=` `audio-talk-min-speech-ratio` →
`talk`. Else if `<=` `audio-music-max-speech-ratio` → `librosa.beat.beat_track`
for tempo + beat strength. Weak beat → `music/unknown-rythm`. Otherwise
the BPM is mapped to a configured band. Anything in between is
`audio/unknown`.

## Move log format

Plain UTF-8 text, safe to inspect / edit between scan and apply phases.

```
# media-organizer move log v1
# started: 2026-05-05T14:23:11
# source: E:\Media\Unsorted
# target: E:\Media\Organized
# total: 12453
PLAN<TAB>SRC<TAB>DST<TAB>CATEGORY<TAB>CONFIDENCE<TAB>SIDECARS<TAB>TAGS
...
CONFIRM<TAB>1<TAB>OK
CONFIRM<TAB>2<TAB>OK
...
# completed: 2026-05-05T15:08:42
```

A `PLAN` line is the proposed move. A matching `CONFIRM<TAB><index><TAB>OK`
record means the move actually finished and was fsync'd. On resume, lines
without a matching `CONFIRM` are retried.

The **TAGS** column is a `|`-separated list of explanatory reasons:

| Tag | Meaning |
|---|---|
| `top=swing(0.14)` | Top ImageNet class and its probability |
| `ocr=42` | High-confidence OCR words found |
| `blank=0.81` | Blank-paper heuristic score |
| `paper-kw=0.45` | ImageNet class score for `keywords-paper` |
| `faces=5` | Face detector found 5 faces |
| `paper-suppressed-by-face` | Face hit zeroed the paper signal |
| `f0=personal(1.05)|f1=…` | Per-frame video votes |
| `below-threshold` | Best score < `unknown-threshold` |

## Limitations / known caveats

- Music **genre** is not detected; only tempo. Sad slow songs and slow
  acoustic ballads land in the same `slow/` bucket.
- Very short audio clips (`< 1 s`) and corrupt media files go to
  `unknown/`.
- The first run pays a ~5 s ONNX Runtime + librosa import startup. Files
  in the same run after that are fast.
- ImageNet has no clean "document" class, which is why we use OCR + the
  blank-paper heuristic + `keywords-paper`. False positives can still
  appear (a screenshot of a face will go to `personal`, not `docs`).
- `keywords-nonsense` is fuzzy by design; tune to taste.
