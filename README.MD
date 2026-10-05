# Media Organizer

Offline tool that scans a source folder for images, videos and audio, detects
their content using local AI models, and **moves** them into a tidy folder
structure under a target directory. Fully offline, GPU-accelerated when a
CUDA GPU is present, falls back to CPU automatically.

Two programs in one portable folder:

- `media-organizer.exe` — the CLI: sorts a whole folder tree (images,
  videos, audio) as shown below.
- `media-organizer-gui.exe` — a window to **convert / shrink images**
  (to JPG or PNG) and optionally sort them into category folders. See
  [GUI: convert & categorize](#gui-convert--categorize).

```
target/
├── images/
│   ├── personal/      photos with people (a real human face)
│   ├── documents/     paper and writing: forms, receipts, ID cards, books
│   ├── objects/       things: products, food, toys, crafts, clothes
│   ├── screenshots/   phone / computer screenshots
│   └── other/         landscapes, streets, animals, rooms, clipart, and
│                      anything below the confidence threshold
├── video/
│   ├── personal/      same buckets, voted across sampled frames
│   ├── documents/
│   ├── objects/
│   ├── screenshots/   screen recordings
│   └── other/
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

- **Offline**. All inference runs locally with ONNX Runtime. No network
  calls at runtime.
- **CLIP zero-shot**. Each category is described by short English
  sentences in the config (`prompts-<type>`). CLIP ViT-B/32 compares the
  image with every sentence, so categories can be tuned without training.
  On a hand-labeled hold-out set of real phone media: **94.9 %** correct
  (the previous MobileNet/ImageNet version: 30.6 %). See `test/`.
- **Screenshot metadata**. iOS writes `Screenshot` into the EXIF of every
  screenshot; Android names them `Screenshot_*`. Such files skip the model.
  A photo with camera EXIF (`Make`) is never called a screenshot.
- **GPU + CPU saturation**. ONNX Runtime uses CUDA when `cudnn64_9.dll` and
  `cublasLt64_12.dll` are findable; otherwise transparent CPU fallback.
  Classification work is fanned out across worker processes; with
  `cpu-workers = auto` their number is capped by free RAM (~700 MB each).
- **Dry-run by default**. The first pass writes a full plan with explanatory
  tags to a move log file and prompts for confirmation before any file is
  moved.
- **Crash-safe resume**. Every successful move is fsync'd to the log; if the
  process is killed mid-run, the next invocation picks up where it left off.
- **YuNet face detection** (DNN, 232 KB). A face with score ≥ 0.88 boosts
  `personal`. Real faces score ~0.90+; cartoon faces on toys and cookies
  mostly score lower, so they don't count.
- **Optional OCR**. Tesseract can add a document boost (`ocr-boost`); it is
  off by default because CLIP already recognizes documents and OCR is the
  slowest step.
- **Sidecars travel with their parent**. `.json`, `.xmp`, `.aae`, `.thm`,
  `.srt`, `.nfo`, `.txt`, `.vtt` are moved alongside their primary file so
  Google Takeout metadata, subtitles, and edit-list markers stay paired.
- **No renames**. Original filenames are preserved; collisions get a numeric
  suffix (`IMG_0001.jpg` → `IMG_0001 (2).jpg`).
- **Dynamic config**. Add or remove `image-<folder>` / `video-<folder>` /
  `prompts-<type>` lines in `media-organizer.conf` without touching code.

## Quick start

For users running the prebuilt portable `.exe`, see
[Run the portable .exe](#run-the-portable-exe) below.

For developers building from source, see
[tech-ref.md → Setup from a clean machine](tech-ref.md#setup-from-a-clean-machine).

### Run the portable .exe

1. Download or build `MediaOrganizer/` (a portable folder, ~1.6 GB).
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

## GUI: convert & categorize

Double-click `media-organizer-gui.exe`.

```
┌ Source ───────────┐ ┌ Options ─────────────┐ ┌ Target ───────────┐
│ folder [Browse]   │ │ Resize: keep / % /   │ │ folder [Browse]   │
│ IMG_0001.HEIC     │ │   width / height     │ │ personal\a.jpg    │
│ IMG_0002.PNG      │ │ Format: JPG / PNG    │ │ objects\b.jpg     │
│ ...               │ │   JPG quality 1-100  │ │ ...               │
│ [ ] subfolders    │ │ [ ] Move (del source)│ │ [Refresh]         │
│                   │ │ [ ] Auto categorize  │ │                   │
│                   │ │     [x] personal ... │ │                   │
│                   │ │ [Preview] [Convert]  │ │                   │
└───────────────────┘ └──────────────────────┘ └───────────────────┘
 Log: IMG_0001.HEIC -> personal\IMG_0001.jpg  2.4 MB -> 310 KB (13%)
```

- **Left pane**: the source folder and its images (JPG, PNG, HEIC, WEBP,
  … — the `ext-image` list of the conf). Tick *Include subfolders* to list
  images in subfolders too.
- **Right pane**: the target folder. Double-click a file to open it.
- **Resize**: *Keep size*, *Percent*, *Width* or *Height* in pixels. The
  aspect ratio is always kept, and images are **never enlarged**.
- **Target format**: JPG (with a quality slider) or PNG.
- **Move (del source)**: after each file is converted and saved, its
  source goes to the Recycle Bin (no prompt). This happens file by file,
  not at the end. On drives without a Recycle Bin (some USB sticks,
  network shares) Windows deletes the file permanently.
- **Auto categorize**: sorts the output into `target\<category>\`. Only
  ticked categories are used; an image whose category is not ticked goes
  to `target\other\`. Without *Auto categorize*, files go straight into
  the target folder.
- **Preview**: converts the selected image (double-click works too) and
  shows it at **1:1 size**, not shrunk to the screen, so you see the real
  result resolution and JPG quality. Scroll or drag to pan; `Esc` closes.
  The overlay shows `original -> converted` size, file size with percent,
  and the detected category.
- **Convert**: works on the selected images, or on all of them when none
  is selected. *Stop* stops after the current file.

Per file the order is: convert in memory → detect the category on the
small converted image (faster than on the big original) → save → recycle
the source. The source's metadata is read **before** converting, so
screenshot detection still works. Name collisions get a ` (2)` suffix.

**Metadata is always copied** — there is no option to turn it off. Every
piece the target format can hold is copied:

| Metadata | JPG target | PNG target |
|---|---|---|
| EXIF: camera, date taken, GPS location, lens, exposure, … | ✅ | ✅ |
| XMP (e.g. iOS screenshot marker, edits, ratings) | ✅ | ✅ |
| ICC color profile | ✅ | ✅ |
| JPEG comment / PNG `Comment` | ✅ | ✅ |
| IPTC (from JPEG sources) | ✅ | — (PNG has no standard place) |
| Other PNG text chunks | — | ✅ |
| File created / modified date | ✅ | ✅ |

Orientation: the pixels are turned upright during conversion, so the
EXIF Orientation tag is written as `1` (normal). The picture looks the
same as the original in every viewer; keeping the old value would make
viewers rotate it twice.

After converting, the output is read back and checked. If any EXIF tag,
the XMP or the color profile is missing, the file is **not** written, the
source is **not** deleted, and the log shows an error. Settings are remembered in
`%APPDATA%\MediaOrganizer\gui.json`.

## CLI flags

These apply to `media-organizer.exe` only.

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
| `unknown-threshold`               | Minimum classifier confidence (0–1). Below this, → `fallback-folder`.   |
| `fallback-folder`                 | Folder for low-confidence images/videos (default `other`).              |
| `collision-suffix`                | Format for the numeric suffix. Default `({n})` → ` (2)`, ` (3)`, …      |
| `move-sidecars` / `sidecar-extensions` | Whether and which sidecar files to move with their parent.         |
| `use-gpu`                         | `auto` / `yes` / `no`.                                                  |
| `cpu-workers`                     | `auto` (= cores, capped by free RAM) or an integer.                     |
| `ext-image` / `ext-video` / `ext-audio` | Which extensions count as each kind.                              |
| `image-<folder>`                  | Comma-separated list of content types that map into `images/<folder>/`. |
| `video-<folder>`                  | Same for video.                                                         |
| `rhythm-<band>`                   | `lo-hi` BPM range; defines `audio/music/<band>/`.                       |
| `prompts-<type>`                  | CLIP sentences that describe this content type.                         |
| `face-detection`                  | `true` / `false`. Enables YuNet face-detection boost.                   |
| `face-min-score`                  | YuNet score (0–1) a face needs to count (default 0.88).                 |
| `face-boost`                      | Score added to `face-target-content-type` when a face is found.         |
| `face-target-content-type`        | Content type that face hits feed into (default `person`).               |
| `screenshot-metadata`             | Use EXIF / file name to find screenshots and camera photos.             |
| `screenshot-tall-boost`           | Added to `screenshot` for phone-shaped images without camera EXIF.      |
| `ocr-boost` / `ocr-content-type`  | Optional Tesseract boost (0 = off).                                     |
| `audio-talk-min-speech-ratio`     | Threshold for routing into `audio/talk/`.                               |
| `audio-music-max-speech-ratio`    | Threshold for routing into `audio/music/`.                              |

You can add new image folders simply by adding a line — the app picks them up:

```
image-art = painting, drawing
prompts-painting = a painting, a photo of a painting in a frame
prompts-drawing  = a pencil drawing, a sketch
```

A new type takes probability from the others, so check the result with
`test/eval.py` (see [Testing](#testing)).

## How classification works

### Images

For each file:

1. **Screenshot metadata**: EXIF `UserComment = Screenshot` (iOS) or
   `screenshot` in the file name (Android) → `screenshots`, done.
2. **CLIP ViT-B/32** compares the image (padded to a square, 224 px) with
   every `prompts-<type>` sentence. A type's score is the summed softmax
   probability of its sentences; all types together sum to 1.
3. **Camera EXIF** (`Make`) present → `screenshot` score set to 0.
   **No camera EXIF and phone-shaped** (≥ 1.9 : 1, e.g. a screenshot
   re-sent by a messenger) → `screenshot-tall-boost` added.
4. **YuNet** finds a face with score ≥ `face-min-score` → `face-boost`
   added to `person`.
5. (Optional) **OCR** word count adds up to `ocr-boost` to `document`.
6. For each `image-<folder>`, sum the scores of its content types. Best
   folder wins; below `unknown-threshold` → `fallback-folder` (`other`).

### Videos

`OpenCV` (cv2) extracts 5 frames evenly distributed through the video.
Each frame goes through the image pipeline (CLIP + faces; no file
metadata and no OCR).
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
| `screenshot-meta` | EXIF / file name says screenshot (no model ran) |
| `clip=object(0.83)` | Top CLIP content type and its probability |
| `camera-exif` | Camera `Make` found; screenshot score zeroed |
| `tall-no-camera` | Phone-shaped without camera EXIF; screenshot boosted |
| `faces=5` | 5 faces with score ≥ `face-min-score` |
| `ocr=42` | OCR words found (only when `ocr-boost` > 0) |
| `f0=personal(1.05)|f1=…` | Per-frame video votes |
| `below-threshold` | Best score < `unknown-threshold` |

## Testing

`test/` holds a labeled test set taken from real phone media:

- `make_sample.py <media-root> <n> [name] [seed]` picks random images and
  draws numbered contact sheets (`test/sheets/<name>/`) for labeling.
- `<name>_labels.txt` has one `<id> <letter>` line per image
  (`P` personal, `D` documents, `O` objects, `S` screenshots, `X` other;
  `A/B` = both are fine).
- `eval.py --set sample|holdout --show-wrong` runs the real classifier
  and prints accuracy, a confusion matrix and every mistake.

`sample` (297 images) was used to choose the model, prompts and
thresholds; `holdout` (98 images) was not. Current results:

| Set | Old (MobileNet) | New (CLIP) |
|---|---|---|
| sample | 31.6 % | 96.3 % |
| holdout | 30.6 % | 94.9 % |

Re-run `eval.py --set holdout` after changing prompts or thresholds.

## Limitations / known caveats

- Music **genre** is not detected; only tempo. Sad slow songs and slow
  acoustic ballads land in the same `slow/` bucket.
- Very short audio clips (`< 1 s`) and corrupt media files go to
  `unknown/`.
- The first run pays a ~5 s ONNX Runtime + librosa import startup. Files
  in the same run after that are fast.
- People with a hidden or covered face (masks, hand over face, back of
  the head) may go to `other` when CLIP is unsure and no face is found.
- Downloaded clipart and logos sometimes land in `objects` instead of
  `other`.
- "Objects" vs "other" is a fuzzy line (flowers, furniture, rooms); tune
  the `prompts-object` / `prompts-scenery` lines to taste.
