"""
Video conversion for the GUI: make phone videos smaller with ffmpeg,
without trimming and without visible problems (no stutter, no color
shift, no broken frames).

What is reduced (all optional, see VideoOptions):
    resolution   the short side is limited (e.g. 720p); never enlarged
    frame rate   only divided by a whole number (60 -> 30, 120 -> 30,
                 50 -> 25), so frames are dropped evenly (no judder)
    codec        HEVC (H.265) or H.264 with a CRF (constant quality) setting
    audio        AAC at 128 / 96 kbps, or kept as is

What is kept:
    every frame's own timestamp when the frame rate is not reduced
    (phones record variable frame rate; forcing a constant rate adds
    duplicate / dropped frames = visible stutter), HDR (HEVC 10-bit with
    the same color tags; H.264 gets a proper tone map to SDR), rotation
    (pixels are turned upright), audio sync, all container metadata
    (creation time, GPS location, Samsung / Apple keys), file times.

Decoding always runs on the CPU.  The GPU encoder (NVENC) is optional:
GPU *decoding* of phone videos is the usual cause of the "all blue" or
green output some converters make.

After encoding, verify() checks the output before it replaces anything:
    * duration matches, video (and audio, if the source had it) present
    * the whole file decodes without a single error
    * colors: frames at 20 / 50 / 80 % are compared with the source
      (catches a blue / green / broken picture)
    * metadata: every source tag arrived, GPS in the atom phones read
If a check fails, nothing is written and the source is kept.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import struct
import subprocess
import sys
import threading
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, Dict, List, Optional, Tuple

from .runtime import third_party_dir


DEFAULT_EXTS = [".mp4", ".mov", ".m4v", ".3gp", ".mkv", ".webm"]

# CRF / CQ per quality level.  Lower = better quality, bigger file.  The
# numbers give a similar look across encoders (checked with VMAF on phone
# videos: Balanced HEVC ~ 89, i.e. hard to tell from the source).
QUALITY_LEVELS = ["very_high", "high", "balanced", "small"]
QUALITY_LABELS = {"very_high": "Very high", "high": "High",
                  "balanced": "Balanced", "small": "Small file"}
_CRF = {
    ("cpu", "hevc"): {"very_high": 20, "high": 23, "balanced": 26, "small": 28},
    ("cpu", "h264"): {"very_high": 18, "high": 20, "balanced": 23, "small": 25},
    ("gpu", "hevc"): {"very_high": 21, "high": 24, "balanced": 27, "small": 29},
    ("gpu", "h264"): {"very_high": 19, "high": 22, "balanced": 25, "small": 27},
}
_SLOW_CRF_OFFSET = {"slow": 2}
_NVENC = {"hevc": "hevc_nvenc", "h264": "h264_nvenc"}
_X26X = {"hevc": "libx265", "h264": "libx264"}
_AUDIO_KBPS = {"aac128": 128, "aac96": 96}
# Audio codecs ffmpeg can decode and an MP4 can hold as they are.
_MP4_AUDIO = {"aac", "mp3", "alac", "ac3", "eac3", "opus", "flac"}
_DECODABLE_AUDIO = _MP4_AUDIO | {"vorbis", "amr_nb", "amr_wb", "pcm_s16le",
                                 "pcm_s16be", "pcm_s24le", "pcm_f32le", "pcm_u8"}
_HDR_TRC = {"smpte2084", "arib-std-b67"}
_STANDARD_FPS = [Fraction(24000, 1001), Fraction(24), Fraction(25),
                 Fraction(30000, 1001), Fraction(30), Fraction(48),
                 Fraction(50), Fraction(60000, 1001), Fraction(60),
                 Fraction(90), Fraction(100), Fraction(120), Fraction(240),
                 Fraction(480), Fraction(960)]
# Tags of the container itself, rewritten by ffmpeg: not compared.
_SKIP_TAGS = {"major_brand", "minor_version", "compatible_brands", "encoder"}
_APPLE_LOCATION = "com.apple.quicktime.location.iso6709"

# Color check: mean absolute difference (0..255) of 32x32 RGB thumbnails,
# and the largest shift of one channel's mean.  Normal encoding stays far
# below; a blue / green / broken frame is far above.
_MAX_PIXEL_DIFF = 28.0
_MAX_CHANNEL_SHIFT = 14.0
_THUMB = 32

_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
_BELOW_NORMAL = getattr(subprocess, "BELOW_NORMAL_PRIORITY_CLASS", 0)


class VideoError(Exception):
    """Conversion or verification failed; the source is untouched."""


class Cancelled(Exception):
    """Stopped by the user; the half-written output was removed."""


class Skipped(Exception):
    """The user chose not to convert this file (the target exists)."""


@dataclass
class VideoOptions:
    codec: str = "hevc"          # hevc | h264
    encoder: str = "cpu"         # cpu (libx265 / libx264) | gpu (NVENC)
    max_short: int = 720         # limit of the short side in px, 0 = keep
    max_fps: int = 30            # 0 = keep
    quality: str = "balanced"    # see QUALITY_LEVELS
    speed: str = "slow"          # CPU preset: fast | medium | slow
    audio: str = "aac128"        # copy | aac128 | aac96

    def crf(self) -> int:
        crf = _CRF[(self.encoder, self.codec)][self.quality]
        if self.encoder == "cpu" and self.codec == "hevc":
            # Same look at every speed: x265 "slow" at the same CRF gives a
            # bigger file with *higher* quality; CRF + 2 gives the medium
            # quality at 15-30 % less size (VMAF-checked).
            crf += _SLOW_CRF_OFFSET.get(self.speed, 0)
        return crf

    def describe(self) -> str:
        enc = "GPU NVENC" if self.encoder == "gpu" else f"CPU {self.speed}"
        size = f"{self.max_short}p" if self.max_short else "keep size"
        fps = f"max {self.max_fps} fps" if self.max_fps else "keep fps"
        return (f"{self.codec.upper()} {enc}, {QUALITY_LABELS[self.quality]} "
                f"(CRF {self.crf()}), {size}, {fps}, audio {self.audio}")


# --- ffmpeg location --------------------------------------------------------

@dataclass
class Tools:
    ffmpeg: Path
    ffprobe: Path
    ffplay: Optional[Path]

    def version(self) -> str:
        out = _run([str(self.ffmpeg), "-hide_banner", "-version"]).stdout
        return out.splitlines()[0] if out else "?"


def find_tools(configured: str = "") -> Optional[Tools]:
    """ffmpeg / ffprobe from (1) the `ffmpeg-dir` config value, (2) the
    portable third_party\\ffmpeg\\bin, (3) PATH."""
    dirs: List[Path] = []
    if configured:
        dirs.append(Path(configured))
    dirs.append(third_party_dir() / "ffmpeg" / "bin")
    for d in dirs:
        ff, fp = d / "ffmpeg.exe", d / "ffprobe.exe"
        if ff.is_file() and fp.is_file():
            play = d / "ffplay.exe"
            return Tools(ff, fp, play if play.is_file() else None)
    ff, fp = shutil.which("ffmpeg"), shutil.which("ffprobe")
    if ff and fp:
        play = shutil.which("ffplay")
        return Tools(Path(ff), Path(fp), Path(play) if play else None)
    return None


def _run(cmd: List[str], timeout: Optional[float] = None) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                          errors="replace", timeout=timeout,
                          creationflags=_NO_WINDOW | _BELOW_NORMAL)


_gpu_ok: Dict[str, bool] = {}


def gpu_available(tools: Tools, codec: str) -> bool:
    """Can this PC encode with NVENC?  (A tiny test encode, cached.)"""
    if codec not in _gpu_ok:
        r = _run([str(tools.ffmpeg), "-v", "error", "-f", "lavfi", "-i",
                  "color=c=gray:s=256x256:d=0.2", "-c:v", _NVENC[codec],
                  "-f", "null", "-"], timeout=60)
        _gpu_ok[codec] = r.returncode == 0
    return _gpu_ok[codec]


# --- probing ----------------------------------------------------------------

@dataclass
class VideoInfo:
    width: int                    # as displayed (rotation applied)
    height: int
    fps: Optional[Fraction]       # average frame rate
    duration: float
    pix_fmt: str
    color: Dict[str, str]         # primaries / trc / space / range, if known
    audio_index: Optional[int]    # stream index of the audio we keep
    audio_codec: str
    audio_kbps: float
    tags: Dict[str, str]          # container tags
    size: int
    vcodec: str = ""              # video codec name (h264, hevc, ...)

    @property
    def hdr(self) -> bool:
        return self.color.get("trc") in _HDR_TRC

    @property
    def ten_bit(self) -> bool:
        return any(b in self.pix_fmt for b in ("10", "12", "p016"))

    @property
    def location(self) -> Optional[str]:
        low = {k.lower(): v for k, v in self.tags.items()}
        return low.get("location") or low.get(_APPLE_LOCATION)


def _fraction(text: str) -> Optional[Fraction]:
    try:
        f = Fraction(text)
    except (ValueError, ZeroDivisionError, TypeError):
        return None
    return f if f > 0 else None


def _rotation(stream: dict) -> int:
    for sd in stream.get("side_data_list", []) or []:
        if "rotation" in sd:
            return int(sd["rotation"])
    try:
        return int(stream.get("tags", {}).get("rotate", 0))
    except ValueError:
        return 0


def _probe_json(tools: Tools, path: Path) -> dict:
    r = _run([str(tools.ffprobe), "-v", "error", "-print_format", "json",
              "-show_format", "-show_streams", str(path)], timeout=120)
    if r.returncode != 0:
        raise VideoError(f"ffprobe failed: {r.stderr.strip()[:300]}")
    return json.loads(r.stdout or "{}")


def probe(tools: Tools, path: Path) -> VideoInfo:
    data = _probe_json(tools, path)
    streams = data.get("streams", [])
    video = [s for s in streams if s.get("codec_type") == "video"
             and not s.get("disposition", {}).get("attached_pic")]
    if not video:
        raise VideoError("no video stream")
    v = video[0]
    w, h = int(v.get("width") or 0), int(v.get("height") or 0)
    if w <= 0 or h <= 0:
        raise VideoError("video size unknown")
    if abs(_rotation(v)) % 180 == 90:
        w, h = h, w
    fmt = data.get("format", {})
    duration = float(fmt.get("duration") or v.get("duration") or 0)
    if duration <= 0:
        raise VideoError("duration unknown")

    color = {}
    for key, name in (("color_primaries", "primaries"), ("color_transfer", "trc"),
                      ("color_space", "space"), ("color_range", "range")):
        val = v.get(key)
        if val and val not in ("unknown", "reserved", "unspecified"):
            color[name] = val

    audio = [s for s in streams if s.get("codec_type") == "audio"
             and s.get("codec_name") in _DECODABLE_AUDIO]
    audio.sort(key=lambda s: s.get("codec_name") != "aac")    # prefer AAC
    a = audio[0] if audio else None
    return VideoInfo(
        width=w, height=h,
        fps=_fraction(v.get("avg_frame_rate", "")) or _fraction(v.get("r_frame_rate", "")),
        duration=duration,
        pix_fmt=v.get("pix_fmt", ""),
        color=color,
        audio_index=int(a["index"]) if a else None,
        audio_codec=a.get("codec_name", "") if a else "",
        audio_kbps=float(a.get("bit_rate") or 0) / 1000 if a else 0.0,
        tags={str(k): str(val) for k, val in fmt.get("tags", {}).items()},
        size=path.stat().st_size,
        vcodec=v.get("codec_name", ""),
    )


# --- command ----------------------------------------------------------------

def target_size(info: VideoInfo, max_short: int) -> Tuple[int, int]:
    """Displayed size with the short side limited; even numbers (needed by
    the encoders); never enlarged."""
    w, h = info.width, info.height
    short = min(w, h)
    if max_short <= 0 or short <= max_short:
        return w - w % 2, h - h % 2
    s = max_short / short
    return max(2, round(w * s / 2) * 2), max(2, round(h * s / 2) * 2)


def _snap_fps(fps: Fraction) -> Fraction:
    for std in _STANDARD_FPS:
        if abs(fps - std) / std < 0.03:
            return std
    return fps


def target_fps(info: VideoInfo, max_fps: int) -> Optional[Fraction]:
    """New frame rate, or None to keep every frame and its timestamp.
    The rate is divided by a whole number so frames drop evenly."""
    if max_fps <= 0 or info.fps is None or info.fps <= max_fps * 1.05:
        return None
    src = _snap_fps(info.fps)
    k = math.ceil(src / max_fps - Fraction(1, 20))
    return src / max(k, 2)


@dataclass
class Plan:
    """What will be done to one file (for the log and for verify)."""
    size: Tuple[int, int]
    fps: Optional[Fraction]
    hdr_kept: bool
    tonemap: bool
    audio: str                    # "copy", "aac 128k", or "none"
    vfilter: str
    args: List[str] = field(default_factory=list)   # everything after -i

    def describe(self, info: VideoInfo) -> str:
        parts = [f"{info.width}x{info.height} -> {self.size[0]}x{self.size[1]}"]
        if self.fps is not None and info.fps is not None:
            parts.append(f"{float(info.fps):.0f} -> {float(self.fps):.0f} fps")
        if self.hdr_kept:
            parts.append("HDR kept")
        if self.tonemap:
            parts.append("HDR -> SDR")
        parts.append(f"audio {self.audio}")
        return ", ".join(parts)


def plan(info: VideoInfo, opts: VideoOptions) -> Plan:
    tw, th = target_size(info, opts.max_short)
    fps = target_fps(info, opts.max_fps)
    tonemap = info.hdr and opts.codec == "h264"
    ten_bit = opts.codec == "hevc" and info.ten_bit

    filters = []
    if fps is not None:
        filters.append(f"fps={fps.numerator}/{fps.denominator}")
    # Always through scale: it also turns full-range ("pc") video into the
    # limited range the output is tagged with (same size = no resize).
    filters.append(f"scale={tw}:{th}:flags=lanczos"
                   + ("" if tonemap else ":out_range=tv"))
    if tonemap:
        filters += ["zscale=t=linear:npl=100", "format=gbrpf32le", "zscale=p=bt709",
                    "tonemap=tonemap=hable:desat=0",
                    "zscale=t=bt709:m=bt709:r=tv"]
    if ten_bit:
        filters.append("format=p010le" if opts.encoder == "gpu" else "format=yuv420p10le")
    else:
        filters.append("format=yuv420p")
    vf = ",".join(filters)

    args = ["-map", "0:v:0"]
    if info.audio_index is not None:
        args += ["-map", f"0:{info.audio_index}"]
    # The source's brand tags would be copied as text keys (e.g. "qt"),
    # which only confuses tools: the real MP4 header is written anyway.
    args += ["-map_metadata", "0", "-metadata", "major_brand=",
             "-metadata", "minor_version=", "-metadata", "compatible_brands=",
             "-vf", vf]
    if fps is None:
        # Keep every frame with its own timestamp, in the source's time base
        # (the default time base rounds phone timestamps: two frames could
        # get the same time = a visible hitch).
        args += ["-fps_mode", "passthrough", "-enc_time_base:v", "demux"]

    crf = str(opts.crf())
    if opts.encoder == "gpu":
        args += ["-c:v", _NVENC[opts.codec], "-preset", "p6", "-tune", "hq",
                 "-rc", "vbr", "-cq", crf, "-b:v", "0", "-spatial-aq", "1",
                 "-temporal-aq", "1", "-rc-lookahead", "32"]
    else:
        args += ["-c:v", _X26X[opts.codec], "-preset", opts.speed, "-crf", crf]
        if opts.codec == "hevc":
            args += ["-x265-params", "log-level=error"]
    if opts.codec == "hevc":
        args += ["-tag:v", "hvc1"]            # needed by Apple players
    else:
        args += ["-profile:v", "high"]

    if tonemap:
        color = {"primaries": "bt709", "trc": "bt709", "space": "bt709", "range": "tv"}
    else:
        color = dict(info.color)
        color["range"] = "tv"                 # format=yuv420p* output is limited range
    for name, flag in (("primaries", "-color_primaries"), ("trc", "-color_trc"),
                       ("space", "-colorspace"), ("range", "-color_range")):
        if name in color:
            args += [flag, color[name]]

    audio = "none"
    if info.audio_index is not None:
        kbps = _AUDIO_KBPS.get(opts.audio)
        can_copy = info.audio_codec in _MP4_AUDIO
        if can_copy and (kbps is None or (info.audio_codec == "aac"
                                          and 0 < info.audio_kbps <= kbps * 1.1)):
            args += ["-c:a", "copy"]
            audio = "copy"
        else:
            kbps = kbps or 160
            args += ["-c:a", "aac", "-b:a", f"{kbps}k"]
            audio = f"aac {kbps}k"

    # moov at the end (no +faststart): add_location_atom() appends to it.
    args += ["-movflags", "+use_metadata_tags", "-max_muxing_queue_size", "4096",
             "-f", "mp4"]
    return Plan(size=(tw, th), fps=fps, hdr_kept=info.hdr and not tonemap,
                tonemap=tonemap, audio=audio, vfilter=vf, args=args)


# --- running ffmpeg ---------------------------------------------------------

def _encode(tools: Tools, src: Path, out: Path, p: Plan, duration: float,
            cancel: threading.Event, progress: Callable[[float], None],
            input_args: Optional[List[str]] = None) -> None:
    """Run ffmpeg, report progress 0..1, kill it on cancel."""
    cmd = ([str(tools.ffmpeg), "-hide_banner", "-nostdin", "-v", "error", "-y"]
           + (input_args or []) + ["-i", str(src)] + p.args
           + ["-progress", "pipe:1", "-nostats", str(out)])
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, encoding="utf-8", errors="replace",
                            creationflags=_NO_WINDOW | _BELOW_NORMAL)
    errors: List[str] = []
    reader = threading.Thread(target=lambda: errors.extend(proc.stderr), daemon=True)
    reader.start()
    killer = threading.Thread(
        target=lambda: (cancel.wait(), proc.poll() is None and proc.kill()), daemon=True)
    killer.start()
    for line in proc.stdout:
        if line.startswith("out_time_us="):
            try:
                progress(min(1.0, int(line[12:]) / 1e6 / max(duration, 0.001)))
            except ValueError:
                pass
    rc = proc.wait()
    reader.join(5)
    if cancel.is_set():
        out.unlink(missing_ok=True)
        raise Cancelled()
    if rc != 0:
        out.unlink(missing_ok=True)
        tail = "".join(errors).strip().splitlines()[-3:]
        raise VideoError("ffmpeg failed: " + (" | ".join(tail) or f"exit code {rc}"))


# --- GPS atom ---------------------------------------------------------------

def _box_header(f, offset: int, end: int) -> Optional[Tuple[int, bytes, int]]:
    """(box size, type, header length) at offset, or None."""
    if offset + 8 > end:
        return None
    f.seek(offset)
    size, kind = struct.unpack(">I4s", f.read(8))
    hdr = 8
    if size == 1:
        size = struct.unpack(">Q", f.read(8))[0]
        hdr = 16
    elif size == 0:
        size = end - offset
    if size < hdr or offset + size > end:
        return None
    return size, kind, hdr


def add_location_atom(path: Path, location: str) -> None:
    """Write the GPS position into moov/udta/©xyz, where Android and many
    tools look for it (ffmpeg's MP4 writer puts it only in the 'keys'
    list).  The moov box must be the last box in the file: then growing
    it moves no media data and no offsets change."""
    end = path.stat().st_size
    with path.open("r+b") as f:
        offset, moov = 0, None
        while offset < end:
            h = _box_header(f, offset, end)
            if h is None:
                raise VideoError("could not read the MP4 structure")
            if h[1] == b"moov":
                moov = (offset, h[0], h[2])
            offset += h[0]
        if moov is None or moov[0] + moov[1] != end or moov[2] != 8:
            raise VideoError("MP4 layout not supported for the GPS atom")
        f.seek(moov[0] + 8)
        body = f.read(moov[1] - 8)

        text = location.encode("utf-8")
        payload = struct.pack(">HH", len(text), 0x15C7) + text    # 0x15C7 = 'eng'
        xyz = struct.pack(">I", 8 + len(payload)) + b"\xa9xyz" + payload

        children, pos = [], 0
        while pos + 8 <= len(body):
            size, kind = struct.unpack(">I4s", body[pos:pos + 8])
            if size < 8 or pos + size > len(body):
                raise VideoError("could not read the MP4 structure")
            children.append([kind, body[pos:pos + size]])
            pos += size
        for child in children:
            if child[0] == b"udta":
                data = child[1] + xyz
                child[1] = struct.pack(">I", len(data)) + data[4:]
                break
        else:
            children.append([b"udta", struct.pack(">I", 8 + len(xyz)) + b"udta" + xyz])
        new_body = b"".join(c[1] for c in children)
        f.seek(moov[0])
        f.write(struct.pack(">I", 8 + len(new_body)) + b"moov" + new_body)
        f.truncate()
        f.flush()
        os.fsync(f.fileno())


def _has_xyz(path: Path) -> bool:
    """Is there a ©xyz atom in moov/udta?"""
    end = path.stat().st_size
    with path.open("rb") as f:
        offset = 0
        while offset < end:
            h = _box_header(f, offset, end)
            if h is None:
                return False
            if h[1] == b"moov":
                f.seek(offset + h[2])
                return b"\xa9xyz" in f.read(h[0] - h[2])
            offset += h[0]
    return False


# --- verification -----------------------------------------------------------

def _video_pts(tools: Tools, path: Path) -> List[str]:
    """Timestamps of all video packets (read from the index, fast)."""
    r = _run([str(tools.ffprobe), "-v", "error", "-select_streams", "v:0",
              "-show_entries", "packet=pts", "-of", "csv=p=0", str(path)], timeout=600)
    if r.returncode != 0:
        raise VideoError(f"could not read the frames of {path.name}")
    return [x.strip() for x in r.stdout.splitlines() if x.strip()]


def _thumbs(tools: Tools, path: Path, times: List[float], vfilter: str) -> List[bytes]:
    out = []
    for t in times:
        r = subprocess.run(
            [str(tools.ffmpeg), "-v", "error", "-nostdin", "-ss", f"{t:.3f}",
             "-i", str(path), "-map", "0:v:0", "-frames:v", "1",
             "-vf", f"{vfilter},scale={_THUMB}:{_THUMB}:flags=area,format=rgb24",
             "-f", "rawvideo", "-"],
            capture_output=True, timeout=120, creationflags=_NO_WINDOW | _BELOW_NORMAL)
        if r.returncode != 0 or len(r.stdout) != _THUMB * _THUMB * 3:
            raise VideoError(f"could not read a frame at {t:.1f} s of {path.name}")
        out.append(r.stdout)
    return out


def _compare(a: bytes, b: bytes) -> Tuple[float, float]:
    """(mean absolute difference, largest channel mean shift)."""
    n = len(a) // 3
    diff = sum(abs(x - y) for x, y in zip(a, b)) / len(a)
    shift = max(abs(sum(a[c::3]) - sum(b[c::3])) / n for c in range(3))
    return diff, shift


@dataclass
class Check:
    color_diff: float = 0.0
    color_shift: float = 0.0


def _check_decodes(tools: Tools, out: Path, cancel: threading.Event) -> None:
    if cancel.is_set():
        raise Cancelled()
    r = subprocess.run([str(tools.ffmpeg), "-v", "error", "-nostdin", "-i", str(out),
                        "-map", "0", "-fps_mode", "passthrough", "-enc_time_base", "demux",
                        "-f", "null", "-"],
                       capture_output=True, text=True, encoding="utf-8",
                       errors="replace", creationflags=_NO_WINDOW | _BELOW_NORMAL)
    if r.returncode != 0 or r.stderr.strip():
        raise VideoError("output has decode errors: "
                         + " | ".join(r.stderr.strip().splitlines()[:3]))


def verify(tools: Tools, src: Path, out: Path, info: VideoInfo, p: Plan,
           cancel: threading.Event) -> Check:
    try:
        o = probe(tools, out)
    except VideoError as e:
        raise VideoError(f"output unreadable: {e}")
    if (o.width, o.height) != p.size:
        raise VideoError(f"output size {o.width}x{o.height}, expected "
                         f"{p.size[0]}x{p.size[1]}")
    if abs(o.duration - info.duration) > max(1.0, info.duration * 0.01):
        raise VideoError(f"output duration {o.duration:.1f} s, source "
                         f"{info.duration:.1f} s")
    if info.audio_index is not None and o.audio_index is None:
        raise VideoError("audio missing in output")

    # Metadata: every source tag must arrive with the same value.
    lost = [k for k, v in info.tags.items()
            if k.lower() not in _SKIP_TAGS and o.tags.get(k) != v]
    if lost:
        raise VideoError("metadata would be lost: " + ", ".join(sorted(lost)[:5]))
    if info.location and not _has_xyz(out):
        raise VideoError("GPS location atom missing")

    # Frame timestamps: none twice (a hitch), and with the frame rate kept,
    # exactly the source's frames (none dropped or added).
    out_pts = _video_pts(tools, out)
    if len(set(out_pts)) != len(out_pts):
        raise VideoError("output has frames with the same timestamp")
    if p.fps is None:
        n_src = len(_video_pts(tools, src))
        if n_src != len(out_pts):
            raise VideoError(f"output has {len(out_pts)} frames, source {n_src}")

    # Whole file decodes without errors.
    _check_decodes(tools, out, cancel)

    # Colors: the source, decoded on its own, must look like the output.
    # So a wrong color range or matrix in our filters is caught too.  Only
    # a tone map (HDR -> SDR) changes colors on purpose: then the source
    # goes through the same filters (no encoding) first.
    if cancel.is_set():
        raise Cancelled()
    times = [info.duration * f for f in (0.2, 0.5, 0.8)]
    ref_filter = "null"
    if p.tonemap:
        ref_filter = ",".join(x for x in p.vfilter.split(",")
                              if not x.startswith("fps="))
    a = _thumbs(tools, src, times, ref_filter)
    b = _thumbs(tools, out, times, "null")
    check = Check()
    for x, y in zip(a, b):
        diff, shift = _compare(x, y)
        check.color_diff = max(check.color_diff, diff)
        check.color_shift = max(check.color_shift, shift)
    if check.color_diff > _MAX_PIXEL_DIFF or check.color_shift > _MAX_CHANNEL_SHIFT:
        raise VideoError(f"colors differ from the source (diff {check.color_diff:.1f}, "
                         f"shift {check.color_shift:.1f})")
    return check


# --- one file ---------------------------------------------------------------

@dataclass
class Result:
    out: Path
    src_bytes: int
    out_bytes: int
    kept_original: bool          # the converted file was not smaller
    summary: str


def _part_path(dst: Path) -> Path:
    return dst.with_name(dst.stem + ".part.mp4")


def _remux(tools: Tools, src: Path, out: Path, info: VideoInfo, audio: str,
           cancel: threading.Event, progress: Callable[[float], None]) -> None:
    """Same picture and sound (no re-encode) in an MP4 container.  audio:
    "copy" or "aac" (when the source audio does not fit in MP4)."""
    args = ["-map", "0:v:0"]
    if info.audio_index is not None:
        args += ["-map", f"0:{info.audio_index}"]
    args += ["-c:v", "copy"]
    if info.vcodec == "hevc":
        args += ["-tag:v", "hvc1"]               # what Apple players expect
    if info.audio_index is not None:
        args += ["-c:a", "copy"] if audio == "copy" else ["-c:a", "aac", "-b:a", "192k"]
    args += ["-map_metadata", "0", "-movflags", "+use_metadata_tags",
             "-max_muxing_queue_size", "4096"]
    _encode(tools, src, out, SimpleNamespace(args=args), info.duration, cancel,  # type: ignore[arg-type]
            progress)
    o = probe(tools, out)
    if abs(o.duration - info.duration) > max(1.0, info.duration * 0.01):
        raise VideoError(f"output duration {o.duration:.1f} s, source {info.duration:.1f} s")
    if info.audio_index is not None and o.audio_index is None:
        raise VideoError("audio missing in output")
    lost = [k for k, v in info.tags.items()
            if k.lower() not in _SKIP_TAGS and o.tags.get(k) != v]
    if lost:
        raise VideoError("metadata would be lost: " + ", ".join(sorted(lost)[:5]))
    if info.location:
        add_location_atom(out, info.location)
    _check_decodes(tools, out, cancel)


def _copy_original(src: Path, dst_dir: Path,
                   target_path: Callable[[Path, str, str], Path],
                   copy_times: Callable[[Path, Path], None]) -> Path:
    dst = target_path(dst_dir, src.stem, src.suffix.lower())
    tmp = dst.with_name(dst.name + ".part")
    try:
        shutil.copyfile(src, tmp)
        if tmp.stat().st_size != src.stat().st_size:
            raise VideoError("copy has a different size")
        os.replace(tmp, dst)
    finally:
        tmp.unlink(missing_ok=True)
    copy_times(src, dst)
    return dst


def _fallback(tools: Tools, src: Path, dst_dir: Path, info: Optional[VideoInfo],
              dst: Path, cancel: threading.Event, progress: Callable[[float], None],
              target_path: Callable[[Path, str, str], Path],
              copy_times: Callable[[Path, Path], None], reason: str) -> Result:
    """The normal conversion failed (or was not smaller): the file must
    still be in the destination.  Not MP4: the same streams are rewrapped
    into MP4 (same specs, no re-encode).  If that is not possible, or it
    already is MP4: the original is copied unchanged."""
    size = src.stat().st_size
    note = f"{reason} -> "
    if src.suffix.lower() != ".mp4" and info is not None:
        # PCM and other odd audio is not standard in MP4: go straight to AAC.
        ok_audio = info.audio_codec in ("aac", "mp3", "ac3", "eac3", "alac", "")
        for audio in (("copy", "aac") if ok_audio else ("aac",)):
            part = _part_path(dst)
            try:
                _remux(tools, src, part, info, audio, cancel, lambda x: progress(0.9 + 0.1 * x))
                os.replace(part, dst)
                copy_times(src, dst)
                return Result(dst, size, dst.stat().st_size, True,
                              note + "rewrapped to mp4, picture unchanged"
                              + (", audio to AAC" if audio == "aac"
                                 and info.audio_index is not None else ""))
            except (VideoError, OSError):
                continue
            finally:
                part.unlink(missing_ok=True)
    out = _copy_original(src, dst_dir, target_path, copy_times)
    return Result(out, size, size, True, note + "original copied unchanged")


def convert_file(tools: Tools, src: Path, dst_dir: Path, opts: VideoOptions,
                 cancel: threading.Event, progress: Callable[[float], None],
                 target_path: Callable[[Path, str, str], Path],
                 copy_times: Callable[[Path, Path], None]) -> Result:
    """Convert src to dst_dir/<stem>.mp4.  target_path(folder, stem, ext)
    picks the output path (a free name, or an existing file to overwrite);
    it may raise Skipped / Cancelled.  If the conversion fails or the result
    is not smaller than the source, the file is still put in the
    destination (see _fallback), so the source is never the only copy."""
    dst = target_path(dst_dir, src.stem, ".mp4")
    dst_dir.mkdir(parents=True, exist_ok=True)
    part = _part_path(dst)
    info: Optional[VideoInfo] = None
    try:
        info = probe(tools, src)
        p = plan(info, opts)
        _encode(tools, src, part, p, info.duration, cancel,
                lambda x: progress(0.9 * x))
        if info.location:
            add_location_atom(part, info.location)
        progress(0.9)
        check = verify(tools, src, part, info, p, cancel)
        progress(1.0)
        out_bytes = part.stat().st_size
        if out_bytes < info.size:
            os.replace(part, dst)
            copy_times(src, dst)
            return Result(dst, info.size, out_bytes, False,
                          p.describe(info) + f", color diff {check.color_diff:.1f}")
        reason = "converted file was not smaller"
    except (Cancelled, Skipped):
        raise
    except (VideoError, OSError, ValueError, subprocess.SubprocessError) as e:
        reason = "conversion failed (" + " ".join(str(e).split())[:200] + ")"
    finally:
        part.unlink(missing_ok=True)
    return _fallback(tools, src, dst_dir, info, dst, cancel, progress, target_path,
                     copy_times, reason)


def details(tools: Tools, path: Path) -> Dict[str, str]:
    """Facts shown in the overwrite dialog: date taken, resolution, length."""
    out: Dict[str, str] = {}
    try:
        data = _probe_json(tools, path)
    except (VideoError, ValueError, subprocess.SubprocessError):
        return out
    tags = {k.lower(): v for k, v in data.get("format", {}).get("tags", {}).items()}
    taken = tags.get("com.apple.quicktime.creationdate")
    if taken:
        out["taken"] = taken.replace("T", " ")[:19]
    elif tags.get("creation_time"):
        from datetime import datetime
        try:
            utc = datetime.fromisoformat(tags["creation_time"].replace("Z", "+00:00"))
            out["taken"] = utc.astimezone().strftime("%Y-%m-%d %H:%M:%S")
        except ValueError:
            out["taken"] = tags["creation_time"]
    for s in data.get("streams", []):
        if s.get("codec_type") == "video" and s.get("width"):
            w, h = int(s["width"]), int(s["height"])
            if abs(_rotation(s)) % 180 == 90:
                w, h = h, w
            out["resolution"] = f"{w} x {h}  ({s.get('codec_name', '?')})"
            break
    try:
        sec = float(data.get("format", {}).get("duration") or 0)
        out["duration"] = f"{int(sec // 60)}:{sec % 60:04.1f}"
    except ValueError:
        pass
    return out


def preview_clip(tools: Tools, src: Path, out: Path, opts: VideoOptions,
                 cancel: threading.Event, progress: Callable[[float], None],
                 seconds: float = 10.0) -> Tuple[VideoInfo, Plan, int]:
    """Encode `seconds` from the middle of src to out (for a quick look and
    a size estimate).  Returns (source info, plan, clip bytes)."""
    info = probe(tools, src)
    p = plan(info, opts)
    length = min(seconds, info.duration)
    start = max(0.0, info.duration / 2 - length / 2)
    _encode(tools, src, out, p, length, cancel, progress,
            input_args=["-ss", f"{start:.3f}", "-t", f"{length:.3f}"])
    return info, p, out.stat().st_size


def play(tools: Tools, path: Path, title: str) -> None:
    """Open a video: ffplay if present (plays every codec), else the
    default Windows player."""
    if tools.ffplay is not None:
        subprocess.Popen([str(tools.ffplay), "-hide_banner", "-loglevel", "error",
                          "-window_title", title, "-loop", "0", str(path)],
                         creationflags=_NO_WINDOW)
    elif sys.platform == "win32":
        os.startfile(path)  # type: ignore[attr-defined]
