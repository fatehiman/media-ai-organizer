"""
Move log: human-readable, append-only, crash-safe.

Format
------
    # media-organizer move log v1
    # started: 2026-05-05T14:23:11
    # source: E:\\Photos\\Unsorted
    # target: E:\\Photos\\Organized
    # total: 12453
    PLAN<TAB>SRC<TAB>DST<TAB>CATEGORY<TAB>CONFIDENCE OK
    PLAN<TAB>SRC<TAB>DST<TAB>CATEGORY<TAB>CONFIDENCE
    ...
    # completed: 2026-05-05T15:08:42

A line is committed (== move actually finished) iff it ends with the literal
suffix `<TAB>OK` (we use a real TAB so user content can never accidentally
collide).  Crash-safe model: write the PLAN line + flush + fsync BEFORE the
move; after a successful move, append `<TAB>OK\\n` + flush + fsync.

Resume rules
------------
  * No log file or completed-footer present  -> fresh run
    (a stale completed log triggers a confirmation prompt at the CLI level).
  * Header present, no completed-footer      -> resume.  Each line missing
    the OK suffix is re-evaluated:
        - if SRC still exists, retry the move
        - if DST exists and SRC is gone, the move actually succeeded but the
          OK marker was not flushed; mark OK now.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import IO, List, Optional

HEADER_VERSION = "v1"
SEP = "\t"
OK_MARKER = SEP + "OK"


@dataclass
class PlanEntry:
    src: Path
    dst: Path
    category: str          # e.g. 'images/personal' or 'audio/music/medium'
    confidence: float      # 0.0 .. 1.0+
    sidecars: List[Path]   # moved alongside src using same suffix policy
    tags: List[str] = field(default_factory=list)
                            # explanatory reasons: 'faces=1', 'top=swing(0.14)',
                            # 'ocr=42', 'blank=0.81', 'below-threshold', etc.

    def to_line(self) -> str:
        # Sidecars are encoded as a comma-separated list inside one column so
        # the log stays one-line-per-primary.  If empty the field is just '-'.
        sc = ",".join(str(p) for p in self.sidecars) if self.sidecars else "-"
        # Tags use a different separator so commas inside a tag (e.g. a
        # class name with a comma) wouldn't confuse the parser.  Pipes are
        # rare in tag values; if a tag accidentally contains one we just
        # strip it.
        tg = "|".join(t.replace("|", "/") for t in self.tags) if self.tags else "-"
        return SEP.join([
            "PLAN",
            str(self.src),
            str(self.dst),
            self.category,
            f"{self.confidence:.3f}",
            sc,
            tg,
        ])

    @classmethod
    def from_line(cls, line: str) -> "PlanEntry":
        parts = line.split(SEP)
        if parts and parts[-1] == "OK":
            parts = parts[:-1]
        # parts: PLAN, src, dst, category, confidence, sidecars[, tags]
        if len(parts) < 6 or parts[0] != "PLAN":
            raise ValueError(f"Malformed move-log line: {line!r}")
        sc_field = parts[5]
        sidecars = (
            [Path(p) for p in sc_field.split(",")]
            if sc_field and sc_field != "-"
            else []
        )
        tags: List[str] = []
        if len(parts) >= 7 and parts[6] and parts[6] != "-":
            tags = parts[6].split("|")
        return cls(
            src=Path(parts[1]),
            dst=Path(parts[2]),
            category=parts[3],
            confidence=float(parts[4]),
            sidecars=sidecars,
            tags=tags,
        )


@dataclass
class LogState:
    completed: bool
    plan: List[PlanEntry]            # all PLAN entries in order
    ok_count: int                    # how many of those carry the OK marker
    ok_flags: List[bool]             # parallel to .plan, True = already done


def _read_lines(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8") as f:
        return f.read().splitlines()


def read(path: Path) -> Optional[LogState]:
    """Return parsed state of an existing log file, or None if missing."""
    if not path.exists():
        return None

    lines = _read_lines(path)
    completed = any(line.startswith("# completed:") for line in lines)

    plan: List[PlanEntry] = []
    ok_flags: List[bool] = []
    for line in lines:
        if not line or line.startswith("#"):
            continue
        ok = line.endswith(OK_MARKER)
        try:
            entry = PlanEntry.from_line(line)
        except ValueError:
            # Skip malformed lines but don't crash; the user can edit by hand.
            continue
        plan.append(entry)
        ok_flags.append(ok)

    return LogState(
        completed=completed,
        plan=plan,
        ok_count=sum(ok_flags),
        ok_flags=ok_flags,
    )


class MoveLogWriter:
    """Append-only writer with explicit fsync on every commit point."""

    def __init__(self, path: Path):
        self.path = path
        self._fh: Optional[IO[str]] = None

    # --- lifecycle -------------------------------------------------------

    def open_new(self, *, source: Path, target: Path, total: int) -> None:
        """Truncate and write a fresh header."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("w", encoding="utf-8", newline="\n")
        self._fh.write(f"# media-organizer move log {HEADER_VERSION}\n")
        self._fh.write(f"# started: {datetime.now().isoformat(timespec='seconds')}\n")
        self._fh.write(f"# source: {source}\n")
        self._fh.write(f"# target: {target}\n")
        self._fh.write(f"# total: {total}\n")
        self._flush()

    def open_resume(self) -> None:
        """Append to an existing in-progress log."""
        self._fh = self.path.open("a", encoding="utf-8", newline="\n")
        self._fh.write(
            f"# resumed: {datetime.now().isoformat(timespec='seconds')}\n"
        )
        self._flush()

    def open_append(self) -> None:
        """Open an existing log for appending CONFIRM records (no header).

        Used when the plan was already written by a separate call to
        `open_new` + `write_plan_block` and we now need to record OK markers.
        """
        self._fh = self.path.open("a", encoding="utf-8", newline="\n")
        self._fh.write(
            f"# applying: {datetime.now().isoformat(timespec='seconds')}\n"
        )
        self._flush()

    def write_plan_block(self, entries: List[PlanEntry]) -> None:
        """Write all PLAN lines up-front (without OK markers).

        Done before the user is asked to confirm: lets them inspect the file.
        """
        assert self._fh is not None
        for e in entries:
            self._fh.write(e.to_line() + "\n")
        self._flush()

    def mark_ok(self, line_number: int) -> None:
        """Append the OK marker for a previously written PLAN line.

        We can't seek-and-write into the middle of the file safely on Windows
        with concurrent flushes, so instead each commit appends a special
        confirmation record:
            CONFIRM<TAB><line_number><TAB>OK
        and on read we walk the file and overlay these onto the plan.
        """
        assert self._fh is not None
        self._fh.write(f"CONFIRM{SEP}{line_number}{OK_MARKER}\n")
        self._flush()

    def write_completed(self) -> None:
        assert self._fh is not None
        self._fh.write(
            f"# completed: {datetime.now().isoformat(timespec='seconds')}\n"
        )
        self._flush()

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    # --- helpers ---------------------------------------------------------

    def _flush(self) -> None:
        assert self._fh is not None
        self._fh.flush()
        try:
            os.fsync(self._fh.fileno())
        except OSError:
            # fsync may fail on some non-local filesystems; flushing is the
            # best-effort fallback.
            pass


# --- resume parsing (overlays CONFIRM records onto the plan) -----------------

def read_with_confirms(path: Path) -> Optional[LogState]:
    """Read the log including CONFIRM<line_no> overlay records.

    The plan is written in one block (lines numbered 1..N within the data
    section); CONFIRM records flip a plan entry's ok flag without rewriting
    the original line.  This sidesteps mid-file rewrites.
    """
    if not path.exists():
        return None

    lines = _read_lines(path)
    completed = any(line.startswith("# completed:") for line in lines)

    plan: List[PlanEntry] = []
    ok_flags: List[bool] = []
    confirm_indices: List[int] = []

    for line in lines:
        if not line or line.startswith("#"):
            continue
        if line.startswith(f"PLAN{SEP}"):
            try:
                plan.append(PlanEntry.from_line(line))
                ok_flags.append(False)
            except ValueError:
                continue
        elif line.startswith(f"CONFIRM{SEP}") and line.endswith(OK_MARKER):
            # CONFIRM<TAB><idx><TAB>OK   (idx is 1-based)
            parts = line.split(SEP)
            if len(parts) >= 3:
                try:
                    confirm_indices.append(int(parts[1]))
                except ValueError:
                    pass

    for idx in confirm_indices:
        i = idx - 1
        if 0 <= i < len(ok_flags):
            ok_flags[i] = True

    return LogState(
        completed=completed,
        plan=plan,
        ok_count=sum(ok_flags),
        ok_flags=ok_flags,
    )
