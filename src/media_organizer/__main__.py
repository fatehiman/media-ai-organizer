"""
CLI entry point.

Usage:
    media-organizer.exe                  # full run, prompts before moving
    media-organizer.exe --config path    # alternate config file
    media-organizer.exe --yes            # skip confirmation prompt
    media-organizer.exe --dry-run        # plan only, never moves
    media-organizer.exe --apply          # apply (overrides dry-run = true in conf)
"""

from __future__ import annotations

import argparse
import multiprocessing
import sys
from pathlib import Path

from . import __version__
from . import config as config_module
from . import pipeline
from .movelog import MoveLogWriter, read_with_confirms
from .runtime import active_provider, app_root, describe_provider, make_session


# --- helpers -----------------------------------------------------------------

def _print_header(version: str) -> None:
    print(f"Media Organizer v{version}")
    print("=" * 60)


def _print_summary(summary: dict, total: int) -> None:
    print(f"\nClassification summary ({total} files):")
    if not summary:
        print("  (no files classified)")
        return
    width = max(len(k) for k in summary.keys())
    for label in sorted(summary.keys()):
        print(f"  {label:<{width}}  {summary[label]:>8,}")


def _confirm(prompt: str, default: bool = False) -> bool:
    suffix = " [Y/n]" if default else " [y/N]"
    try:
        ans = input(prompt + suffix + ": ").strip().lower()
    except EOFError:
        return default
    if not ans:
        return default
    return ans in ("y", "yes")


def _check_provider(use_gpu: str) -> None:
    """Print which ONNX execution provider the image model will use.  The
    test session is released right away (it would hold GPU memory)."""
    from .classifiers import clip
    try:
        sess = make_session(clip.vision_model_path(), use_gpu)
    except FileNotFoundError as e:
        print(str(e), file=sys.stderr)
        sys.exit(2)
    print(f"AI runs on    : {describe_provider(active_provider(sess))}")
    del sess


# --- main --------------------------------------------------------------------

def _resolve_config_path(arg: str | None) -> Path:
    if arg:
        return Path(arg).expanduser().resolve()
    # Look next to the executable / project root first.
    near_exe = app_root() / "media-organizer.conf"
    if near_exe.exists():
        return near_exe
    return Path.cwd() / "media-organizer.conf"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="media-organizer")
    parser.add_argument("--config", help="Path to media-organizer.conf")
    parser.add_argument("--yes", action="store_true",
                        help="Skip the confirmation prompt before moving")
    parser.add_argument("--dry-run", action="store_true",
                        help="Plan only; never move files (overrides config)")
    parser.add_argument("--apply", action="store_true",
                        help="Apply moves (overrides dry-run = true in config)")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    args = parser.parse_args(argv)

    _print_header(__version__)

    config_path = _resolve_config_path(args.config)
    print(f"Config        : {config_path}")
    try:
        cfg = config_module.load(config_path)
    except (FileNotFoundError, ValueError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    if args.apply:
        cfg.dry_run = False
    if args.dry_run:
        cfg.dry_run = True

    print(f"Source        : {cfg.source}")
    print(f"Target        : {cfg.target}")
    print(f"Dry-run       : {cfg.dry_run}")
    print(f"GPU policy    : {cfg.use_gpu}")
    print(f"CPU workers   : {cfg.cpu_workers if cfg.cpu_workers > 0 else 'auto'}")
    _check_provider(cfg.use_gpu)
    print("-" * 60)

    log_path = cfg.target / cfg.move_log

    # --- resume an in-progress run? ---
    existing = read_with_confirms(log_path)
    if existing is not None and not existing.completed and existing.plan:
        remaining = len(existing.plan) - existing.ok_count
        print(
            f"Found in-progress move log "
            f"({existing.ok_count}/{len(existing.plan)} done, {remaining} remaining)."
        )
        if cfg.dry_run:
            print("Dry-run mode is set; refusing to resume.  "
                  "Re-run with --apply to continue, or delete the log to restart.")
            return 0
        if not args.yes and not _confirm("Resume the previous run?", default=True):
            return 0
        plan = pipeline.Plan(entries=existing.plan, summary={})
        moved, failed = pipeline.apply_plan(cfg, plan, log_path, resume=True)
        print(f"\nResume complete: {moved} moved, {failed} failed.")
        return 0 if failed == 0 else 1

    # --- prior completed run sitting in the target? ---
    if existing is not None and existing.completed:
        print(f"A completed move log already exists at: {log_path}")
        if not args.yes and not _confirm(
            "Overwrite it and start a fresh run?", default=False
        ):
            return 0

    # --- fresh run ---
    print("Scanning source...")
    plan = pipeline.run_full(cfg)
    _print_summary(plan.summary, len(plan.entries))

    log_path.parent.mkdir(parents=True, exist_ok=True)
    # Persist plan to disk (no OK markers yet) so the user can review the file.
    writer = MoveLogWriter(log_path)
    writer.open_new(source=cfg.source, target=cfg.target, total=len(plan.entries))
    writer.write_plan_block(plan.entries)
    writer.close()
    print(f"\nFull plan written to: {log_path}")

    if cfg.dry_run:
        print("Dry-run mode: not moving any files.  Re-run with --apply to move.")
        return 0

    if not args.yes and cfg.require_confirmation:
        print("\nReview the plan above (and the log file).")
        if not _confirm("Apply moves now?", default=False):
            print("Cancelled by user.  No files were moved.")
            return 0

    moved, failed = pipeline.apply_plan(cfg, plan, log_path, resume=False)
    print(f"\nDone: {moved} moved, {failed} failed.")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    multiprocessing.freeze_support()      # required for Windows + PyInstaller
    sys.exit(main())
