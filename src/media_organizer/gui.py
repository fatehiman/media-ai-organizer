"""
Media Organizer GUI (Tkinter): convert / resize images, optionally
auto-categorize them with the same CLIP pipeline as the CLI, and optionally
send each source to the Recycle Bin.

Layout:   [ Source: Folders | Files tabs ] [ options + buttons ] [ Target file list ]
          [ log ........................................................................ ]

Disk IO is kept low, because the GUI is used for mass conversion of big
(often USB / HDD) photo trees:
  * Folders mode (the default at start) shows folders only.  At start only
    the top-level folder list of the source is read; a folder's subfolders
    are read when it is expanded.  "Scan folders" reads the whole tree once
    to show image counts and sizes.  Convert reads only the ticked folders.
  * Files mode reads files only when its tab is opened.
  * The target list is read only on "Refresh" (and after a Files-mode run).
All scans run in background threads (os.scandir: on Windows file sizes come
with the directory listing), so big trees don't freeze the window.

The source folder structure is mirrored in the target:
    <source>/2023/holidays/x.heic -> <target>/2023/holidays/x.jpg
    with auto categorize:         -> <target>/2023/holidays/<category>/x.jpg

Per file (one at a time, in a worker thread):
    convert in memory -> categorize the small converted image (using the
    source file's metadata) -> write the output -> send the source to the
    Recycle Bin (if "Move (del source)" is ticked).
"""

from __future__ import annotations

import io
import json
import os
import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Callable, Dict, List, Optional, Set, Tuple

from PIL import Image, ImageDraw, ImageTk

from . import __version__
from . import config as config_module
from . import convert as conv
from . import runtime
from .trash import send_to_recycle_bin


_PAD = 6
_WIN_W, _WIN_H = 1280, 760
_FOLDERS_TAB, _FILES_TAB = 0, 1
_DUMMY = "|dummy"             # placeholder child: "not loaded yet" (| is not
                               # allowed in Windows file names)

# One scanned image: (absolute path, size in bytes).
ScanEntry = Tuple[Path, int]
Progress = Callable[[int, int], None]     # (folders read, images found)


def _human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit in ("B", "KB") else f"{n:.2f} {unit}"
        n /= 1024
    return f"{n:.2f} GB"


def _settings_path() -> Path:
    base = Path(os.environ.get("APPDATA") or Path.home())
    return base / "MediaOrganizer" / "gui.json"


def _config_path() -> Path:
    near_exe = runtime.app_root() / "media-organizer.conf"
    return near_exe if near_exe.exists() else Path.cwd() / "media-organizer.conf"


def _center(win: tk.Misc, w: int, h: int) -> None:
    """Size `win` to w x h (clipped to the screen) and center it."""
    sw, sh = win.winfo_screenwidth(), win.winfo_screenheight()
    w, h = min(w, sw - 40), min(h, sh - 80)
    win.geometry(f"{w}x{h}+{max(0, (sw - w) // 2)}+{max(0, (sh - h) // 2 - 20)}")


def _parent_key(key: str) -> Optional[str]:
    """Parent of a relative folder key ("" = source root, which has none)."""
    if key == "":
        return None
    return key.rsplit("/", 1)[0] if "/" in key else ""


def _join(key: str, name: str) -> str:
    return f"{key}/{name}" if key else name


def _walk(
    root: Path,
    exts: Set[str],
    cancel: threading.Event,
    progress: Progress,
    *,
    recursive: bool = True,
    want_dir: Callable[[str], bool] = lambda key: True,
    on_dir: Callable[[str], None] = lambda key: None,
    on_image: Callable[[str, Path, int], None] = lambda key, p, size: None,
) -> List[str]:
    """Walk the folder tree under root with os.scandir.  Folder keys are
    paths relative to root ("" = root, "/" separated).  `want_dir(key)`
    decides whether a folder is read at all.  Returns read errors; a
    folder that can't be read is skipped, so a failing drive doesn't stop
    (or crash) the walk."""
    errors: List[str] = []
    stack = [""]
    n_dirs = n_images = 0
    while stack and not cancel.is_set():
        key = stack.pop()
        if not want_dir(key):
            continue
        on_dir(key)
        n_dirs += 1
        folder = root / key if key else root
        try:
            with os.scandir(folder) as it:
                for entry in it:
                    try:
                        if entry.is_dir(follow_symlinks=False):
                            if recursive:
                                stack.append(_join(key, entry.name))
                        elif os.path.splitext(entry.name)[1].lower() in exts:
                            on_image(key, Path(entry.path), entry.stat().st_size)
                            n_images += 1
                    except OSError as e:
                        errors.append(f"{entry.path}: {e}")
        except OSError as e:
            errors.append(f"{folder}: {e}")
        if n_dirs % 20 == 0:
            progress(n_dirs, n_images)
    progress(n_dirs, n_images)
    return errors


def _checkbox_images() -> Tuple[ImageTk.PhotoImage, ImageTk.PhotoImage]:
    """Small unchecked / checked box images for the folder tree."""
    def box(checked: bool) -> ImageTk.PhotoImage:
        im = Image.new("RGBA", (14, 14), (0, 0, 0, 0))
        d = ImageDraw.Draw(im)
        d.rectangle([0, 0, 13, 13], fill="white", outline="#555")
        if checked:
            d.line([(3, 7), (6, 10), (11, 3)], fill="#0a64c8", width=2)
        return ImageTk.PhotoImage(im)
    return box(False), box(True)


class App(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title(f"Media Organizer {__version__} - Convert & Categorize")
        self.minsize(1000, 600)
        _center(self, _WIN_W, _WIN_H)

        try:
            self.cfg = config_module.load(_config_path(), require_paths=False)
        except (FileNotFoundError, ValueError) as e:
            messagebox.showerror("Config error", str(e))
            raise SystemExit(2)
        self.categories: List[str] = list(self.cfg.image_folders)
        self.exts: Set[str] = set(self.cfg.ext_image)

        self.src_root: Optional[Path] = None     # source folder of the tree
        # Folders mode.  Tick state: explicit entries; a folder without one
        # follows its nearest ancestor that has one (default: not ticked).
        self.states: Dict[str, bool] = {}
        self.states_src = ""                     # source folder the states belong to
        self.folder_stats: Dict[str, Tuple[int, int]] = {}   # key -> (images, bytes) direct
        self.stats_valid = False
        self._totals: Dict[str, Tuple[int, int]] = {}         # key -> incl. subfolders
        self._scan_errors: List[str] = []
        # Files mode.
        self.files_root: Optional[Path] = None
        self.files_recursive = False
        self.src_scan: List[ScanEntry] = []
        self.src_files: List[Path] = []
        self.dst_files: List[Path] = []
        self.scans: Dict[str, Tuple[int, threading.Event]] = {}
        self._scan_gen = 0
        self._files_errors: List[str] = []

        self.msgs: "queue.Queue[tuple]" = queue.Queue()
        self.stop_event = threading.Event()
        self.worker: Optional[threading.Thread] = None
        self.models_ready = False
        self._save_job: Optional[str] = None
        self._poll_job: Optional[str] = None

        self._make_vars()
        self._load_settings()
        self.box_off, self.box_on = _checkbox_images()
        self._build()
        self._watch_vars()
        self._sync_states()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self._poll_job = self.after(100, self._poll)
        self.after(150, self._load_source)

    # --- settings ------------------------------------------------------------

    def _make_vars(self) -> None:
        self.v_src = tk.StringVar()
        self.v_dst = tk.StringVar()
        self.v_subfolders = tk.BooleanVar(value=False)
        self.v_resize = tk.StringVar(value="percent")
        self.v_percent = tk.IntVar(value=50)
        self.v_width = tk.IntVar(value=1920)
        self.v_height = tk.IntVar(value=1920)
        self.v_fmt = tk.StringVar(value="jpg")
        self.v_quality = tk.IntVar(value=85)
        self.v_move = tk.BooleanVar(value=False)
        self.v_auto = tk.BooleanVar(value=False)
        self.v_cats: Dict[str, tk.BooleanVar] = {
            c: tk.BooleanVar(value=True) for c in self.categories
        }
        self.v_status = tk.StringVar(value="Ready.")

    def _simple_vars(self) -> Dict[str, tk.Variable]:
        return {
            "src": self.v_src, "dst": self.v_dst, "subfolders": self.v_subfolders,
            "resize": self.v_resize, "percent": self.v_percent,
            "width": self.v_width, "height": self.v_height, "fmt": self.v_fmt,
            "quality": self.v_quality, "move": self.v_move, "auto": self.v_auto,
        }

    def _load_settings(self) -> None:
        try:
            data = json.loads(_settings_path().read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        for key, var in self._simple_vars().items():
            if key in data:
                try:
                    var.set(data[key])
                except tk.TclError:
                    pass
        for c, on in data.get("categories", {}).items():
            if c in self.v_cats:
                self.v_cats[c].set(bool(on))
        self.states = {k: bool(v) for k, v in data.get("folder_states", {}).items()}
        self.states_src = data.get("folder_states_source", "")
        # The view (Folders / Files) is not remembered: start in Folders mode.

    def _save_settings(self) -> None:
        self._save_job = None
        data = {}
        for key, var in self._simple_vars().items():
            try:
                data[key] = var.get()
            except tk.TclError:          # e.g. a half-typed spinbox value
                pass
        data["categories"] = {c: v.get() for c, v in self.v_cats.items()}
        data["folder_states"] = dict(sorted(self.states.items()))
        data["folder_states_source"] = self.states_src
        try:
            p = _settings_path()
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except OSError:
            pass

    def _schedule_save(self, *_args) -> None:
        """Save shortly after any change (so settings survive a crash or a
        killed process, not only a normal close)."""
        if self._save_job is not None:
            self.after_cancel(self._save_job)
        self._save_job = self.after(500, self._save_settings)

    def _watch_vars(self) -> None:
        for var in list(self._simple_vars().values()) + list(self.v_cats.values()):
            var.trace_add("write", self._schedule_save)

    @staticmethod
    def _int(var: tk.IntVar, default: int) -> int:
        try:
            return int(var.get())
        except (tk.TclError, ValueError):
            return default

    def _convert_options(self) -> conv.ConvertOptions:
        mode = self.v_resize.get()
        value = {
            "percent": self._int(self.v_percent, 100),
            "width": self._int(self.v_width, 1920),
            "height": self._int(self.v_height, 1920),
        }.get(mode, 100)
        return conv.ConvertOptions(
            resize_mode=mode,
            resize_value=max(1, value),
            fmt=self.v_fmt.get(),
            quality=min(100, max(1, self._int(self.v_quality, 85))),
        )

    # --- layout ----------------------------------------------------------------

    def _build(self) -> None:
        self.columnconfigure(0, weight=1)
        self.columnconfigure(2, weight=1)
        self.rowconfigure(0, weight=1)
        self._source_pane(0)
        self._center_pane(1)
        self._target_pane(2)

        logf = ttk.LabelFrame(self, text="Log")
        logf.grid(row=1, column=0, columnspan=3, sticky="nsew", padx=_PAD, pady=(0, _PAD))
        logf.columnconfigure(0, weight=1)
        self.log = tk.Text(logf, height=7, state=tk.DISABLED, wrap="none")
        sb = ttk.Scrollbar(logf, command=self.log.yview)
        self.log.configure(yscrollcommand=sb.set)
        self.log.grid(row=0, column=0, sticky="nsew")
        sb.grid(row=0, column=1, sticky="ns")

    def _path_row(self, frame: ttk.Frame, var: tk.StringVar, browse, refresh) -> None:
        entry = ttk.Entry(frame, textvariable=var)
        entry.grid(row=0, column=0, sticky="ew", padx=(0, 4))
        entry.bind("<Return>", lambda e: refresh())
        ttk.Button(frame, text="Browse...", command=browse).grid(row=0, column=1)

    def _scan_row(self, frame: ttk.Frame, row: int) -> Tuple[ttk.Label, ttk.Progressbar]:
        info = ttk.Label(frame, text="")
        info.grid(row=row, column=0, columnspan=2, sticky="w")
        bar = ttk.Progressbar(frame, mode="indeterminate")
        bar.grid(row=row + 1, column=0, columnspan=2, sticky="ew", pady=(0, 2))
        bar.grid_remove()
        return info, bar

    def _source_pane(self, col: int) -> None:
        frame = ttk.LabelFrame(self, text="Source")
        frame.grid(row=0, column=col, sticky="nsew", padx=_PAD, pady=_PAD)
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(3, weight=1)
        self._path_row(frame, self.v_src, self._browse_src, self._load_source)
        self.src_info, self.src_bar = self._scan_row(frame, 1)

        self.tabs = ttk.Notebook(frame)
        self.tabs.grid(row=3, column=0, columnspan=2, sticky="nsew")

        # Folders view (mass convert): folders only, no files.
        dt = ttk.Frame(self.tabs)
        dt.columnconfigure(0, weight=1)
        dt.rowconfigure(1, weight=1)
        bar = ttk.Frame(dt)
        bar.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(2, 2))
        ttk.Button(bar, text="Scan folders", width=13,
                   command=self._scan_folders).pack(side="left")
        ttk.Button(bar, text="Check all", width=10,
                   command=lambda: self._set_checked("", True)).pack(side="left", padx=4)
        ttk.Button(bar, text="Uncheck all", width=11,
                   command=self._uncheck_all).pack(side="left")
        self.tree = ttk.Treeview(dt, columns=("images", "size"), selectmode="browse")
        self.tree.heading("#0", text="Folder")
        self.tree.heading("images", text="Images")
        self.tree.heading("size", text="Size")
        self.tree.column("#0", width=230, stretch=True)
        self.tree.column("images", width=70, anchor="e", stretch=False)
        self.tree.column("size", width=80, anchor="e", stretch=False)
        tsb = ttk.Scrollbar(dt, command=self.tree.yview)
        self.tree.configure(yscrollcommand=tsb.set)
        self.tree.grid(row=1, column=0, sticky="nsew")
        tsb.grid(row=1, column=1, sticky="ns")
        ttk.Label(dt, text="Tick folders to convert (subfolders included). "
                           "Counts include subfolders.",
                  foreground="gray").grid(row=2, column=0, sticky="w")
        self.tree.bind("<Button-1>", self._on_tree_click)
        self.tree.bind("<space>", self._on_tree_space)
        self.tree.bind("<<TreeviewOpen>>", self._on_tree_open)
        self.tabs.add(dt, text="Folders")

        # Files view
        ft = ttk.Frame(self.tabs)
        ft.columnconfigure(0, weight=1)
        ft.rowconfigure(0, weight=1)
        self.src_list = tk.Listbox(ft, activestyle="none", exportselection=False,
                                   selectmode=tk.EXTENDED)
        sb = ttk.Scrollbar(ft, command=self.src_list.yview)
        self.src_list.configure(yscrollcommand=sb.set)
        self.src_list.grid(row=0, column=0, sticky="nsew")
        sb.grid(row=0, column=1, sticky="ns")
        ttk.Checkbutton(ft, text="Include subfolders", variable=self.v_subfolders,
                        command=self._scan_files).grid(row=1, column=0, sticky="w")
        ttk.Label(ft, text="Convert = selected files, or all listed if none selected",
                  foreground="gray").grid(row=2, column=0, sticky="w")
        self.src_list.bind("<Double-Button-1>", lambda e: self._preview())
        self.src_list.bind("<<ListboxSelect>>", lambda e: self._update_info())
        self.tabs.add(ft, text="Files")

        self.tabs.select(_FOLDERS_TAB)
        self.tabs.bind("<<NotebookTabChanged>>", self._on_tab_changed)

    def _target_pane(self, col: int) -> None:
        frame = ttk.LabelFrame(self, text="Target  (source folder structure is kept)")
        frame.grid(row=0, column=col, sticky="nsew", padx=_PAD, pady=_PAD)
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(3, weight=1)
        self._path_row(frame, self.v_dst, self._browse_dst, self._refresh_dst)
        self.dst_info, self.dst_bar = self._scan_row(frame, 1)
        self.dst_info.configure(text="Press Refresh to list the target folder.")
        lf = ttk.Frame(frame)
        lf.grid(row=3, column=0, columnspan=2, sticky="nsew")
        lf.columnconfigure(0, weight=1)
        lf.rowconfigure(0, weight=1)
        self.dst_list = tk.Listbox(lf, activestyle="none", exportselection=False)
        sb = ttk.Scrollbar(lf, command=self.dst_list.yview)
        self.dst_list.configure(yscrollcommand=sb.set)
        self.dst_list.grid(row=0, column=0, sticky="nsew")
        sb.grid(row=0, column=1, sticky="ns")
        ttk.Button(frame, text="Refresh", command=self._refresh_dst).grid(
            row=4, column=0, sticky="w")
        self.dst_list.bind("<Double-Button-1>", self._open_dst)

    def _center_pane(self, col: int) -> None:
        c = ttk.Frame(self)
        c.grid(row=0, column=col, sticky="ns", pady=_PAD)

        rs = ttk.LabelFrame(c, text="Resize (aspect ratio kept, never enlarged)")
        rs.pack(fill="x", pady=(0, _PAD))
        ttk.Radiobutton(rs, text="Keep size", value="none", variable=self.v_resize,
                        command=self._sync_states).grid(row=0, column=0, sticky="w")
        self.spins = {}
        for row, (mode, label, var, hi, unit) in enumerate([
            ("percent", "Percent", self.v_percent, 100, "%"),
            ("width", "Width", self.v_width, 20000, "px"),
            ("height", "Height", self.v_height, 20000, "px"),
        ], start=1):
            ttk.Radiobutton(rs, text=label, value=mode, variable=self.v_resize,
                            command=self._sync_states).grid(row=row, column=0, sticky="w")
            sp = ttk.Spinbox(rs, from_=1, to=hi, textvariable=var, width=7)
            sp.grid(row=row, column=1, padx=4, pady=1)
            ttk.Label(rs, text=unit).grid(row=row, column=2, sticky="w")
            self.spins[mode] = sp

        fm = ttk.LabelFrame(c, text="Target format")
        fm.pack(fill="x", pady=(0, _PAD))
        ttk.Radiobutton(fm, text="JPG", value="jpg", variable=self.v_fmt,
                        command=self._sync_states).grid(row=0, column=0, sticky="w")
        ttk.Radiobutton(fm, text="PNG", value="png", variable=self.v_fmt,
                        command=self._sync_states).grid(row=0, column=1, sticky="w")
        ttk.Label(fm, text="JPG quality").grid(row=1, column=0, sticky="w")
        self.q_label = ttk.Label(fm, width=4)
        self.q_label.grid(row=1, column=1, sticky="e")
        self.q_scale = ttk.Scale(fm, from_=1, to=100, orient="horizontal",
                                 command=self._on_quality)
        self.q_scale.set(self.v_quality.get())
        self.q_scale.grid(row=2, column=0, columnspan=2, sticky="ew", padx=2)
        fm.columnconfigure(1, weight=1)

        ttk.Checkbutton(c, text="Move (del source)", variable=self.v_move).pack(
            anchor="w", pady=(0, _PAD))

        ac = ttk.LabelFrame(c, text="Categories")
        ac.pack(fill="x", pady=(0, _PAD))
        ttk.Checkbutton(ac, text="Auto categorize", variable=self.v_auto,
                        command=self._sync_states).pack(anchor="w")
        self.cat_checks = []
        for cat in self.categories:
            cb = ttk.Checkbutton(ac, text=cat, variable=self.v_cats[cat])
            cb.pack(anchor="w", padx=(18, 0))
            self.cat_checks.append(cb)
        ttk.Label(ac, text=f"unticked -> {self.cfg.fallback_folder}",
                  foreground="gray").pack(anchor="w", padx=(18, 0))

        self.btn_preview = ttk.Button(c, text="Preview", command=self._preview)
        self.btn_preview.pack(fill="x", pady=2)
        self.btn_convert = ttk.Button(c, text="Convert", command=self._convert)
        self.btn_convert.pack(fill="x", pady=2)
        self.btn_stop = ttk.Button(c, text="Stop", command=self.stop_event.set)
        self.btn_stop.pack(fill="x", pady=(2, _PAD))

        self.progress = ttk.Progressbar(c, mode="determinate")
        self.progress.pack(fill="x")
        ttk.Label(c, textvariable=self.v_status, wraplength=230).pack(fill="x")

    def _on_quality(self, value: str) -> None:
        self.v_quality.set(int(float(value)))
        self.q_label.configure(text=str(self.v_quality.get()))

    def _tab(self) -> int:
        try:
            return self.tabs.index(self.tabs.select())
        except (AttributeError, tk.TclError):
            return _FOLDERS_TAB

    def _sync_states(self) -> None:
        for mode, sp in self.spins.items():
            sp.configure(state="normal" if self.v_resize.get() == mode else "disabled")
        self.q_scale.configure(state="normal" if self.v_fmt.get() == "jpg" else "disabled")
        self.q_label.configure(text=str(self.v_quality.get()))
        for cb in self.cat_checks:
            cb.configure(state="normal" if self.v_auto.get() else "disabled")
        busy = self.worker is not None and self.worker.is_alive()
        scanning = "src" in self.scans
        state = "disabled" if busy or scanning else "normal"
        self.btn_convert.configure(state=state)
        self.btn_preview.configure(state=state)
        self.btn_stop.configure(state="normal" if busy else "disabled")

    # --- source / target selection ----------------------------------------------

    def _browse_src(self) -> None:
        d = filedialog.askdirectory(initialdir=self.v_src.get() or None)
        if d:
            self.v_src.set(os.path.normpath(d))
            self._load_source()

    def _browse_dst(self) -> None:
        d = filedialog.askdirectory(initialdir=self.v_dst.get() or None)
        if d:
            self.v_dst.set(os.path.normpath(d))
            self._refresh_dst()

    def _source_path(self) -> Optional[Path]:
        text = self.v_src.get().strip()
        # An empty entry means nothing (Path("") would be the cwd).
        return Path(text) if text else None

    def _load_source(self) -> None:
        """New source folder: rebuild the folder tree (top level only, no
        file scan).  Files mode re-reads its list when shown."""
        self._cancel_scan("src")
        root = self._source_path()
        if root is not None and os.path.normcase(str(root)) != os.path.normcase(
                self.states_src):
            self.states = {}            # tick states belong to one source
            self.states_src = str(root)
            self._schedule_save()
        self.src_root = root
        self.files_root = None
        self.folder_stats, self.stats_valid = {}, False
        self.tree.delete(*self.tree.get_children())
        self.src_list.delete(0, tk.END)
        self.src_files, self.src_scan = [], []
        if root is None or not root.is_dir():
            self.src_root = None
            self.src_info.configure(text="Folder not found." if root else "")
            self._sync_states()
            return
        self.tree.insert("", "end", iid=self._iid(""), text=" " + (root.name or str(root)),
                         image=self._box(""), values=("", ""))
        self.tree.insert(self._iid(""), "end", iid=self._iid("") + _DUMMY)
        self.tree.item(self._iid(""), open=True)
        self._load_children("")
        if self._tab() == _FILES_TAB:
            self._scan_files()
        self._update_info()
        self._sync_states()

    def _refresh_dst(self) -> None:
        text = self.v_dst.get().strip()
        root = Path(text) if text else None
        if root is None or not root.is_dir():
            self._cancel_scan("dst")
            self.dst_files = []
            self.dst_list.delete(0, tk.END)
            self.dst_info.configure(text="Folder not found." if root else "")
            return
        entries: List[ScanEntry] = []
        self._start_scan(
            "dst",
            lambda cancel, prog: _walk(root, self.exts, cancel, prog,
                                       on_image=lambda k, p, s: entries.append((p, s))),
            lambda errors: self._dst_done(root, entries, errors))

    def _dst_done(self, root: Path, entries: List[ScanEntry], errors: List[str]) -> None:
        entries.sort(key=lambda e: str(e[0]).lower())
        self.dst_files = [p for p, _s in entries]
        self.dst_list.delete(0, tk.END)
        if entries:
            self.dst_list.insert(tk.END, *[
                f"{p.relative_to(root)}   ({_human(s)})" for p, s in entries])
        self.dst_info.configure(text=self._summary(len(entries), sum(s for _p, s in entries),
                                                   errors))

    # --- background scans --------------------------------------------------------

    def _cancel_scan(self, pane: str) -> None:
        old = self.scans.pop(pane, None)
        if old:
            old[1].set()
        bar = self.src_bar if pane == "src" else self.dst_bar
        bar.stop()
        bar.grid_remove()
        self._sync_states()

    def _start_scan(self, pane: str, work: Callable, done: Callable) -> None:
        """Run work(cancel, progress) -> errors in a thread, then done(errors)
        in the UI thread.  A newer scan of the same pane cancels the older."""
        self._cancel_scan(pane)
        self._scan_gen += 1
        gen = self._scan_gen
        cancel = threading.Event()
        self.scans[pane] = (gen, cancel)
        info, bar = (self.src_info, self.src_bar) if pane == "src" else (
            self.dst_info, self.dst_bar)
        info.configure(text="Scanning...")
        bar.grid()
        bar.start(12)
        self._sync_states()

        def progress(n_dirs: int, n_images: int) -> None:
            self.msgs.put(("scan_progress", pane, gen,
                           f"Scanning... {n_images:,} images in {n_dirs:,} folders"))

        def run() -> None:
            errors = work(cancel, progress)
            if not cancel.is_set():
                self.msgs.put(("scan_done", pane, gen, done, errors))

        threading.Thread(target=run, daemon=True).start()

    def _scan_finished(self, pane: str, done: Callable, errors: List[str]) -> None:
        self._cancel_scan(pane)          # stops the bar, frees the pane
        for e in errors[:20]:
            self._log(f"ERROR reading {e}")
        if len(errors) > 20:
            self._log(f"... and {len(errors) - 20} more read errors")
        done(errors)
        self._sync_states()

    @staticmethod
    def _summary(n: int, size: int, errors: List[str]) -> str:
        text = f"{n:,} images, {_human(size)}"
        if errors:
            text += f"  ({len(errors)} read errors, see log)"
        return text

    # --- folders view ------------------------------------------------------------

    @staticmethod
    def _iid(key: str) -> str:
        return "/" + key            # Treeview ids can't be "" (that is its root)

    def _included(self, key: str) -> bool:
        """Effective tick state: own explicit state, else the nearest
        ancestor's, else not ticked."""
        k: Optional[str] = key
        while k is not None:
            if k in self.states:
                return self.states[k]
            k = _parent_key(k)
        return False

    def _box(self, key: str) -> ImageTk.PhotoImage:
        return self.box_on if self._included(key) else self.box_off

    def _list_subfolders(self, key: str) -> List[str]:
        """Names of the subfolders of one folder (a single directory read)."""
        assert self.src_root is not None
        folder = self.src_root / key if key else self.src_root
        try:
            with os.scandir(folder) as it:
                names = [e.name for e in it if e.is_dir(follow_symlinks=False)]
        except OSError as e:
            self._log(f"ERROR reading {folder}: {e}")
            return []
        return sorted(names, key=str.lower)

    def _insert_folder(self, parent_key: str, key: str, has_children: bool) -> None:
        name = key.rsplit("/", 1)[-1]
        self.tree.insert(self._iid(parent_key), "end", iid=self._iid(key),
                         text=" " + name, image=self._box(key),
                         values=self._stat_values(key))
        if has_children:
            self.tree.insert(self._iid(key), "end", iid=self._iid(key) + _DUMMY)

    def _load_children(self, key: str) -> None:
        """Replace the placeholder under `key` with its real subfolders."""
        iid = self._iid(key)
        dummy = iid + _DUMMY
        if not self.tree.exists(dummy):
            return                                   # already loaded
        self.tree.delete(dummy)
        for name in self._list_subfolders(key):
            # Subfolders of the child are unknown until it is expanded.
            self._insert_folder(key, _join(key, name), has_children=True)

    def _on_tree_open(self, _event) -> None:
        # Load every open folder that still has its placeholder (the event
        # does not say which item was opened).
        for iid in self._all_iids():
            if self.tree.item(iid, "open") and self.tree.exists(iid + _DUMMY):
                self._load_children(iid[1:])

    def _refresh_boxes(self, key: str = "") -> None:
        stack = [self._iid(key)]
        while stack:
            iid = stack.pop()
            if iid.endswith(_DUMMY):
                continue
            self.tree.item(iid, image=self._box(iid[1:]))
            stack.extend(self.tree.get_children(iid))

    def _set_checked(self, key: str, on: bool) -> None:
        """Tick / untick a folder; all its subfolders (loaded or not) follow."""
        if self.src_root is None:
            return
        prefix = key + "/" if key else ""
        self.states = {k: v for k, v in self.states.items()
                       if not (k == key or (k.startswith(prefix) if key else True))}
        self.states[key] = on
        self._refresh_boxes(key)
        self._update_info()
        self._schedule_save()

    def _uncheck_all(self) -> None:
        self.states = {}
        if self.src_root is not None:
            self._refresh_boxes()
        self._update_info()
        self._schedule_save()

    def _on_tree_click(self, event) -> Optional[str]:
        iid = self.tree.identify_row(event.y)
        if iid and "image" in self.tree.identify_element(event.x, event.y):
            key = iid[1:]
            self._set_checked(key, not self._included(key))
            return "break"
        return None

    def _on_tree_space(self, _event) -> str:
        iid = self.tree.focus()
        if iid:
            self._set_checked(iid[1:], not self._included(iid[1:]))
        return "break"

    def _scan_folders(self) -> None:
        """Read the whole tree once: image count + size per folder."""
        root = self.src_root
        if root is None:
            messagebox.showinfo("Scan folders", "Choose a source folder first.")
            return
        stats: Dict[str, List[int]] = {}
        dirs: List[str] = []

        def on_image(key: str, _p: Path, size: int) -> None:
            s = stats.setdefault(key, [0, 0])
            s[0] += 1
            s[1] += size

        self._start_scan(
            "src",
            lambda cancel, prog: _walk(root, self.exts, cancel, prog,
                                       on_dir=dirs.append, on_image=on_image),
            lambda errors: self._folders_scanned(root, dirs, stats, errors))

    def _folders_scanned(self, root: Path, dirs: List[str],
                         stats: Dict[str, List[int]], errors: List[str]) -> None:
        if root != self.src_root:
            return
        self.folder_stats = {k: (v[0], v[1]) for k, v in stats.items()}
        self.stats_valid = True
        self._totals = self._recursive_totals(dirs)
        # Rebuild the tree fully loaded, keeping the open folders open.
        opened = {iid[1:] for iid in self._all_iids() if self.tree.item(iid, "open")}
        self.tree.delete(*self.tree.get_children())
        self.tree.insert("", "end", iid=self._iid(""), text=" " + (root.name or str(root)),
                         image=self._box(""), values=self._stat_values(""), open=True)
        children: Dict[str, List[str]] = {}
        for key in dirs:
            parent = _parent_key(key)
            if parent is not None:
                children.setdefault(parent, []).append(key)
        stack = [""]
        while stack:
            key = stack.pop()
            for child in sorted(children.get(key, []), key=str.lower):
                self._insert_folder(key, child, has_children=False)
                self.tree.item(self._iid(child), open=child in opened)
            stack.extend(children.get(key, []))
        self._scan_errors = errors
        self._update_info()

    def _recursive_totals(self, dirs: List[str]) -> Dict[str, Tuple[int, int]]:
        totals: Dict[str, List[int]] = {k: [0, 0] for k in dirs}
        totals.setdefault("", [0, 0])
        for key, (n, size) in self.folder_stats.items():
            k: Optional[str] = key
            while k is not None:
                t = totals.setdefault(k, [0, 0])
                t[0] += n
                t[1] += size
                k = _parent_key(k)
        return {k: (v[0], v[1]) for k, v in totals.items()}

    def _stat_values(self, key: str) -> Tuple[str, str]:
        if not self.stats_valid:
            return ("", "")
        n, size = self._totals.get(key, (0, 0))
        return (f"{n:,}", _human(size))

    def _all_iids(self) -> List[str]:
        out, stack = [], list(self.tree.get_children())
        while stack:
            iid = stack.pop()
            if not iid.endswith(_DUMMY):
                out.append(iid)
                stack.extend(self.tree.get_children(iid))
        return out

    # --- files view --------------------------------------------------------------

    def _on_tab_changed(self, _event) -> None:
        if self._tab() == _FILES_TAB and (
                self.files_root != self.src_root
                or self.files_recursive != self.v_subfolders.get()):
            self._scan_files()
        self._update_info()

    def _scan_files(self) -> None:
        root = self.src_root
        if root is None:
            return
        recursive = self.v_subfolders.get()
        entries: List[ScanEntry] = []
        self._start_scan(
            "src",
            lambda cancel, prog: _walk(root, self.exts, cancel, prog, recursive=recursive,
                                       on_image=lambda k, p, s: entries.append((p, s))),
            lambda errors: self._files_scanned(root, recursive, entries, errors))

    def _files_scanned(self, root: Path, recursive: bool, entries: List[ScanEntry],
                       errors: List[str]) -> None:
        if root != self.src_root:
            return
        entries.sort(key=lambda e: str(e[0]).lower())
        self.files_root, self.files_recursive = root, recursive
        self.src_scan = entries
        self.src_files = [p for p, _s in entries]
        self.src_list.delete(0, tk.END)
        if entries:
            self.src_list.insert(tk.END, *[
                f"{p.relative_to(root)}   ({_human(s)})" for p, s in entries])
        self._files_errors = errors
        self._update_info()

    # --- info line ---------------------------------------------------------------

    def _update_info(self) -> None:
        if "src" in self.scans:
            return                       # the scan shows its own progress
        if self.src_root is None:
            self.v_status.set("Ready.")
            return
        if self._tab() == _FOLDERS_TAB:
            if self.stats_valid:
                n_all, size_all = self._totals.get("", (0, 0))
                n_t = size_t = 0
                for key, (n, size) in self.folder_stats.items():
                    if self._included(key):
                        n_t += n
                        size_t += size
                text = (f"Total: {n_all:,} images, {_human(size_all)}   |   "
                        f"Ticked: {n_t:,} images, {_human(size_t)}")
                if self._scan_errors:
                    text += f"  ({len(self._scan_errors)} read errors, see log)"
            else:
                text = "Press 'Scan folders' to count images (not done "
                text += "automatically, to save disk access)."
            self.src_info.configure(text=text)
            ticked = sum(1 for v in self.states.values() if v)
            self.v_status.set("Folders ticked." if ticked else "Tick folders to convert.")
        else:
            if self.files_root == self.src_root:
                self.src_info.configure(text=self._summary(
                    len(self.src_scan), sum(s for _p, s in self.src_scan),
                    self._files_errors))
            n = len(self.src_list.curselection())
            self.v_status.set(f"{n} selected." if n else "Ready.")

    def _open_dst(self, _event) -> None:
        sel = self.dst_list.curselection()
        if sel:
            os.startfile(self.dst_files[sel[0]])  # type: ignore[attr-defined]

    # --- log / messages --------------------------------------------------------

    def _log(self, text: str) -> None:
        self.log.configure(state=tk.NORMAL)
        self.log.insert(tk.END, text + "\n")
        self.log.see(tk.END)
        self.log.configure(state=tk.DISABLED)

    def _current_scan(self, pane: str, gen: int) -> bool:
        cur = self.scans.get(pane)
        return cur is not None and cur[0] == gen

    def _poll(self) -> None:
        try:
            while True:
                msg = self.msgs.get_nowait()
                kind = msg[0]
                if kind == "log":
                    self._log(msg[1])
                elif kind == "status":
                    self.v_status.set(msg[1])
                elif kind == "progress":
                    self.progress.configure(maximum=msg[2], value=msg[1])
                elif kind == "scan_progress":
                    if self._current_scan(msg[1], msg[2]):
                        (self.src_info if msg[1] == "src" else self.dst_info).configure(
                            text=msg[3])
                elif kind == "scan_done":
                    if self._current_scan(msg[1], msg[2]):
                        self._scan_finished(msg[1], msg[3], msg[4])
                elif kind == "preview":
                    PreviewWindow(self, *msg[1:])
                elif kind == "done":
                    self.worker = None
                    self._after_run(msg[1])
        except queue.Empty:
            pass
        self._poll_job = self.after(100, self._poll)

    def _after_run(self, mode: Optional[str]) -> None:
        """mode: None (preview), "files" or "folders" (a convert run)."""
        self._sync_states()
        if mode == "files":
            self._scan_files()
            self._refresh_dst()
        elif mode == "folders" and self.stats_valid:
            # Counts may have changed (Move); don't re-read the disk on our own.
            self.stats_valid = False
            for iid in self._all_iids():
                self.tree.item(iid, values=("", ""))
        self._update_info()

    # --- work ------------------------------------------------------------------

    def _start(self, target, *args, mode: Optional[str]) -> None:
        self._save_settings()
        self.stop_event.clear()
        self.worker = threading.Thread(target=self._guard,
                                       args=(target, mode, *args), daemon=True)
        self.worker.start()
        self._sync_states()

    def _guard(self, target, mode: Optional[str], *args) -> None:
        try:
            target(*args)
        except Exception as e:      # never leave the UI stuck in "busy"
            self.msgs.put(("log", f"ERROR: {e}"))
            self.msgs.put(("status", f"Error: {e}"))
        finally:
            self.msgs.put(("done", mode))

    def _ensure_models(self) -> None:
        """Load CLIP once, in the worker thread (takes a few seconds)."""
        if self.models_ready:
            return
        from .classifiers import clip
        self.msgs.put(("status", "Loading AI models..."))
        runtime.set_intra_op_threads(os.cpu_count() or 4)
        clip.set_text_embeddings(clip.compute_text_embeddings(self.cfg))
        self.models_ready = True

    def _ticked(self) -> Set[str]:
        """Ticked categories, read in the UI thread (Tk variables must not be
        touched from the worker thread)."""
        return {c for c, v in self.v_cats.items() if v.get()}

    def _categorize(self, path: Path, c: conv.Converted, ticked: Set[str]):
        """Return (folder, ImageResult) for a converted image."""
        from .classifiers import image as image_clf
        self._ensure_models()
        img = c.image if c.image.mode == "RGB" else c.image.convert("RGB")
        if max(img.size) > 1024:
            img = img.copy()
            img.thumbnail((1024, 1024), Image.BILINEAR)
        r = image_clf.classify_loaded(path, img, c.meta, self.cfg)
        folder = r.folder
        if r.error or folder not in ticked:
            folder = self.cfg.fallback_folder
        return folder, r

    def _first_image(self, key: str) -> Optional[Path]:
        assert self.src_root is not None
        folder = self.src_root / key if key else self.src_root
        try:
            with os.scandir(folder) as it:
                names = sorted(e.name for e in it if not e.is_dir()
                               and os.path.splitext(e.name)[1].lower() in self.exts)
        except OSError as e:
            self._log(f"ERROR reading {folder}: {e}")
            return None
        return folder / names[0] if names else None

    def _preview(self) -> None:
        if self.src_root is None:
            messagebox.showinfo("Preview", "Choose a source folder first.")
            return
        if self._tab() == _FOLDERS_TAB:
            iid = self.tree.focus()
            path = self._first_image(iid[1:]) if iid else None
            if path is None:
                messagebox.showinfo("Preview", "Click a folder that directly contains "
                                               "images (its first image is previewed), "
                                               "or use the Files tab.")
                return
        else:
            sel = self.src_list.curselection()
            if not sel:
                messagebox.showinfo("Preview", "Select an image in the Files list first.")
                return
            path = self.src_files[sel[0]]
        ticked = self._ticked() if self.v_auto.get() else None
        self._start(self._do_preview, path, self._convert_options(), ticked, mode=None)

    def _do_preview(self, path: Path, opts: conv.ConvertOptions,
                    ticked: Optional[Set[str]]) -> None:
        self.msgs.put(("status", f"Converting {path.name}..."))
        c = conv.convert(path, opts)
        category = None
        if ticked is not None:
            folder, r = self._categorize(path, c, ticked)
            category = f"{r.folder} ({r.confidence:.2f})"
            if folder != r.folder:
                category += f" -> {folder} (unticked)"
        self.msgs.put(("preview", path, opts, c, category))
        self.msgs.put(("status", "Ready."))

    def _convert(self) -> None:
        root = self.src_root
        dst = self.v_dst.get().strip()
        if root is None:
            messagebox.showinfo("Convert", "Choose a source folder first.")
            return
        if not dst:
            messagebox.showinfo("Convert", "Choose a target folder first.")
            return
        dst_root = Path(dst).resolve()
        if dst_root == root.resolve() or root.resolve() in dst_root.parents:
            messagebox.showerror("Convert", "The target folder must not be the source "
                                            "folder or inside it.")
            return
        ticked = self._ticked() if self.v_auto.get() else None
        opts = self._convert_options()
        if self._tab() == _FOLDERS_TAB:
            if not any(self.states.values()):
                messagebox.showinfo("Convert", "Tick at least one folder.")
                return
            # Copy the tick states: the worker must not touch UI state.
            self._start(self._do_convert_folders, dict(self.states), root, dst_root,
                        opts, self.v_move.get(), ticked, mode="folders")
        else:
            if not self.src_files:
                messagebox.showinfo("Convert", "No images listed.")
                return
            sel = self.src_list.curselection()
            files = [self.src_files[i] for i in sel] if sel else list(self.src_files)
            self._start(self._do_convert, files, root, dst_root, opts,
                        self.v_move.get(), ticked, mode="files")

    def _do_convert_folders(self, states: Dict[str, bool], src_root: Path,
                            dst_root: Path, opts: conv.ConvertOptions, move: bool,
                            ticked: Optional[Set[str]]) -> None:
        """List the images of the ticked folders (reading only folders that
        are ticked or hold a ticked subfolder), then convert them."""
        def included(key: str) -> bool:
            k: Optional[str] = key
            while k is not None:
                if k in states:
                    return states[k]
                k = _parent_key(k)
            return False

        true_keys = [k for k, v in states.items() if v]

        def want_dir(key: str) -> bool:
            if included(key):
                return True
            prefix = key + "/" if key else ""
            return any(k.startswith(prefix) for k in true_keys)

        files: List[Path] = []
        self.msgs.put(("status", "Listing images in the ticked folders..."))
        errors = _walk(
            src_root, self.exts, self.stop_event,
            lambda d, n: self.msgs.put(("status", f"Listing... {n:,} images in {d:,} folders")),
            want_dir=want_dir,
            on_image=lambda key, p, _s: files.append(p) if included(key) else None)
        for e in errors[:20]:
            self.msgs.put(("log", f"ERROR reading {e}"))
        files.sort(key=lambda p: str(p).lower())
        self._do_convert(files, src_root, dst_root, opts, move, ticked)

    def _do_convert(self, files: List[Path], src_root: Path, dst_root: Path,
                    opts: conv.ConvertOptions, move: bool,
                    ticked: Optional[Set[str]]) -> None:
        """Mirror the source structure: <dst>/<rel dir>/[<category>/]<name>.
        `ticked` is None when auto categorize is off."""
        auto = ticked is not None
        n = len(files)
        before = after = 0
        done = 0
        self.msgs.put(("log", f"--- Converting {n} file(s) -> {dst_root} "
                              f"({opts.fmt.upper()}, resize {opts.resize_mode} "
                              f"{opts.resize_value}, move={move}, auto={auto})"))
        for i, path in enumerate(files, 1):
            if self.stop_event.is_set():
                self.msgs.put(("log", "Stopped by user."))
                break
            self.msgs.put(("progress", i - 1, n))
            self.msgs.put(("status", f"{i}/{n}: {path.name}"))
            try:
                c = conv.convert(path, opts)
                out_dir = dst_root / path.parent.relative_to(src_root)
                if ticked is not None:
                    folder, _r = self._categorize(path, c, ticked)
                    out_dir = out_dir / folder
                out = conv.unique_path(out_dir, path.stem, opts.ext)
                conv.write_output(out, c.data, path)
                line = (f"{path.relative_to(src_root)} -> {out.relative_to(dst_root)}  "
                        f"{_human(c.src_bytes)} -> {_human(len(c.data))} "
                        f"({len(c.data) * 100 / max(1, c.src_bytes):.0f}%)")
                if move:
                    send_to_recycle_bin(path)
                    line += "  [source recycled]"
                before += c.src_bytes
                after += len(c.data)
                done += 1
                self.msgs.put(("log", line))
            except Exception as e:
                self.msgs.put(("log", f"ERROR {path}: {e} (source kept)"))
        self.msgs.put(("progress", n, max(n, 1)))
        summary = (f"Done: {done}/{n} converted, {_human(before)} -> {_human(after)}"
                   + (f" ({after * 100 / before:.0f}%)" if before else ""))
        self.msgs.put(("log", summary))
        self.msgs.put(("status", summary))

    def _on_close(self) -> None:
        self.stop_event.set()
        for _gen, cancel in self.scans.values():
            cancel.set()
        if self._save_job is not None:
            self.after_cancel(self._save_job)
        if self._poll_job is not None:
            self.after_cancel(self._poll_job)
        self._save_settings()
        self.destroy()


class PreviewWindow(tk.Toplevel):
    """Shows the converted image at 1:1 (no fitting to the screen), with
    scrollbars, drag-to-pan and a size overlay that stays in the corner."""

    def __init__(self, master: App, path: Path, opts: conv.ConvertOptions,
                 c: conv.Converted, category: Optional[str]) -> None:
        super().__init__(master)
        # Decode the encoded bytes, so JPG artifacts are visible as they will be.
        img = Image.open(io.BytesIO(c.data))
        img.load()
        self.photo = ImageTk.PhotoImage(img)
        w, h = img.size
        self.title(f"Preview - {path.name} - {w}x{h} (1:1)")
        _center(self, w + 20, h + 20)

        self.canvas = tk.Canvas(self, highlightthickness=0, background="#333")
        hbar = ttk.Scrollbar(self, orient="horizontal", command=self.canvas.xview)
        vbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(
            scrollregion=(0, 0, w, h),
            xscrollcommand=lambda *a: (hbar.set(*a), self._place_overlay()),
            yscrollcommand=lambda *a: (vbar.set(*a), self._place_overlay()),
        )
        self.canvas.grid(row=0, column=0, sticky="nsew")
        vbar.grid(row=0, column=1, sticky="ns")
        hbar.grid(row=1, column=0, sticky="ew")
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)
        self.canvas.create_image(0, 0, anchor="nw", image=self.photo)

        sw0, sh0 = c.src_size
        pct = len(c.data) * 100 / max(1, c.src_bytes)
        fmt = "JPG q%d" % opts.quality if opts.fmt == "jpg" else "PNG"
        lines = [
            f"{path.name}:  {sw0}x{sh0} -> {w}x{h}  ({fmt})",
            f"File size: {_human(c.src_bytes)} -> {_human(len(c.data))}  ({pct:.0f}%)",
        ]
        if category:
            lines.append(f"Category: {category}")
        self.text = self.canvas.create_text(0, 0, anchor="nw", text="\n".join(lines),
                                            fill="white", font=("Segoe UI", 12, "bold"))
        self.bg = self.canvas.create_rectangle(0, 0, 0, 0, fill="black",
                                               outline="", stipple="gray75")
        self.canvas.tag_lower(self.bg, self.text)
        self._place_overlay()

        self.canvas.bind("<ButtonPress-1>", lambda e: self.canvas.scan_mark(e.x, e.y))
        self.canvas.bind("<B1-Motion>",
                         lambda e: self.canvas.scan_dragto(e.x, e.y, gain=1))
        self.canvas.bind("<MouseWheel>", lambda e: self.canvas.yview_scroll(
            -1 if e.delta > 0 else 1, "units"))
        self.canvas.bind("<Shift-MouseWheel>", lambda e: self.canvas.xview_scroll(
            -1 if e.delta > 0 else 1, "units"))
        self.bind("<Escape>", lambda e: self.destroy())
        self.focus_set()

    def _place_overlay(self) -> None:
        if not hasattr(self, "text"):
            return
        x, y = self.canvas.canvasx(10), self.canvas.canvasy(10)
        self.canvas.coords(self.text, x + 8, y + 6)
        x1, y1, x2, y2 = self.canvas.bbox(self.text)
        self.canvas.coords(self.bg, x1 - 8, y1 - 6, x2 + 8, y2 + 6)


def main() -> None:
    App().mainloop()
