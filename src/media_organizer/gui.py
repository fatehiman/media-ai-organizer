"""
Media Organizer GUI (Tkinter): convert / resize images, optionally
auto-categorize them with the same CLIP pipeline as the CLI, and optionally
send each source to the Recycle Bin.

Layout:   [ Source: Files | Folders tabs ] [ options + buttons ] [ Target file list ]
          [ log ........................................................................ ]

Folder scans run in a background thread (os.scandir: file sizes come with
the directory listing on Windows), so big trees don't freeze the window.

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
_FILES_TAB, _FOLDERS_TAB = 0, 1

# One scanned image: (absolute path, size in bytes).
ScanEntry = Tuple[Path, int]


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


def scan_images(
    root: Path,
    exts: Set[str],
    cancel: threading.Event,
    progress: Callable[[int, int], None],
) -> Tuple[List[ScanEntry], List[str]]:
    """Recursively list images under root.  Returns (entries, errors).
    A folder that can't be read is reported in `errors` and skipped, so a
    failing drive doesn't stop (or crash) the whole scan."""
    found: List[ScanEntry] = []
    errors: List[str] = []
    stack = [root]
    n_dirs = 0
    while stack and not cancel.is_set():
        folder = stack.pop()
        n_dirs += 1
        try:
            with os.scandir(folder) as it:
                for entry in it:
                    try:
                        if entry.is_dir(follow_symlinks=False):
                            stack.append(Path(entry.path))
                        elif os.path.splitext(entry.name)[1].lower() in exts:
                            found.append((Path(entry.path), entry.stat().st_size))
                    except OSError as e:
                        errors.append(f"{entry.path}: {e}")
        except OSError as e:
            errors.append(f"{folder}: {e}")
        if n_dirs % 20 == 0:
            progress(n_dirs, len(found))
    progress(n_dirs, len(found))
    found.sort(key=lambda e: str(e[0]).lower())
    return found, errors


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

        # Scan results.  src_root is the folder the scan belongs to.
        self.src_root: Optional[Path] = None
        self.src_scan: List[ScanEntry] = []
        self.src_files: List[Path] = []          # rows of the Files list
        self.dst_files: List[Path] = []
        self.scans: Dict[str, Tuple[int, threading.Event]] = {}
        self.checked: Set[str] = set()           # tree folders, rel. posix paths

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
        self.after(150, self._refresh_src)
        self.after(150, self._refresh_dst)

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
        self.view_tab = _FILES_TAB

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
        self.checked = set(data.get("checked_folders", []))
        self.view_tab = _FOLDERS_TAB if data.get("view") == "folders" else _FILES_TAB

    def _save_settings(self) -> None:
        self._save_job = None
        data = {}
        for key, var in self._simple_vars().items():
            try:
                data[key] = var.get()
            except tk.TclError:          # e.g. a half-typed spinbox value
                pass
        data["categories"] = {c: v.get() for c, v in self.v_cats.items()}
        data["checked_folders"] = sorted(self.checked)
        data["view"] = "folders" if self._tab() == _FOLDERS_TAB else "files"
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
        self._path_row(frame, self.v_src, self._browse_src, self._refresh_src)
        self.src_info, self.src_bar = self._scan_row(frame, 1)

        self.tabs = ttk.Notebook(frame)
        self.tabs.grid(row=3, column=0, columnspan=2, sticky="nsew")

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
                        command=self._fill_src_list).grid(row=1, column=0, sticky="w")
        ttk.Label(ft, text="Convert = selected files, or all listed if none selected",
                  foreground="gray").grid(row=2, column=0, sticky="w")
        self.src_list.bind("<Double-Button-1>", lambda e: self._preview())
        self.src_list.bind("<<ListboxSelect>>", lambda e: self._update_count())
        self.tabs.add(ft, text="Files")

        # Folders view
        dt = ttk.Frame(self.tabs)
        dt.columnconfigure(0, weight=1)
        dt.rowconfigure(0, weight=1)
        self.tree = ttk.Treeview(dt, columns=("here", "all"), selectmode="browse")
        self.tree.heading("#0", text="Folder")
        self.tree.heading("here", text="Images")
        self.tree.heading("all", text="Incl. subfolders")
        self.tree.column("#0", width=230, stretch=True)
        self.tree.column("here", width=60, anchor="e", stretch=False)
        self.tree.column("all", width=100, anchor="e", stretch=False)
        tsb = ttk.Scrollbar(dt, command=self.tree.yview)
        self.tree.configure(yscrollcommand=tsb.set)
        self.tree.grid(row=0, column=0, sticky="nsew")
        tsb.grid(row=0, column=1, sticky="ns")
        btns = ttk.Frame(dt)
        btns.grid(row=1, column=0, sticky="w")
        ttk.Button(btns, text="Check all",
                   command=lambda: self._check_all(True)).pack(side="left")
        ttk.Button(btns, text="Uncheck all",
                   command=lambda: self._check_all(False)).pack(side="left", padx=4)
        ttk.Label(dt, text="Ticking a folder ticks its subfolders too. "
                           "Convert = images in ticked folders.",
                  foreground="gray").grid(row=2, column=0, sticky="w")
        self.tree.bind("<Button-1>", self._on_tree_click)
        self.tree.bind("<space>", self._on_tree_space)
        self.tree.bind("<Double-Button-1>", lambda e: "break")   # no expand toggle
        self.tabs.add(dt, text="Folders")

        self.tabs.select(self.view_tab)
        self.tabs.bind("<<NotebookTabChanged>>",
                       lambda e: (self._update_count(), self._schedule_save()))

    def _target_pane(self, col: int) -> None:
        frame = ttk.LabelFrame(self, text="Target  (source folder structure is kept)")
        frame.grid(row=0, column=col, sticky="nsew", padx=_PAD, pady=_PAD)
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(3, weight=1)
        self._path_row(frame, self.v_dst, self._browse_dst, self._refresh_dst)
        self.dst_info, self.dst_bar = self._scan_row(frame, 1)
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
            return self.view_tab

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

    # --- scanning ----------------------------------------------------------------

    def _browse_src(self) -> None:
        d = filedialog.askdirectory(initialdir=self.v_src.get() or None)
        if d:
            self.v_src.set(os.path.normpath(d))
            self._refresh_src()

    def _browse_dst(self) -> None:
        d = filedialog.askdirectory(initialdir=self.v_dst.get() or None)
        if d:
            self.v_dst.set(os.path.normpath(d))
            self._refresh_dst()

    def _start_scan(self, pane: str, folder: str) -> None:
        """Scan `folder` in a background thread; results arrive as a
        ("scan_done", pane, ...) message.  A newer scan of the same pane
        cancels the older one."""
        old = self.scans.pop(pane, None)
        if old:
            old[1].set()
        info, bar = (self.src_info, self.src_bar) if pane == "src" else (
            self.dst_info, self.dst_bar)
        root = Path(folder.strip())
        # An empty entry must list nothing (Path("") would mean the cwd).
        if not folder.strip() or not root.is_dir():
            self._scan_finished(pane, root, [], [])
            info.configure(text="Folder not found." if folder.strip() else "")
            return
        gen = (old[0] + 1) if old else 1
        cancel = threading.Event()
        self.scans[pane] = (gen, cancel)
        info.configure(text="Scanning...")
        bar.grid()
        bar.start(12)
        self._sync_states()

        def progress(n_dirs: int, n_images: int) -> None:
            self.msgs.put(("scan_progress", pane, gen,
                           f"Scanning... {n_images:,} images in {n_dirs:,} folders"))

        def run() -> None:
            files, errors = scan_images(root, self.exts, cancel, progress)
            if not cancel.is_set():
                self.msgs.put(("scan_done", pane, gen, root, files, errors))

        threading.Thread(target=run, daemon=True).start()

    def _refresh_src(self) -> None:
        self._start_scan("src", self.v_src.get())

    def _refresh_dst(self) -> None:
        self._start_scan("dst", self.v_dst.get())

    def _scan_finished(self, pane: str, root: Path, files: List[ScanEntry],
                       errors: List[str]) -> None:
        self.scans.pop(pane, None)
        bar = self.src_bar if pane == "src" else self.dst_bar
        bar.stop()
        bar.grid_remove()
        for e in errors[:20]:
            self._log(f"ERROR reading {e}")
        if len(errors) > 20:
            self._log(f"... and {len(errors) - 20} more read errors")
        total = sum(size for _p, size in files)
        text = f"{len(files):,} images, {_human(total)}"
        if errors:
            text += f"  ({len(errors)} read errors, see log)"
        if pane == "src":
            self.src_root, self.src_scan = root, files
            self.src_info.configure(text=text)
            self._fill_src_list()
            self._fill_tree()
        else:
            self.dst_files = [p for p, _s in files]
            self.dst_list.delete(0, tk.END)
            self.dst_list.insert(tk.END, *[
                f"{p.relative_to(root)}   ({_human(s)})" for p, s in files])
            self.dst_info.configure(text=text)
        self._sync_states()

    # --- files view --------------------------------------------------------------

    def _fill_src_list(self) -> None:
        root = self.src_root
        rows = self.src_scan if self.v_subfolders.get() or root is None else [
            e for e in self.src_scan if e[0].parent == root]
        self.src_files = [p for p, _s in rows]
        self.src_list.delete(0, tk.END)
        if root is not None and rows:
            self.src_list.insert(tk.END, *[
                f"{p.relative_to(root)}   ({_human(s)})" for p, s in rows])
        self._update_count()

    # --- folders view ------------------------------------------------------------

    @staticmethod
    def _rel(root: Path, folder: Path) -> str:
        rel = folder.relative_to(root).as_posix()
        return "" if rel == "." else rel

    def _fill_tree(self) -> None:
        """Build the folder tree from the scan: every folder that holds
        images, plus its parents.  Item id = folder path relative to the
        source root ("" is the root itself)."""
        self.tree.delete(*self.tree.get_children())
        root = self.src_root
        if root is None:
            return
        here: Dict[str, int] = {}
        total: Dict[str, int] = {}
        for p, _s in self.src_scan:
            rel = self._rel(root, p.parent)
            here[rel] = here.get(rel, 0) + 1
            parts = rel.split("/") if rel else []
            for i in range(len(parts) + 1):
                key = "/".join(parts[:i])
                total[key] = total.get(key, 0) + 1
        if not total:
            return
        self.checked &= set(total)        # forget folders that are gone
        for key in sorted(total, key=lambda k: (k.count("/"), k.lower())):
            parent = key.rsplit("/", 1)[0] if "/" in key else ("" if key else None)
            text = (root.name or str(root)) if key == "" else key.rsplit("/", 1)[-1]
            self.tree.insert(
                "" if parent is None else self._iid(parent), "end",
                iid=self._iid(key), text=" " + text, open=key == "",
                image=self.box_on if key in self.checked else self.box_off,
                values=(f"{here.get(key, 0):,}", f"{total[key]:,}"),
            )
        self._update_count()

    @staticmethod
    def _iid(key: str) -> str:
        return "/" + key            # Treeview ids must not be "" (root id)

    def _set_checked(self, key: str, on: bool) -> None:
        """Tick / untick a folder and all its subfolders."""
        stack = [self._iid(key)]
        while stack:
            iid = stack.pop()
            k = iid[1:]
            (self.checked.add if on else self.checked.discard)(k)
            self.tree.item(iid, image=self.box_on if on else self.box_off)
            stack.extend(self.tree.get_children(iid))
        self._update_count()
        self._schedule_save()

    def _on_tree_click(self, event) -> Optional[str]:
        iid = self.tree.identify_row(event.y)
        if iid and "image" in self.tree.identify_element(event.x, event.y):
            key = iid[1:]
            self._set_checked(key, key not in self.checked)
            return "break"
        return None

    def _on_tree_space(self, _event) -> str:
        iid = self.tree.focus()
        if iid:
            self._set_checked(iid[1:], iid[1:] not in self.checked)
        return "break"

    def _check_all(self, on: bool) -> None:
        for iid in self.tree.get_children():
            self._set_checked(iid[1:], on)

    def _checked_files(self) -> List[Path]:
        root = self.src_root
        if root is None:
            return []
        return [p for p, _s in self.src_scan if self._rel(root, p.parent) in self.checked]

    def _update_count(self) -> None:
        if self._tab() == _FOLDERS_TAB:
            n = len(self._checked_files())
            self.v_status.set(f"{len(self.checked)} folders ticked, {n:,} images.")
        else:
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
                        self._scan_finished(msg[1], msg[3], msg[4], msg[5])
                elif kind == "preview":
                    PreviewWindow(self, *msg[1:])
                elif kind == "done":
                    self.worker = None
                    self._sync_states()
                    if msg[1]:                      # files were written / moved
                        self._refresh_src()
                        self._refresh_dst()
        except queue.Empty:
            pass
        self._poll_job = self.after(100, self._poll)

    # --- work ------------------------------------------------------------------

    def _start(self, target, *args, rescan: bool = True) -> None:
        self._save_settings()
        self.stop_event.clear()
        self.worker = threading.Thread(target=self._guard,
                                       args=(target, rescan, *args), daemon=True)
        self.worker.start()
        self._sync_states()

    def _guard(self, target, rescan: bool, *args) -> None:
        try:
            target(*args)
        except Exception as e:      # never leave the UI stuck in "busy"
            self.msgs.put(("log", f"ERROR: {e}"))
            self.msgs.put(("status", f"Error: {e}"))
        finally:
            self.msgs.put(("done", rescan))

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

    def _preview(self) -> None:
        if self._tab() == _FOLDERS_TAB:
            iid = self.tree.focus()
            root = self.src_root
            files = [p for p, _s in self.src_scan
                     if root is not None and iid and self._rel(root, p.parent) == iid[1:]]
            if not files:
                messagebox.showinfo("Preview", "Click a folder that contains images "
                                               "(its first image is previewed), or use "
                                               "the Files tab.")
                return
            path = files[0]
        else:
            sel = self.src_list.curselection()
            if not sel:
                messagebox.showinfo("Preview", "Select an image in the Files list first.")
                return
            path = self.src_files[sel[0]]
        ticked = self._ticked() if self.v_auto.get() else None
        self._start(self._do_preview, path, self._convert_options(), ticked,
                    rescan=False)

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
        if root is None or not self.src_scan:
            messagebox.showinfo("Convert", "No images in the source folder.")
            return
        if not dst:
            messagebox.showinfo("Convert", "Choose a target folder first.")
            return
        dst_root = Path(dst).resolve()
        if dst_root == root.resolve() or root.resolve() in dst_root.parents:
            messagebox.showerror("Convert", "The target folder must not be the source "
                                            "folder or inside it.")
            return
        if self._tab() == _FOLDERS_TAB:
            files = self._checked_files()
            if not files:
                messagebox.showinfo("Convert", "Tick at least one folder that "
                                               "contains images.")
                return
        else:
            sel = self.src_list.curselection()
            files = [self.src_files[i] for i in sel] if sel else list(self.src_files)
        ticked = self._ticked() if self.v_auto.get() else None
        self._start(self._do_convert, files, root, dst_root, self._convert_options(),
                    self.v_move.get(), ticked)

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
        self.msgs.put(("progress", n, n))
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
