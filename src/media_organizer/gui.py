"""
Media Organizer GUI (Tkinter): convert / resize images, optionally
auto-categorize them with the same CLIP pipeline as the CLI, and optionally
send each source to the Recycle Bin.

Layout:   [ Source folder + file list ] [ options + buttons ] [ Target folder + file list ]
          [ log ......................................................................... ]

Per file (one at a time, in a worker thread):
    convert in memory -> categorize the small converted image (using the
    source file's metadata) -> write target/<category>/<name>.<ext>
    -> send source to the Recycle Bin (if "Move (del source)" is ticked).
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
from typing import Dict, List, Optional, Set

from PIL import Image, ImageTk

from . import __version__
from . import config as config_module
from . import convert as conv
from . import runtime
from .trash import send_to_recycle_bin


_PAD = 6


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


class App(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title(f"Media Organizer {__version__} - Convert & Categorize")
        self.geometry("1280x760")
        self.minsize(1000, 600)

        try:
            self.cfg = config_module.load(_config_path(), require_paths=False)
        except (FileNotFoundError, ValueError) as e:
            messagebox.showerror("Config error", str(e))
            raise SystemExit(2)
        self.categories: List[str] = list(self.cfg.image_folders)

        self.src_files: List[Path] = []
        self.dst_files: List[Path] = []
        self.msgs: "queue.Queue[tuple]" = queue.Queue()
        self.stop_event = threading.Event()
        self.worker: Optional[threading.Thread] = None
        self.models_ready = False

        self._make_vars()
        self._load_settings()
        self._build()
        self._refresh_src()
        self._refresh_dst()
        self._sync_states()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.after(100, self._poll)

    # --- state ---------------------------------------------------------------

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

    def _load_settings(self) -> None:
        try:
            data = json.loads(_settings_path().read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        simple = {
            "src": self.v_src, "dst": self.v_dst, "subfolders": self.v_subfolders,
            "resize": self.v_resize, "percent": self.v_percent,
            "width": self.v_width, "height": self.v_height, "fmt": self.v_fmt,
            "quality": self.v_quality, "move": self.v_move, "auto": self.v_auto,
        }
        for key, var in simple.items():
            if key in data:
                try:
                    var.set(data[key])
                except tk.TclError:
                    pass
        for c, on in data.get("categories", {}).items():
            if c in self.v_cats:
                self.v_cats[c].set(bool(on))

    def _save_settings(self) -> None:
        data = {
            "src": self.v_src.get(), "dst": self.v_dst.get(),
            "subfolders": self.v_subfolders.get(), "resize": self.v_resize.get(),
            "percent": self._int(self.v_percent, 50),
            "width": self._int(self.v_width, 1920),
            "height": self._int(self.v_height, 1920),
            "fmt": self.v_fmt.get(), "quality": self._int(self.v_quality, 85),
            "move": self.v_move.get(), "auto": self.v_auto.get(),
            "categories": {c: v.get() for c, v in self.v_cats.items()},
        }
        try:
            p = _settings_path()
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except OSError:
            pass

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

        left = self._file_pane(0, "Source", self.v_src, self._browse_src)
        self.src_list = left
        ttk.Checkbutton(
            left.master, text="Include subfolders", variable=self.v_subfolders,
            command=self._refresh_src,
        ).grid(row=3, column=0, columnspan=3, sticky="w")
        self.src_list.configure(selectmode=tk.EXTENDED)
        self.src_list.bind("<Double-Button-1>", lambda e: self._preview())
        self.src_list.bind("<<ListboxSelect>>", lambda e: self._update_count())

        right = self._file_pane(2, "Target", self.v_dst, self._browse_dst)
        self.dst_list = right
        ttk.Button(right.master, text="Refresh", command=self._refresh_dst).grid(
            row=3, column=0, sticky="w")
        self.dst_list.bind("<Double-Button-1>", self._open_dst)

        self._center(1)

        logf = ttk.LabelFrame(self, text="Log")
        logf.grid(row=1, column=0, columnspan=3, sticky="nsew", padx=_PAD, pady=(0, _PAD))
        logf.columnconfigure(0, weight=1)
        self.log = tk.Text(logf, height=7, state=tk.DISABLED, wrap="none")
        sb = ttk.Scrollbar(logf, command=self.log.yview)
        self.log.configure(yscrollcommand=sb.set)
        self.log.grid(row=0, column=0, sticky="nsew")
        sb.grid(row=0, column=1, sticky="ns")

    def _file_pane(self, col: int, title: str, var: tk.StringVar, browse) -> tk.Listbox:
        frame = ttk.LabelFrame(self, text=title)
        frame.grid(row=0, column=col, sticky="nsew", padx=_PAD, pady=_PAD)
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(2, weight=1)
        entry = ttk.Entry(frame, textvariable=var)
        entry.grid(row=0, column=0, sticky="ew", padx=(0, 4))
        entry.bind("<Return>", lambda e: (self._refresh_src(), self._refresh_dst()))
        ttk.Button(frame, text="Browse...", command=browse).grid(row=0, column=1)
        count = ttk.Label(frame, text="")
        count.grid(row=1, column=0, columnspan=2, sticky="w")
        lb = tk.Listbox(frame, activestyle="none", exportselection=False)
        sb = ttk.Scrollbar(frame, command=lb.yview)
        lb.configure(yscrollcommand=sb.set)
        lb.grid(row=2, column=0, sticky="nsew")
        sb.grid(row=2, column=1, sticky="ns")
        lb.count_label = count  # type: ignore[attr-defined]
        return lb

    def _center(self, col: int) -> None:
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
        self.btn_stop.pack(fill="x", pady=2)
        ttk.Label(c, text="Convert = selected files,\nor all if none selected",
                  foreground="gray", justify="center").pack(pady=(0, _PAD))

        self.progress = ttk.Progressbar(c, mode="determinate")
        self.progress.pack(fill="x")
        ttk.Label(c, textvariable=self.v_status, wraplength=230).pack(fill="x")

    def _on_quality(self, value: str) -> None:
        self.v_quality.set(int(float(value)))
        self.q_label.configure(text=str(self.v_quality.get()))

    def _sync_states(self) -> None:
        for mode, sp in self.spins.items():
            sp.configure(state="normal" if self.v_resize.get() == mode else "disabled")
        self.q_scale.configure(state="normal" if self.v_fmt.get() == "jpg" else "disabled")
        self.q_label.configure(text=str(self.v_quality.get()))
        for cb in self.cat_checks:
            cb.configure(state="normal" if self.v_auto.get() else "disabled")
        busy = self.worker is not None and self.worker.is_alive()
        self.btn_convert.configure(state="disabled" if busy else "normal")
        self.btn_preview.configure(state="disabled" if busy else "normal")
        self.btn_stop.configure(state="normal" if busy else "disabled")

    # --- file lists ------------------------------------------------------------

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

    def _list_images(self, folder: str, recursive: bool) -> List[Path]:
        # An empty entry must list nothing (Path("") would mean the cwd).
        root = Path(folder.strip())
        if not folder.strip() or not root.is_dir():
            return []
        exts = set(self.cfg.ext_image)
        it = root.rglob("*") if recursive else root.iterdir()
        return sorted(p for p in it if p.is_file() and p.suffix.lower() in exts)

    def _fill(self, lb: tk.Listbox, root: Path, files: List[Path]) -> None:
        lb.delete(0, tk.END)
        total = 0
        for p in files:
            try:
                size = p.stat().st_size
            except OSError:
                size = 0
            total += size
            lb.insert(tk.END, f"{p.relative_to(root)}   ({_human(size)})")
        lb.count_label.configure(  # type: ignore[attr-defined]
            text=f"{len(files)} images, {_human(total)}")

    def _refresh_src(self) -> None:
        self.src_files = self._list_images(self.v_src.get(), self.v_subfolders.get())
        self._fill(self.src_list, Path(self.v_src.get().strip()), self.src_files)

    def _refresh_dst(self) -> None:
        self.dst_files = self._list_images(self.v_dst.get(), True)
        self._fill(self.dst_list, Path(self.v_dst.get().strip()), self.dst_files)

    def _update_count(self) -> None:
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
                elif kind == "preview":
                    PreviewWindow(self, *msg[1:])
                elif kind == "done":
                    self.worker = None
                    self._sync_states()
                    self._refresh_src()
                    self._refresh_dst()
        except queue.Empty:
            pass
        self.after(100, self._poll)

    # --- work ------------------------------------------------------------------

    def _start(self, target, *args) -> None:
        self._save_settings()
        self.stop_event.clear()
        self.worker = threading.Thread(target=self._guard, args=(target, *args),
                                       daemon=True)
        self.worker.start()
        self._sync_states()

    def _guard(self, target, *args) -> None:
        try:
            target(*args)
        except Exception as e:      # never leave the UI stuck in "busy"
            self.msgs.put(("log", f"ERROR: {e}"))
            self.msgs.put(("status", f"Error: {e}"))
        finally:
            self.msgs.put(("done",))

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
        sel = self.src_list.curselection()
        if not sel:
            messagebox.showinfo("Preview", "Select an image in the left list first.")
            return
        ticked = self._ticked() if self.v_auto.get() else None
        self._start(self._do_preview, self.src_files[sel[0]], self._convert_options(),
                    ticked)

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
        if not self.src_files:
            messagebox.showinfo("Convert", "No images in the source folder.")
            return
        dst = self.v_dst.get().strip()
        if not dst:
            messagebox.showinfo("Convert", "Choose a target folder first.")
            return
        sel = self.src_list.curselection()
        files = [self.src_files[i] for i in sel] if sel else list(self.src_files)
        ticked = self._ticked() if self.v_auto.get() else None
        self._start(self._do_convert, files, Path(dst), self._convert_options(),
                    self.v_move.get(), ticked)

    def _do_convert(self, files: List[Path], dst_root: Path,
                    opts: conv.ConvertOptions, move: bool,
                    ticked: Optional[Set[str]]) -> None:
        """`ticked` is None when auto categorize is off."""
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
                folder = ""
                if ticked is not None:
                    folder, _r = self._categorize(path, c, ticked)
                out_dir = dst_root / folder if folder else dst_root
                out = conv.unique_path(out_dir, path.stem, opts.ext)
                conv.write_output(out, c.data, path)
                line = (f"{path.name} -> {out.relative_to(dst_root)}  "
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
                self.msgs.put(("log", f"ERROR {path}: {e}"))
        self.msgs.put(("progress", n, n))
        summary = (f"Done: {done}/{n} converted, {_human(before)} -> {_human(after)}"
                   + (f" ({after * 100 / before:.0f}%)" if before else ""))
        self.msgs.put(("log", summary))
        self.msgs.put(("status", summary))

    def _on_close(self) -> None:
        self.stop_event.set()
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

        sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
        self.geometry(f"{min(w + 20, sw - 40)}x{min(h + 20, sh - 80)}+10+10")

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
