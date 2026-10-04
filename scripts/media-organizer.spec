# -*- mode: python ; coding: utf-8 -*-
#
# PyInstaller spec: builds BOTH programs into one folder, so they share
# _internal\ (Python + libraries) and the models\ folder next to them:
#
#   dist\media-organizer\media-organizer.exe       CLI (console)
#   dist\media-organizer\media-organizer-gui.exe   GUI (Tkinter, no console)
#
# Run via scripts\build.ps1 (it renames the folder to MediaOrganizer\ and
# copies models, conf, third_party and README next to the exes).

import os

from PyInstaller.utils.hooks import collect_data_files
from PyInstaller.utils.hooks import collect_dynamic_libs
from PyInstaller.utils.hooks import collect_submodules

ROOT = os.path.dirname(SPECPATH)
SRC = os.path.join(ROOT, "src")

datas = []
binaries = []
hiddenimports = [
    "onnxruntime", "onnxruntime.capi._pybind_state", "pytesseract",
    "pillow_heif", "tokenizers", "cv2",
]
datas += collect_data_files("librosa")
datas += collect_data_files("soundfile")
datas += collect_data_files("pillow_heif")
datas += collect_data_files("onnxruntime")
binaries += collect_dynamic_libs("onnxruntime")
hiddenimports += collect_submodules("librosa")
hiddenimports += collect_submodules("onnxruntime")


def analysis(script):
    return Analysis(
        [os.path.join(SRC, script)],
        pathex=[SRC],
        binaries=binaries,
        datas=datas,
        hiddenimports=hiddenimports,
        hookspath=[],
        hooksconfig={},
        runtime_hooks=[],
        # torch may be in the dev venv (old model export); the app never
        # imports it, but optional hooks would pull in ~1 GB of DLLs.
        excludes=["torch", "torchvision", "tensorboard", "onnxscript"],
        noarchive=False,
        optimize=0,
    )


def exe(a, name, console):
    return EXE(
        PYZ(a.pure),
        a.scripts,
        [],
        exclude_binaries=True,
        name=name,
        debug=False,
        bootloader_ignore_signals=False,
        strip=False,
        upx=False,
        console=console,
        disable_windowed_traceback=False,
        argv_emulation=False,
        target_arch=None,
        codesign_identity=None,
        entitlements_file=None,
    )


cli = analysis("run_media_organizer.py")
gui = analysis("run_media_organizer_gui.py")

coll = COLLECT(
    exe(cli, "media-organizer", console=True),
    cli.binaries,
    cli.datas,
    exe(gui, "media-organizer-gui", console=False),
    gui.binaries,
    gui.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="media-organizer",
)
