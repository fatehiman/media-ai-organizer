"""
Send a file to the Windows Recycle Bin (no confirmation dialog).

Uses SHFileOperationW with FOF_ALLOWUNDO via ctypes, so no extra package
is needed.  Note: on drives without a Recycle Bin (some USB sticks,
network shares) Windows deletes the file permanently instead.
"""

from __future__ import annotations

import ctypes
from ctypes import wintypes
from pathlib import Path


class _SHFILEOPSTRUCTW(ctypes.Structure):
    _fields_ = [
        ("hwnd", wintypes.HWND),
        ("wFunc", wintypes.UINT),
        ("pFrom", wintypes.LPCWSTR),
        ("pTo", wintypes.LPCWSTR),
        ("fFlags", ctypes.c_uint16),
        ("fAnyOperationsAborted", wintypes.BOOL),
        ("hNameMappings", ctypes.c_void_p),
        ("lpszProgressTitle", wintypes.LPCWSTR),
    ]


_FO_DELETE = 0x0003
_FOF_SILENT = 0x0004
_FOF_NOCONFIRMATION = 0x0010
_FOF_ALLOWUNDO = 0x0040
_FOF_NOERRORUI = 0x0400


def send_to_recycle_bin(path: Path) -> None:
    """Raise OSError if the file could not be recycled."""
    op = _SHFILEOPSTRUCTW()
    op.wFunc = _FO_DELETE
    # pFrom is a list of paths ending with an extra NUL; ctypes adds one.
    op.pFrom = str(Path(path).resolve()) + "\0"
    op.fFlags = _FOF_ALLOWUNDO | _FOF_NOCONFIRMATION | _FOF_SILENT | _FOF_NOERRORUI
    rc = ctypes.windll.shell32.SHFileOperationW(ctypes.byref(op))
    if rc != 0 or op.fAnyOperationsAborted:
        raise OSError(f"Recycle Bin delete failed (code {rc:#x}): {path}")
