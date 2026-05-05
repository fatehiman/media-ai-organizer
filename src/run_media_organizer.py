"""
Top-level launcher for PyInstaller.

PyInstaller bundles a single entry-point module without preserving the
parent package, which breaks relative imports inside media_organizer/.
This thin launcher uses an absolute import so it always works.
"""
import multiprocessing
import sys

if __name__ == "__main__":
    multiprocessing.freeze_support()   # required on Windows
    from media_organizer.__main__ import main
    sys.exit(main())
