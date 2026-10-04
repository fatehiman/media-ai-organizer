"""PyInstaller entry point for the GUI (see run_media_organizer.py for why
a thin launcher is used instead of the package's __main__)."""

import multiprocessing

from media_organizer.gui import main

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
