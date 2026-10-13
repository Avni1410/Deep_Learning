"""
Phase 8 - cross-platform checkpoint loading compatibility.

Checkpoints saved on Windows (torch.save with pathlib.WindowsPath objects
embedded in the 'config' dict, e.g. from argparse Path arguments) fail to
unpickle on Linux/Colab with an AttributeError on WindowsPath. This patches
pathlib so such checkpoints load correctly on non-Windows systems.

Call patch_path_compat() ONCE, before any torch.load() of a checkpoint that
may have been produced on a different OS. Safe to call on Windows too (it
becomes a no-op there).
"""
from __future__ import annotations

import pathlib
import platform


def patch_path_compat() -> None:
    if platform.system() != "Windows":
        # Checkpoints saved on Windows may contain pickled WindowsPath
        # objects (e.g. inside a saved argparse config). On Linux/Colab,
        # WindowsPath cannot be instantiated (no drive concept), so we
        # alias it to PosixPath for the duration of this process.
        pathlib.WindowsPath = pathlib.PosixPath  # type: ignore[assignment,misc]
