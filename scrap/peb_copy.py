"""Copy PEB Files (WSL-Safe).

Run: import and call ``copy_peb_to_folder`` (library helper, not a standalone entry script).
"""
from __future__ import annotations

import re
import shutil
from pathlib import Path, PureWindowsPath


def resolve_windows_path(path_str: str) -> Path:
    """Resolve a Windows path; on WSL use /mnt/<drive>/... when the drive is mounted."""
    if re.match(r"^[A-Za-z]:[\\/]", path_str):
        win = PureWindowsPath(path_str.replace("/", "\\"))
        drive = win.drive.rstrip(":").lower()
        mount_root = Path("/mnt") / drive
        if mount_root.is_dir():
            return mount_root.joinpath(*win.parts[1:])
        return Path(path_str)
    return Path(path_str)


def copy_peb_to_folder(
    peb_file: Path,
    dest_dir: str | Path,
    *,
    mkdir: bool = True,
) -> Path:
    """Copy *peb_file* into *dest_dir* (created if ``mkdir``). Returns destination path."""
    src = Path(peb_file)
    if not src.is_file():
        raise FileNotFoundError(f"PEB not found: {src}")

    dest = resolve_windows_path(str(dest_dir))
    if mkdir:
        dest.mkdir(parents=True, exist_ok=True)
    if not dest.is_dir():
        raise NotADirectoryError(f"PEB destination is not a directory: {dest}")

    target = dest / src.name
    shutil.copy2(str(src), str(target))
    return target
