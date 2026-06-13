from __future__ import annotations

import os
import subprocess
from pathlib import Path


def run_ecadstar_batch(
    cfg: dict,
    peb_path: Path,
    groot: Path,
) -> int:
    """
    Invoke Windows PowerShell + AutoHotkey to Load Batch in PI/EMI.
    Must run on Windows (or via powershell.exe from WSL calling Windows paths).
    """
    ec = cfg.get("ecadstar", {})
    erf = Path(ec["erf_path"])
    if not erf.is_file():
        raise FileNotFoundError(f"ERF not found: {erf}")
    if not peb_path.is_file():
        raise FileNotFoundError(f"PEB not found: {peb_path}")

    ps_script = groot / "scripts" / "run_ecadstar_piemi_batch.ps1"
    if not ps_script.is_file():
        raise FileNotFoundError(f"AHK runner not found: {ps_script}")

    ahk_exe = ec.get("ahk_exe")
    skip = bool(ec.get("skip_open_erf", False))

    cmd = [
        "powershell.exe",
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        str(ps_script),
        "-ErfPath",
        str(erf),
        "-PebPath",
        str(peb_path),
    ]
    if ahk_exe:
        cmd.extend(["-AhkExe", str(ahk_exe)])
    if skip:
        cmd.append("-SkipOpenErf")

    env = os.environ.copy()
    log_hint = Path(os.environ.get("TEMP", "/tmp")) / "ecadstar_piemi_batch.log"
    print(f"Running ECADSTAR automation: {' '.join(cmd)}")
    print(f"  Log (Windows): {log_hint}")
    proc = subprocess.run(cmd, cwd=str(groot))
    return int(proc.returncode)
