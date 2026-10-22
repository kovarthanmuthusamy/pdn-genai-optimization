"""eCADSTAR driver: Windows paths from WSL, one headless batch run, waiting for and moving outputs.

What: eCADSTAR runs on Windows; this code runs in WSL and converts between drive-letter paths and their WSL mount form
    A batch is run with ``engineer.exe <erf> --batch <peb> --batch-auto-exit``, which opens the
    board project, runs every group of the PEB, writes PI-1..PI-N into the project's output folder
    (PEB order: group n -> PI-n) and exits. The outputs are then moved into the campaign folder.
    eCADSTAR is a shared host that runs one job at a time: run_ecadstar_batch_headless refuses to
    start unless settings.ECADSTAR_RUN is True; write_command() records the command line instead.
Usage (import only): resolve_windows_path, windows_path_str, clear_ecadstar_lock,
    stage_peb_for_ecadstar, write_command, run_ecadstar_batch_headless, wait_for_pi_outputs,
    move_pi_outputs.
"""
from __future__ import annotations

import re
import shutil
import subprocess
import time
from pathlib import Path, PureWindowsPath

from thesis_code.common import settings

_PI_RE = re.compile(r"^PI-(\d+)(?:\..+)?$", re.IGNORECASE)


def _is_windows_abs(path_str: str) -> bool:
    return bool(re.match(r"^[A-Za-z]:[\\/]", path_str))


def _windows_path_from_wsl(path: Path) -> str | None:
    parts = path.parts
    if len(parts) >= 3 and parts[0] == "/" and parts[1] == "mnt" and len(parts[2]) == 1:
        return f"{parts[2].upper()}:\\" + "\\".join(parts[3:])
    return None


def windows_path_str(path: Path | str) -> str:
    """Drive-letter form of a Windows path or of a WSL mount path."""
    raw = str(path)
    if _is_windows_abs(raw):
        return PureWindowsPath(raw.replace("/", "\\")).as_posix().replace("/", "\\")
    p = Path(path)
    return _windows_path_from_wsl(p) or _windows_path_from_wsl(p.resolve()) or str(p.resolve())


def resolve_windows_path(path_str: str) -> Path:
    """WSL mount form of a drive-letter path (other paths unchanged)."""
    if _is_windows_abs(path_str):
        win = PureWindowsPath(path_str.replace("/", "\\"))
        mount_root = Path("/mnt") / win.drive.rstrip(":").lower()
        if mount_root.is_dir():
            return mount_root.joinpath(*win.parts[1:])
    return Path(path_str)


def parse_pi_number(name: str) -> int | None:
    m = _PI_RE.match(name)
    return int(m.group(1)) if m else None


def clear_ecadstar_lock(erf_path: str) -> bool:
    """Remove the project lock file (<erf>.rlk) a crashed run leaves behind."""
    erf = resolve_windows_path(erf_path)
    lock = erf.parent / f"{erf.stem}.rlk"
    if not lock.is_file():
        return False
    lock.unlink()
    print(f"  Removed eCADSTAR lock: {windows_path_str(lock)}")
    return True


def stage_peb_for_ecadstar(peb_path: Path, *, erf_path: str) -> Path:
    """Copy the PEB beside the board project, where engineer.exe reads it."""
    peb_path = peb_path.resolve()
    if not peb_path.is_file():
        raise FileNotFoundError(f"PEB not found: {peb_path}")
    erf_dir = resolve_windows_path(erf_path).parent
    staged = erf_dir / peb_path.name
    erf_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(peb_path, staged)
    print(f"    Staged PEB: {windows_path_str(staged)}")
    return staged


def _argline(peb_win: str, erf_win: str) -> str:
    return f'"{erf_win}" --batch "{peb_win}" --batch-auto-exit'


def write_command(peb_path: Path, *, erf_path: str, engineer_exe: str) -> Path:
    """Append the command that would simulate ``peb_path`` to commands.txt in the PEB folder."""
    erf_win = windows_path_str(erf_path)
    staged_win = str(PureWindowsPath(erf_win).parent / peb_path.name)
    out = peb_path.parent / "commands.txt"
    with out.open("a", encoding="utf-8") as f:
        f.write(f'copy "{peb_path.name}" "{staged_win}"\n"{engineer_exe}" {_argline(staged_win, erf_win)}\n')
    return out


def run_ecadstar_batch_headless(peb_path: Path, *, erf_path: str, engineer_exe: str,
                                timeout_sec: int = 172_800) -> int:
    """Run one batch and block until engineer.exe exits. Returns its exit status (124 = timeout)."""
    if not settings.ECADSTAR_RUN:
        raise SystemExit("settings.ECADSTAR_RUN is False -- eCADSTAR is not started")
    erf = resolve_windows_path(erf_path)
    if not erf.is_file():
        raise FileNotFoundError(f"ERF not found: {erf}")
    if not peb_path.is_file():
        raise FileNotFoundError(f"PEB not found: {peb_path}")
    argline = _argline(windows_path_str(peb_path), windows_path_str(erf))
    ps = (f"$exe = '{engineer_exe}';"
          f"$argline = '{argline}';"
          "$p = Start-Process -FilePath $exe -ArgumentList $argline -PassThru -NoNewWindow;"
          f"if ($p.WaitForExit({int(timeout_sec * 1000)})) {{ Write-Output ('HEADLESS_EXIT code=' + $p.ExitCode); exit 0 }}"
          " else { Write-Output 'HEADLESS_TIMEOUT'; try { $p.Kill() } catch {}; exit 124 }")
    print(f"Running eCADSTAR headless: engineer.exe --batch {windows_path_str(peb_path)}")
    proc = subprocess.run(["powershell.exe", "-NoProfile", "-Command", ps], capture_output=True, text=True)
    if (proc.stdout or "").strip():
        print(f"  {proc.stdout.strip()}")
    if (proc.stderr or "").strip():
        print(f"  [engineer stderr] {proc.stderr.strip()}")
    return int(proc.returncode)


def _pi_folder_index(emc_dir: Path) -> dict[int, Path]:
    return {n: item for item in emc_dir.iterdir()
            if (n := parse_pi_number(item.name)) is not None and item.is_dir()}


def _newest(folder: Path, pattern: str) -> float | None:
    files = list(folder.rglob(pattern))
    return max((f.stat().st_mtime for f in files), default=None)


def wait_for_pi_outputs(emc_output_dir: str, *, pi_count: int, batch_start_ts: float, mode: str,
                        timeout_sec: int, poll_sec: int) -> None:
    """Poll until PI-1..PI-N hold outputs newer than the batch start (map: .map, spectrum: .csv)."""
    emc = resolve_windows_path(emc_output_dir)
    pattern = "Z_*MHz.map" if mode == "distribution" else "*.csv"
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        index = _pi_folder_index(emc)
        ready = sum(1 for n in range(1, pi_count + 1)
                    if n in index and (t := _newest(index[n], pattern)) is not None and t >= batch_start_ts - 2.0)
        if ready >= pi_count:
            print(f"  all {pi_count} PI outputs ready")
            return
        print(f"  {ready}/{pi_count} ready -- retry in {poll_sec}s")
        time.sleep(poll_sec)
    raise SystemExit(f"timeout waiting for PI-1..PI-{pi_count} ({mode}) in {emc}")


def move_pi_outputs(*, source_emc_dir: str, dest_dir: Path, pi_count: int) -> int:
    """Move PI-1..PI-{pi_count} from the project output folder into ``dest_dir`` (names unchanged);
    PI-* entries already in ``dest_dir`` are removed first."""
    source = resolve_windows_path(source_emc_dir)
    if not source.is_dir():
        raise FileNotFoundError(f"EMC dir missing: {source}")
    dest_dir.mkdir(parents=True, exist_ok=True)
    for child in dest_dir.iterdir():
        if parse_pi_number(child.name) is not None:
            shutil.rmtree(child) if child.is_dir() else child.unlink()
    index = _pi_folder_index(source)
    for pi_num in range(1, pi_count + 1):
        if pi_num not in index:
            raise FileNotFoundError(f"Missing PI-{pi_num} under {source}")
        shutil.move(str(index[pi_num]), str(dest_dir / index[pi_num].name))
    return pi_count
