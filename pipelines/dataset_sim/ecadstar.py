"""ECADStar batch automation helpers for combinations simulation."""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path, PureWindowsPath

_PI_RE = re.compile(r"^PI-(\d+)(?:\..+)?$", re.IGNORECASE)
_MAP_MHZ_RE = re.compile(r"Z_([\d.]+)MHz", re.IGNORECASE)


def _is_windows_abs(path_str: str) -> bool:
    return bool(re.match(r"^[A-Za-z]:[\\/]", path_str))


def _wsl_path_from_windows(path_str: str) -> Path | None:
    if not _is_windows_abs(path_str):
        return None
    win = PureWindowsPath(path_str.replace("/", "\\"))
    drive = win.drive.rstrip(":").lower()
    mount_root = Path("/mnt") / drive
    if not mount_root.is_dir():
        return None
    return mount_root.joinpath(*win.parts[1:])


def _windows_path_from_wsl(path: Path) -> str | None:
    parts = path.parts
    if len(parts) >= 3 and parts[0] == "/" and parts[1] == "mnt" and len(parts[2]) == 1:
        drive = parts[2].upper()
        rest = "\\".join(parts[3:])
        return f"{drive}:\\{rest}"
    return None


def _windows_path_from_mangled_wsl(path: Path) -> str | None:
    """Recover C:\\... when a Windows path was resolved under the Linux cwd."""
    for index, part in enumerate(path.parts):
        if _is_windows_abs(part):
            tail = "\\".join(path.parts[index + 1 :])
            head = part.replace("/", "\\")
            return f"{head}\\{tail}" if tail else head
    return None


def windows_path_str(path: Path | str) -> str:
    raw = str(path)
    if _is_windows_abs(raw):
        return PureWindowsPath(raw.replace("/", "\\")).as_posix().replace("/", "\\")

    p = Path(path)
    win = _windows_path_from_wsl(p)
    if win is not None:
        return win

    win = _windows_path_from_mangled_wsl(p)
    if win is not None:
        return win

    resolved = p.resolve()
    win = _windows_path_from_wsl(resolved) or _windows_path_from_mangled_wsl(resolved)
    if win is not None:
        return win
    return str(resolved)


def resolve_windows_path(path_str: str) -> Path:
    wsl_path = _wsl_path_from_windows(path_str)
    if wsl_path is not None:
        return wsl_path
    if _is_windows_abs(path_str):
        return Path(path_str)
    return Path(path_str)


def parse_pi_number(name: str) -> int | None:
    m = _PI_RE.match(name)
    return int(m.group(1)) if m else None


_KILL_PS = (
    "$titles=@('*eCADSTAR*PI/EMI*','*PI/EMI Analysis*');"
    "$procs=Get-Process | Where-Object {"
    " $mt=$_.MainWindowTitle;"
    " if([string]::IsNullOrEmpty($mt)){$false}"
    " else{ ($titles | Where-Object { $mt -like $_ }).Count -gt 0 } };"
    "$ids=@($procs | ForEach-Object { $_.Id });"
    "foreach($p in $procs){ try{ Stop-Process -Id $p.Id -Force -ErrorAction Stop }catch{} }"
    "if($ids.Count -gt 0){ Write-Output ('KILLED ' + ($ids -join ',')) }"
    "else{ Write-Output 'NONE' }"
)


def kill_ecadstar_windows(*, verbose: bool = True) -> list[int]:
    """Force-close any running ECADStar PI/EMI instance (matched by window title).

    Mirrors the manual recovery kill: find the "eCADSTAR PI/EMI Analysis" window
    and Stop-Process it. Use ONLY before a fresh ERF open (never on skipopen — that
    would destroy the persistent instance). Returns the list of killed PIDs; [] if
    none were running or PowerShell is unavailable.
    """
    try:
        proc = subprocess.run(
            ["powershell.exe", "-NoProfile", "-Command", _KILL_PS],
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        if verbose:
            print(f"  kill_ecadstar: skipped (PowerShell unavailable: {exc})")
        return []

    out = (proc.stdout or "").strip()
    if out.startswith("KILLED"):
        pids = [int(x) for x in out.split(" ", 1)[1].split(",") if x.strip().isdigit()]
        if verbose:
            print(f"  Killed stale ECADStar PID(s): {pids}")
        return pids
    if verbose:
        print("  No running ECADStar PI/EMI instance to kill.")
    return []


def clear_ecadstar_lock(erf_path: str) -> bool:
    erf = resolve_windows_path(erf_path)
    lock = erf.parent / f"{erf.stem}.rlk"
    if not lock.is_file():
        return False
    lock.unlink()
    print(f"  Removed ECADStar lock: {windows_path_str(lock)}")
    return True


def stage_peb_for_ecadstar(
    peb_path: Path,
    *,
    erf_path: str,
    peb_copy_dest: str | None,
) -> Path:
    """Copy generated PEB beside .erf and optional PEB folder."""
    peb_path = peb_path.resolve()
    if not peb_path.is_file():
        raise FileNotFoundError(f"PEB not found: {peb_path}")

    staged: list[Path] = []
    if peb_copy_dest:
        dest = resolve_windows_path(peb_copy_dest) / peb_path.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(peb_path, dest)
        staged.append(dest)

    erf_dir = resolve_windows_path(erf_path).parent
    erf_peb = erf_dir / peb_path.name
    erf_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(peb_path, erf_peb)
    staged.append(erf_peb)

    for copy_path in staged:
        print(f"    Staged PEB: {windows_path_str(copy_path)}")

    return staged[0] if staged else peb_path


def run_ecadstar_batch_headless(
    peb_path: Path,
    *,
    erf_path: str,
    engineer_exe: str,
    impulse_port: int | None = None,
    timeout_sec: int = 172_800,
    verbose: bool = True,
) -> int:
    """Run a PI/EMI batch fully headless via ``engineer.exe --batch --batch-auto-exit``.

    This is the native eCADSTAR batch interface — no AutoHotkey, no GUI focus, no
    window activation. ``engineer.exe`` opens the .erf, executes the .peb batch,
    writes PI-1..N outputs, and quits itself (``--batch-auto-exit``). Because the
    call blocks until the process exits, process self-exit is the ground-truth
    completion signal.

    Returns 0 on clean self-exit, 124 on timeout (process force-killed), or -1 if
    PowerShell is unavailable.
    """
    erf = resolve_windows_path(erf_path)
    if not erf.is_file():
        raise FileNotFoundError(f"ERF not found: {erf}")
    if not peb_path.is_file():
        raise FileNotFoundError(f"PEB not found: {peb_path}")

    erf_win = windows_path_str(erf)
    peb_win = windows_path_str(peb_path)
    exe_win = engineer_exe if _is_windows_abs(engineer_exe) else windows_path_str(Path(engineer_exe))

    argline = f'"{erf_win}" --batch "{peb_win}" --batch-auto-exit'
    if impulse_port is not None:
        argline += f" --impulse-port {int(impulse_port)}"

    timeout_ms = int(timeout_sec * 1000)
    ps = (
        f"$exe = '{exe_win}';"
        f"$argline = '{argline}';"
        "$p = Start-Process -FilePath $exe -ArgumentList $argline -PassThru -NoNewWindow;"
        f"if ($p.WaitForExit({timeout_ms})) {{ Write-Output ('HEADLESS_EXIT code=' + $p.ExitCode); exit 0 }}"
        " else { Write-Output 'HEADLESS_TIMEOUT'; try { $p.Kill() } catch {}; exit 124 }"
    )

    if verbose:
        print(f"Running ECADSTAR headless: engineer.exe --batch → {peb_win}")
        print(f"  engineer.exe : {exe_win}")
        print("  --batch-auto-exit (no GUI window / no AutoHotkey / background-safe)")

    try:
        proc = subprocess.run(
            ["powershell.exe", "-NoProfile", "-Command", ps],
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        print(f"  headless run failed (PowerShell unavailable: {exc})")
        return -1

    out = (proc.stdout or "").strip()
    err = (proc.stderr or "").strip()
    if verbose and out:
        print(f"  {out}")
    if err:
        print(f"  [engineer stderr] {err}")
    return int(proc.returncode)


def _pi_folder_index(emc_dir: Path) -> dict[int, Path]:
    index: dict[int, Path] = {}
    for item in emc_dir.iterdir():
        pi_num = parse_pi_number(item.name)
        if pi_num is not None and item.is_dir():
            index[pi_num] = item
    return index


def _pi_folder(emc_dir: Path, pi_num: int) -> Path | None:
    return _pi_folder_index(emc_dir).get(pi_num)


def _fmt_ts(ts: float) -> str:
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")


def _map_matches_mhz(path: Path, expected_mhz: float) -> bool:
    m = _MAP_MHZ_RE.search(path.name)
    if not m:
        return False
    return abs(float(m.group(1)) - expected_mhz) < 0.5


def _pi_map_files(pi_dir: Path, expected_mhz: float | None = None) -> list[Path]:
    maps = list(pi_dir.rglob("Z_*MHz.map"))
    if expected_mhz is None:
        return maps
    return [p for p in maps if _map_matches_mhz(p, expected_mhz)]


def _pi_map_mtime(pi_dir: Path, expected_mhz: float | None = None) -> float | None:
    maps = _pi_map_files(pi_dir, expected_mhz)
    if not maps:
        return None
    return max(m.stat().st_mtime for m in maps)


def _pi_csv_mtime(pi_dir: Path) -> float | None:
    csvs = list(pi_dir.rglob("*.csv"))
    if not csvs:
        return None
    return max(m.stat().st_mtime for m in csvs)


def _pi_ready_distribution(
    pi_index: dict[int, Path],
    pi_num: int,
    since_ts: float,
    expected_mhz: float | None = None,
) -> bool:
    folder = pi_index.get(pi_num)
    if folder is None:
        return False
    mtime = _pi_map_mtime(folder, expected_mhz)
    return mtime is not None and mtime >= since_ts - 2.0


def _pi_ready_impedance(pi_index: dict[int, Path], pi_num: int, since_ts: float) -> bool:
    folder = pi_index.get(pi_num)
    if folder is None:
        return False
    mtime = _pi_csv_mtime(folder)
    return mtime is not None and mtime >= since_ts - 2.0


def _inspect_pi_folder(
    folder: Path | None,
    *,
    since_ts: float,
    expected_mhz: float | None,
) -> dict[str, object]:
    if folder is None:
        return {"exists": False}
    maps = _pi_map_files(folder, expected_mhz)
    mabs = list(folder.rglob("Z_*MHz.mab"))
    mpxs = list(folder.rglob("Z_*MHz.mpx"))
    csvs = list(folder.rglob("*.csv"))
    power_gnd = folder / "Power_GND"
    pg_files = sorted(p.name for p in power_gnd.iterdir()) if power_gnd.is_dir() else []
    map_mtime = max((m.stat().st_mtime for m in maps), default=None)
    return {
        "exists": True,
        "path": str(folder),
        "power_gnd_files": pg_files[:8],
        "n_map": len(maps),
        "n_mab": len(mabs),
        "n_mpx": len(mpxs),
        "n_csv": len(csvs),
        "map_names": [m.name for m in maps[:3]],
        "map_mtime": map_mtime,
        "map_fresh": map_mtime is not None and map_mtime >= since_ts - 2.0,
    }


def _summarize_wait_state(
    pi_index: dict[int, Path],
    pi_count: int,
    since_ts: float,
    mode: str,
    expected_mhz: float | None,
) -> dict[str, int]:
    counts = {
        "missing_folder": 0,
        "fresh_map": 0,
        "stale_map": 0,
        "mab_only": 0,
        "no_artifacts": 0,
        "fresh_csv": 0,
        "stale_csv": 0,
    }
    for pi_num in range(1, pi_count + 1):
        folder = pi_index.get(pi_num)
        if folder is None:
            counts["missing_folder"] += 1
            continue
        if mode == "distribution":
            maps = _pi_map_files(folder, expected_mhz)
            mabs = list(folder.rglob("Z_*MHz.mab"))
            if maps:
                mtime = max(m.stat().st_mtime for m in maps)
                if mtime >= since_ts - 2.0:
                    counts["fresh_map"] += 1
                else:
                    counts["stale_map"] += 1
            elif mabs:
                counts["mab_only"] += 1
            else:
                counts["no_artifacts"] += 1
        else:
            mtime = _pi_csv_mtime(folder)
            if mtime is None:
                counts["no_artifacts"] += 1
            elif mtime >= since_ts - 2.0:
                counts["fresh_csv"] += 1
            else:
                counts["stale_csv"] += 1
    return counts


def _print_wait_debug(
    *,
    emc: Path,
    pi_index: dict[int, Path],
    pi_count: int,
    batch_start_ts: float,
    mode: str,
    ready_count: int,
    expected_mhz: float | None,
    poll_index: int,
) -> None:
    print("\n  [wait debug] ─────────────────────────────────────────")
    print(f"  poll #{poll_index}  ready={ready_count}/{pi_count}")
    print(f"  emc dir     : {emc}")
    print(f"  batch_start : {_fmt_ts(batch_start_ts)}  (files must be newer to count)")
    print(f"  pi folders  : {len(pi_index):,} found in EMC (expect {pi_count:,})")
    if mode == "distribution" and expected_mhz is not None:
        print(f"  expected    : Z_*{int(round(expected_mhz))}*.map text heatmap files")

    summary = _summarize_wait_state(
        pi_index, pi_count, batch_start_ts, mode, expected_mhz,
    )
    if mode == "distribution":
        print(
            "  counts      : "
            f"fresh .map={summary['fresh_map']:,}  "
            f"stale .map={summary['stale_map']:,}  "
            f".mab-only={summary['mab_only']:,}  "
            f"no files={summary['no_artifacts']:,}  "
            f"missing PI folder={summary['missing_folder']:,}"
        )
        if summary["mab_only"]:
            print(
                "  ⚠ .mab-only folders are NOT treated as complete "
                "(pipeline expects text .map like heatmaps_10MHz)."
            )
            print("    Delete wrong PI-* under EMC and re-run this MHz step.")
        if summary["stale_map"] and not summary["fresh_map"]:
            print(
                "  ⚠ .map files exist but pre-date this batch_start "
                "(leftover from an earlier run)."
            )
    else:
        print(
            "  counts      : "
            f"fresh .csv={summary['fresh_csv']:,}  "
            f"stale .csv={summary['stale_csv']:,}  "
            f"no files={summary['no_artifacts']:,}  "
            f"missing PI folder={summary['missing_folder']:,}"
        )

    for sample in (1, 2, min(100, pi_count), pi_count):
        info = _inspect_pi_folder(
            pi_index.get(sample),
            since_ts=batch_start_ts,
            expected_mhz=expected_mhz if mode == "distribution" else None,
        )
        if not info.get("exists"):
            print(f"  PI-{sample:<5}: (missing)")
            continue
        fresh = info.get("map_fresh") if mode == "distribution" else None
        extra = f"  fresh={fresh}" if fresh is not None else ""
        print(
            f"  PI-{sample:<5}: maps={info['n_map']} mab={info['n_mab']} mpx={info['n_mpx']}"
            f"{extra}"
        )
        if info["map_names"]:
            print(f"           map files: {info['map_names']}")
        if info["map_mtime"] is not None:
            print(f"           map mtime: {_fmt_ts(info['map_mtime'])}")
        elif mode == "distribution" and info["n_mab"]:
            print(f"           power_gnd: {info['power_gnd_files']}")
    print("  [wait debug] ─────────────────────────────────────────\n")


def wait_for_pi_outputs(
    emc_output_dir: str,
    *,
    pi_count: int,
    batch_start_ts: float,
    mode: str,
    timeout_sec: int,
    poll_sec: int,
    expected_mhz: float | None = None,
) -> None:
    """Poll until PI-1..PI-N are ready (distribution=.map, impedance=.csv)."""
    emc = resolve_windows_path(emc_output_dir)
    need = list(range(1, pi_count + 1))
    label = "PI-Distribution (.map)" if mode == "distribution" else "PI-Spectrum (.csv)"

    print(f"\nWaiting for {len(need)} {label} output(s) in {emc} (timeout {timeout_sec}s)…")
    print(f"  batch_start={_fmt_ts(batch_start_ts)}")
    if mode == "distribution" and expected_mhz is not None:
        print(f"  expected MHz={expected_mhz:g}  (looking for Z_*{int(round(expected_mhz))}*.map)")

    deadline = time.time() + timeout_sec
    poll_index = 0
    while time.time() < deadline:
        poll_index += 1
        pi_index = _pi_folder_index(emc)
        if mode == "distribution":
            ready = [
                n for n in need
                if _pi_ready_distribution(pi_index, n, batch_start_ts, expected_mhz)
            ]
        else:
            ready = [n for n in need if _pi_ready_impedance(pi_index, n, batch_start_ts)]

        if len(ready) >= len(need):
            print(f"✓ All {len(need)} PI outputs ready.")
            return

        show_debug = poll_index == 1 or (
            len(ready) == 0 and poll_index in {2, 5, 10, 20}
        )
        if show_debug:
            _print_wait_debug(
                emc=emc,
                pi_index=pi_index,
                pi_count=pi_count,
                batch_start_ts=batch_start_ts,
                mode=mode,
                ready_count=len(ready),
                expected_mhz=expected_mhz,
                poll_index=poll_index,
            )

        print(f"  … {len(ready)}/{len(need)} ready — retry in {poll_sec}s")
        time.sleep(poll_sec)

    _print_wait_debug(
        emc=emc,
        pi_index=_pi_folder_index(emc),
        pi_count=pi_count,
        batch_start_ts=batch_start_ts,
        mode=mode,
        ready_count=0,
        expected_mhz=expected_mhz,
        poll_index=poll_index + 1,
    )
    raise SystemExit(
        f"Timeout waiting for PI outputs in {emc}\n"
        f"  Expected PI numbers: 1..{pi_count}\n"
        f"  Mode: {mode}\n"
        f"  batch_start: {_fmt_ts(batch_start_ts)}"
    )
