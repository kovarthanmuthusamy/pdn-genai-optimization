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


def run_ecadstar_batch(peb_path: Path, *, erf_path: str, repo_root: Path, ahk_exe: str | None, skip_open_erf: bool) -> int:
    erf = resolve_windows_path(erf_path)
    if not erf.is_file():
        raise FileNotFoundError(f"ERF not found: {erf}")
    if not peb_path.is_file():
        raise FileNotFoundError(f"PEB not found: {peb_path}")

    ps_script = repo_root / "tools" / "ecadstar" / "run_ecadstar_piemi_batch.ps1"
    if not ps_script.is_file():
        raise FileNotFoundError(f"AHK runner not found: {ps_script}")

    cmd = [
        "powershell.exe",
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        windows_path_str(ps_script),
        "-ErfPath",
        windows_path_str(erf),
        "-PebPath",
        windows_path_str(peb_path),
    ]
    if ahk_exe:
        cmd.extend(["-AhkExe", str(ahk_exe)])
    if skip_open_erf:
        cmd.append("-SkipOpenErf")

    log_hint = Path(os.environ.get("TEMP", "/tmp")) / "ecadstar_piemi_batch.log"
    print(f"Running ECADSTAR: Load Batch → {windows_path_str(peb_path)}")
    print(f"  Log (Windows): {log_hint}")
    proc = subprocess.run(cmd, cwd=str(repo_root), env=os.environ.copy())
    return int(proc.returncode)


_BATCH_START_MARKERS = ("Perform batch step", "Batch file", "read successfully")


def _peb_name_in_log(text: str, peb_path: Path) -> bool:
    name = peb_path.name
    if name in text:
        return True
    win = windows_path_str(peb_path)
    return win in text or win.replace("\\", "/") in text


def _pi_log_shows_batch_started(log_path: Path, peb_path: Path, since_ts: float) -> bool:
    if not log_path.is_file():
        return False
    if log_path.stat().st_mtime < since_ts - 2.0:
        return False
    text = log_path.read_text(encoding="utf-8", errors="replace")
    if not _peb_name_in_log(text, peb_path):
        return False
    return any(marker in text for marker in _BATCH_START_MARKERS)


def _pi_folder_has_fresh_activity(folder: Path, since_ts: float) -> bool:
    if not folder.is_dir():
        return False
    for path in folder.rglob("*"):
        if path.is_file() and path.stat().st_mtime >= since_ts - 2.0:
            return True
    return False


def wait_for_batch_started(
    emc_output_dir: str,
    *,
    peb_path: Path,
    batch_start_ts: float,
    timeout_sec: int,
    poll_sec: int,
) -> None:
    """Block until PI-1 log confirms this PEB batch started (backup to AHK wait)."""
    emc = resolve_windows_path(emc_output_dir)
    pi1 = emc / "PI-1"
    pi_log = pi1 / "log.txt"
    peb_path = peb_path.resolve()

    print(
        f"\nConfirming ECADStar batch started for {peb_path.name} "
        f"(keep RDP on PI/EMI, timeout {timeout_sec}s)…"
    )
    deadline = time.time() + timeout_sec
    poll_index = 0
    while time.time() < deadline:
        poll_index += 1
        if _pi_log_shows_batch_started(pi_log, peb_path, batch_start_ts):
            print(f"  ✓ Batch started — {peb_path.name} seen in PI-1/log.txt")
            return
        if _pi_folder_has_fresh_activity(pi1, batch_start_ts):
            print(
                f"  ✓ Batch activity in PI-1/ (mtime after {_fmt_ts(batch_start_ts)})"
            )
            return
        if poll_index == 1 or poll_index in {2, 5, 10}:
            print(
                f"  … waiting for batch start ({poll_index}) — "
                f"PI-1/log exists={pi_log.is_file()}  "
                f"keep PI/EMI visible"
            )
            if pi_log.is_file():
                age = pi_log.stat().st_mtime
                print(
                    f"     PI-1/log.txt mtime={_fmt_ts(age)}  "
                    f"fresh={age >= batch_start_ts - 2.0}"
                )
        time.sleep(poll_sec)

    raise SystemExit(
        f"ECADStar batch did not start within {timeout_sec}s\n"
        f"  PEB: {windows_path_str(peb_path)}\n"
        f"  Check: {windows_path_str(pi_log)}\n"
        f"  AHK log: %TEMP%\\ecadstar_piemi_batch.log\n"
        f"  Keep RDP focused on PI/EMI during Load Batch."
    )


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
