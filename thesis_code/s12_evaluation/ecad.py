"""eCADSTAR simulations of the Chapter 4 checks: placements in, spectra and maps out.

What: a check hands over rows -- one dict per simulated placement with "codes" (52 type codes) and,
    for a map, "map_mhz" -- and gets the spectra (ohm, 231 points) and maps (ohm, 64 x 64) back.
    Each check keeps its simulations in a store folder of its own name. store() returns the saved
    store (settings.SIM_STORE_DIR, read only) when it exists, else a new one in
    OUTPUT_DIR/s12_evaluation; only a new store is written to.
    Runs are resumable (progress.json) and never share eCADSTAR: before every batch the run waits
    until no engineer.exe and no dataset-simulation job is running. Rows that mount the first slot
    are put at the front, so every batch that contains such a row starts with one (eCADSTAR reads
    that slot wrongly in a whole batch whose first group leaves it empty).
    With settings.ECADSTAR_RUN False the missing batches are written as PEB files plus commands.txt
    and the check stops (SystemExit) before its analysis.
Usage: import only -- store, writable, assign, simulate, read_spectra, read_maps, spectra_cached,
    maps_cached, n_missing.
"""
from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

import numpy as np

from thesis_code.common import settings
from thesis_code.common.board import INPUTS
from thesis_code.common.paths import out_dir
from thesis_code.s01_generation.ecadstar import (clear_ecadstar_lock, move_pi_outputs, parse_pi_number,
                                                 resolve_windows_path, run_ecadstar_batch_headless,
                                                 stage_peb_for_ecadstar, wait_for_pi_outputs, write_command)
from thesis_code.s01_generation.peb import COMPONENTS, POWERBUS, build_distribution_peb, build_impedance_peb
from thesis_code.s02_processing.read_outputs import create_heatmaps, read_impedance_file

BATCH_SIZE = 250
ECADSTAR_WAIT_TIMEOUT_SEC = 7200
ECADSTAR_WAIT_POLL_SEC = 30
ECADSTAR_BATCH_TRIES = 3              # engineer.exe can exit without running the batch


def store(name: str) -> Path:
    """The store of a check: the new one if this output folder has it, else the saved one, else a new one."""
    new = Path(settings.OUTPUT_DIR) / "s12_evaluation" / name
    saved = Path(settings.SIM_STORE_DIR) / name
    return saved if saved.is_dir() and not new.exists() else new


def writable(path: Path) -> Path:
    """``path`` if it lies in OUTPUT_DIR (created); a saved store is never written to."""
    path = Path(path)
    if not path.resolve().is_relative_to(Path(settings.OUTPUT_DIR).resolve()):
        raise SystemExit(f"{path} is a saved store (read only): copy it to "
                         f"{out_dir('s12_evaluation') / path.name} to add simulations")
    path.mkdir(parents=True, exist_ok=True)
    return path


def assign(rows: list[dict], batch_size: int = BATCH_SIZE) -> list[dict]:
    """Give every row its batch and its position in the batch ("batch", "pi"); rows keep their order in
    the list. Rows mounting the first slot are simulated first (see the module docstring)."""
    order = sorted(range(len(rows)), key=lambda i: (rows[i]["codes"][0] == 0, i))
    for pos, i in enumerate(order):
        rows[i]["batch"], rows[i]["pi"] = pos // batch_size, pos % batch_size + 1
    for b in {r["batch"] for r in rows}:
        sel = sorted((r for r in rows if r["batch"] == b), key=lambda r: r["pi"])
        assert sel[0]["codes"][0] > 0 or not any(r["codes"][0] > 0 for r in sel), "batch order rule broken"
    return rows


def busy() -> str | None:
    """Another eCADSTAR job (engineer.exe, or a dataset-simulation pipeline) is running."""
    try:
        tl = subprocess.run(["tasklist.exe"], capture_output=True, text=True, timeout=60).stdout
    except Exception as e:                     # cannot tell -> treat as busy
        return f"tasklist failed: {e}"
    if "engineer.exe" in tl.lower():
        return "engineer.exe running"
    ps = subprocess.run(["ps", "-eo", "pid,args"], capture_output=True, text=True).stdout
    for line in ps.splitlines():
        if "dataset_sim" in line or "sim_pipeline" in line or "queue_" in line or "s01_generation.simulate" in line:
            return f"pipeline running: {line.strip()[:120]}"
    return None


def wait_until_free(poll_sec: int = 60) -> None:
    while (why := busy()) is not None:
        print(f"  eCADSTAR busy ({why}); waiting {poll_sec}s", flush=True)
        time.sleep(poll_sec)


def _run(peb: Path, n: int, mode: str, dest: Path) -> float:
    emc = resolve_windows_path(settings.EMC_DIR)
    for attempt in range(1, ECADSTAR_BATCH_TRIES + 1):
        wait_until_free()
        clear_ecadstar_lock(settings.ERF_PATH)
        staged = stage_peb_for_ecadstar(peb, erf_path=settings.ERF_PATH)
        t0 = time.time()
        rc = run_ecadstar_batch_headless(staged, erf_path=settings.ERF_PATH, engineer_exe=settings.ENGINEER_EXE,
                                         timeout_sec=ECADSTAR_WAIT_TIMEOUT_SEC)
        if rc != 0:
            raise SystemExit(f"eCADSTAR failed rc={rc} for {peb.name}")
        # engineer.exe has exited. Not one result folder of this batch then means the batch never ran:
        # start it again (after the same check that eCADSTAR is free) instead of waiting.
        if attempt == ECADSTAR_BATCH_TRIES or any(
                q.is_dir() and parse_pi_number(q.name) is not None and q.stat().st_mtime >= t0 - 5 for q in emc.iterdir()):
            break
        print(f"  engineer.exe exited without any output for {peb.name}; starting the batch again "
              f"(attempt {attempt + 1} of {ECADSTAR_BATCH_TRIES})", flush=True)
    wait_for_pi_outputs(settings.EMC_DIR, pi_count=n, batch_start_ts=t0, mode=mode,
                        timeout_sec=ECADSTAR_WAIT_TIMEOUT_SEC, poll_sec=ECADSTAR_WAIT_POLL_SEC)
    move_pi_outputs(source_emc_dir=settings.EMC_DIR, dest_dir=dest, pi_count=n)
    return time.time() - t0


def simulate(out: Path, spectra: list[dict], maps: list[dict]) -> dict:
    """Simulate the rows (already given "batch"/"pi" by assign()); returns the progress record with the
    seconds per batch. Finished batches are skipped; with ECADSTAR_RUN False the missing ones are
    written as PEB files and the call stops."""
    prog_p = out / "progress.json"
    prog = json.loads(prog_p.read_text()) if prog_p.is_file() else {"done": [], "seconds": {}, "groups": {}}
    todo = [(kind, rows, b) for kind, rows in (("spectra", spectra), ("maps", maps)) if rows
            for b in range(max(r["batch"] for r in rows) + 1) if f"{kind}{b}" not in prog["done"]]
    if not todo:
        return prog
    peb_dir = writable(out) / "peb"
    (peb_dir / "commands.txt").unlink(missing_ok=True)
    for kind, rows, b in todo:
        tag = f"{kind}{b}"
        sel = sorted((r for r in rows if r["batch"] == b), key=lambda r: r["pi"])
        codes = np.array([r["codes"] for r in sel], dtype=np.int64)
        peb = peb_dir / f"{tag}.peb"
        if kind == "spectra":
            build_impedance_peb(codes, peb, powerbus=POWERBUS, components=COMPONENTS)
        else:
            build_distribution_peb(codes, peb, powerbus=POWERBUS, per_sample_mhz=[r["map_mhz"] for r in sel])
        if not settings.ECADSTAR_RUN:
            write_command(peb, erf_path=settings.ERF_PATH, engineer_exe=settings.ENGINEER_EXE)
            continue
        print(f"eCADSTAR {out.name} {tag}: {len(sel)} groups ...", flush=True)
        dt = _run(peb, len(sel), "impedance" if kind == "spectra" else "distribution", out / kind / f"batch{b}")
        print(f"  done in {dt:.0f}s ({dt / len(sel):.2f}s per group)", flush=True)
        prog["done"].append(tag); prog["seconds"][tag] = dt; prog["groups"][tag] = len(sel)
        prog_p.write_text(json.dumps(prog, indent=1))
    if not settings.ECADSTAR_RUN:
        raise SystemExit(f"settings.ECADSTAR_RUN is False: {len(todo)} PEB file(s) and commands.txt written to "
                         f"{peb_dir}; simulate them on the eCADSTAR host, then run this check again")
    return prog


def read_spectra(out: Path, rows: list[dict]) -> np.ndarray:
    Z = np.zeros((len(rows), 231))
    for i, r in enumerate(rows):
        f = sorted((out / "spectra" / f"batch{r['batch']}" / f"PI-{r['pi']}").rglob("*PIPinZ*.csv"))
        Z[i] = np.asarray(read_impedance_file(str(f[0])), dtype=np.float64)
    return Z


def read_maps(out: Path, rows: list[dict]) -> np.ndarray:
    mask = np.load(INPUTS / "binary_mask.npy").astype(np.float32)
    H = np.zeros((len(rows), 64, 64), dtype=np.float32)
    for i, r in enumerate(rows):
        f = sorted((out / "maps" / f"batch{r['batch']}" / f"PI-{r['pi']}").rglob("Z_*MHz.map"))
        H[i] = np.asarray(create_heatmaps(str(f[0]), mask_board=mask))[0]
    return H


# ---- simulations with a store: a placement is simulated once, whoever asks for it ------------------
def _cached(store: Path, kind: str, codes: np.ndarray, mhz: np.ndarray | None) -> np.ndarray:
    codes = np.asarray(codes, dtype=np.int64).reshape(-1, 52)
    key = (lambda i: codes[i].tobytes()) if mhz is None else (lambda i: codes[i].tobytes() + f"{float(mhz[i]):.1f}".encode())
    f = store / f"{kind}.npz"
    if f.is_file():
        d = np.load(f)
        s_codes, s_val = d["codes"], d["values"]
        s_mhz = d["mhz"] if "mhz" in d.files else None
    else:
        s_codes = np.zeros((0, 52), np.int64)
        s_val = np.zeros((0, 231)) if kind == "spectra" else np.zeros((0, 64, 64), np.float32)
        s_mhz = None if mhz is None else np.zeros(0)
    have = {(s_codes[i].tobytes() if s_mhz is None else s_codes[i].tobytes() + f"{float(s_mhz[i]):.1f}".encode()): i
            for i in range(len(s_codes))}
    missing, seen = [], set()
    for i in range(len(codes)):
        k = key(i)
        if k not in have and k not in seen:
            seen.add(k); missing.append(i)
    if missing:
        writable(store)
        rows = [{"codes": codes[i].tolist(), **({} if mhz is None else {"map_mhz": float(mhz[i])})} for i in missing]
        runs = sorted(store.glob(f"{kind}_run*"))
        run = None
        if runs and not (runs[-1] / "merged").is_file():            # an interrupted run of the same request: resume it
            old = json.loads((runs[-1] / "manifest.json").read_text())["rows"]
            if [(r["codes"], r.get("map_mhz")) for r in old] == [(r["codes"], r.get("map_mhz")) for r in rows]:
                run, rows = runs[-1], old
        if run is None:
            run = store / f"{kind}_run{len(runs):03d}"
            run.mkdir(parents=True, exist_ok=True)
            (run / "manifest.json").write_text(json.dumps({"rows": assign(rows)}))
        if kind == "spectra":
            simulate(run, rows, []); new = read_spectra(run, rows)
        else:
            simulate(run, [], rows); new = read_maps(run, rows)
        s_codes = np.concatenate([s_codes, codes[missing]]); s_val = np.concatenate([s_val, new])
        extra = {}
        if mhz is not None:
            s_mhz = np.concatenate([s_mhz, np.asarray(mhz, dtype=np.float64)[missing]]); extra["mhz"] = s_mhz
        tmp = store / f"{kind}.tmp.npz"
        np.savez_compressed(tmp, codes=s_codes, values=s_val, **extra)
        tmp.replace(f)
        (run / "merged").write_text("")
        for j, i in enumerate(missing):
            have[key(i)] = len(s_codes) - len(missing) + j
    return s_val[[have[key(i)] for i in range(len(codes))]]


def spectra_cached(store: Path, codes: np.ndarray) -> np.ndarray:
    """Simulated spectra (N, 231) of the placements ``codes`` (N, 52). Placements simulated before are read
    from ``store``/spectra.npz; the others are simulated in a new run folder of the store and added."""
    return _cached(store, "spectra", codes, None)


def maps_cached(store: Path, codes: np.ndarray, mhz: np.ndarray) -> np.ndarray:
    """Simulated maps (N, 64, 64): placement n at frequency mhz[n]; same store logic as spectra_cached."""
    return _cached(store, "maps", codes, np.asarray(mhz, dtype=np.float64))


def n_missing(store: Path, codes: np.ndarray) -> int:
    """How many of the placements ``codes`` have no simulated spectrum in the store yet."""
    f = store / "spectra.npz"
    have = {x.tobytes() for x in np.load(f)["codes"]} if f.is_file() else set()
    return len({x.tobytes() for x in np.asarray(codes, dtype=np.int64).reshape(-1, 52)} - have)
