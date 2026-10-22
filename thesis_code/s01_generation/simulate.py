"""Simulate one campaign of the sampling plan with eCADSTAR: spectra and maps at per-row frequencies.

What: for each chunk of placements one spectrum PEB (one group per placement; the 1-600 MHz sweep
    comes out of each), and for each chunk of (placement, frequency) pairs one map PEB whose rows
    each carry their own frequency. Pairs are shuffled so every chunk mixes frequencies. Each batch
    gets a batch_manifest.csv (pi_number -> layout_row, freq_mhz, K), which processing reads.
    eCADSTAR sets how slot "C1" is simulated for a whole batch from the batch's first group, so a
    group that mounts C1 is swapped to the front of every PEB (c1_first).
    The "fill" campaign adds maps to placements that already have a spectrum, so it writes no
    spectrum PEB.
    With settings.ECADSTAR_RUN False (default) only the PEB files and commands.txt are written.
Usage: python -m thesis_code.s01_generation.simulate
    Writes OUTPUT_DIR/s01_generation/peb_<campaign>/{imp,dist}_batch_NNNN.peb (+ commands.txt) and,
    when simulating, OUTPUT_DIR/s01_generation/raw_<campaign>/{impedance,heatmaps}/batch_NNNN/PI-*
    with batch_manifest.csv and sim_progress.json (a rerun resumes from the saved batches).
Config keys: CAMPAIGN, N_LAYOUTS, CHUNK_GROUPS, PAIR_SHUFFLE_SEED, wait and retry limits.
"""
from __future__ import annotations

import csv
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from thesis_code.common import settings
from thesis_code.common.paths import out_dir
from thesis_code.s01_generation.ecadstar import (clear_ecadstar_lock, move_pi_outputs, resolve_windows_path,
                                                 run_ecadstar_batch_headless, stage_peb_for_ecadstar,
                                                 wait_for_pi_outputs, write_command)
from thesis_code.s01_generation.peb import (COMPONENTS, POWERBUS, build_distribution_peb, build_impedance_peb,
                                            load_multitype_csv)

# =============================================================================
# CONFIGURATION — edit these before running
# =============================================================================
CAMPAIGN = "new"                    # "new" (spectrum + 11 maps) or "fill" (maps only)
N_LAYOUTS: int | None = None        # None = every placement of the campaign
CHUNK_GROUPS = 2_000                # groups per PEB; a crash costs one chunk, not the run
PAIR_SHUFFLE_SEED = 45              # mixes frequencies within each chunk
ECADSTAR_WAIT_TIMEOUT_SEC = 172_800 # cap on engineer.exe itself
POST_EXIT_WAIT_SEC = 300            # file-system settling window after engineer.exe exits
ECADSTAR_WAIT_POLL_SEC = 30
CRASH_MARKER = "Severe problem occured"   # eCADSTAR writes this to log.txt on an internal crash
MAX_BATCH_ATTEMPTS = 3
RETRY_PAUSE_SEC = 30
# =============================================================================


def _ecad_log_text() -> str:
    try:
        return (resolve_windows_path(settings.EMC_DIR) / "log.txt").read_text(errors="ignore")
    except OSError:
        return ""


def _simulate(peb: Path, *, pi_count: int, mode: str) -> bool:
    """Run one batch, retrying an eCADSTAR crash. False = dry run (command written, nothing started)."""
    if not settings.ECADSTAR_RUN:
        write_command(peb, erf_path=settings.ERF_PATH, engineer_exe=settings.ENGINEER_EXE)
        return False
    for attempt in range(1, MAX_BATCH_ATTEMPTS + 1):
        clear_ecadstar_lock(settings.ERF_PATH)
        staged = stage_peb_for_ecadstar(peb, erf_path=settings.ERF_PATH)
        t0 = time.time()
        rc = run_ecadstar_batch_headless(staged, erf_path=settings.ERF_PATH, engineer_exe=settings.ENGINEER_EXE,
                                         timeout_sec=ECADSTAR_WAIT_TIMEOUT_SEC)
        try:
            if rc != 0:
                raise SystemExit(f"engineer.exe returned rc={rc}")
            wait_for_pi_outputs(settings.EMC_DIR, pi_count=pi_count, batch_start_ts=t0, mode=mode,
                                timeout_sec=POST_EXIT_WAIT_SEC, poll_sec=ECADSTAR_WAIT_POLL_SEC)
            return True
        except SystemExit as exc:
            why = f"eCADSTAR crash ('{CRASH_MARKER}')" if CRASH_MARKER in _ecad_log_text() else str(exc)
            if attempt >= MAX_BATCH_ATTEMPTS:
                raise SystemExit(f"{peb.name}: {why} -- still failing after {attempt} attempt(s). "
                                 "Progress is saved, so a rerun resumes from this batch.") from exc
            print(f"    attempt {attempt}/{MAX_BATCH_ATTEMPTS} failed: {why} -- retrying")
            time.sleep(RETRY_PAUSE_SEC)
    return True


def _write_batch_manifest(dest: Path, rows: list[dict]) -> None:
    with (dest / "batch_manifest.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["pi_number", "layout_row", "freq_mhz", "K"])
        w.writeheader()
        w.writerows(rows)


def c1_first(chunk_codes: np.ndarray) -> np.ndarray:
    """Group order for one PEB: unchanged, except the first group that mounts slot C1 is swapped
    to the front (only C1 depends on the first group; the other 51 slots do not)."""
    order = np.arange(len(chunk_codes))
    hit = np.flatnonzero(chunk_codes[:, 0] > 0)
    if hit.size and hit[0] != 0:
        order[[0, hit[0]]] = order[[hit[0], 0]]
    return order


def build_pairs(n_layouts: int, freqs: np.ndarray, seed: int) -> np.ndarray:
    """(P, 2) array of [layout_row, freq_mhz], shuffled; a NaN cell means "no map"."""
    rows = np.repeat(np.arange(n_layouts), freqs.shape[1])
    pairs = np.stack([rows.astype(np.float64), freqs[:n_layouts].ravel()], axis=1)
    np.random.default_rng(seed).shuffle(pairs, axis=0)
    return pairs[np.isfinite(pairs[:, 1])]


class _Campaign:
    def __init__(self, raw_root: Path, peb_dir: Path):
        self.raw_root, self.peb_dir = raw_root, peb_dir
        self.progress_path = raw_root / "sim_progress.json"
        self.prog = (json.loads(self.progress_path.read_text(encoding="utf-8")) if self.progress_path.is_file()
                     else {"impedance_batches_done": [], "heatmap_batches_done": []})

    def done(self, key: str, b: int) -> None:
        self.prog[key] = sorted(set(self.prog.get(key, [])) | {b})
        self.prog["updated_at"] = datetime.now(timezone.utc).isoformat()
        self.progress_path.write_text(json.dumps(self.prog, indent=2), encoding="utf-8")

    def run_impedance(self, codes: np.ndarray) -> None:
        done = set(self.prog.get("impedance_batches_done", []))
        n = len(codes)
        n_batches = (n + CHUNK_GROUPS - 1) // CHUNK_GROUPS
        print(f"\n[spectra] {n:,} placements -> {n_batches} batch(es) of <= {CHUNK_GROUPS:,}")
        for b in range(n_batches):
            if b in done:
                continue
            lo, hi = b * CHUNK_GROUPS, min((b + 1) * CHUNK_GROUPS, n)
            idx = lo + c1_first(codes[lo:hi])
            sl = codes[idx]
            peb = build_impedance_peb(sl, self.peb_dir / f"imp_batch_{b:04d}.peb",
                                      powerbus=POWERBUS, components=COMPONENTS)
            if _simulate(peb, pi_count=len(sl), mode="impedance"):
                dest = self.raw_root / "impedance" / f"batch_{b:04d}"
                move_pi_outputs(source_emc_dir=settings.EMC_DIR, dest_dir=dest, pi_count=len(sl))
                _write_batch_manifest(dest, [{"pi_number": i + 1, "layout_row": int(idx[i]), "freq_mhz": "",
                                              "K": int((codes[idx[i]] > 0).sum())} for i in range(len(sl))])
                self.done("impedance_batches_done", b)

    def run_heatmaps(self, codes: np.ndarray, pairs: np.ndarray) -> None:
        done = set(self.prog.get("heatmap_batches_done", []))
        n = len(pairs)
        n_batches = (n + CHUNK_GROUPS - 1) // CHUNK_GROUPS
        print(f"\n[maps] {n:,} (placement, frequency) pairs -> {n_batches} batch(es) of <= {CHUNK_GROUPS:,}")
        for b in range(n_batches):
            if b in done:
                continue
            lo, hi = b * CHUNK_GROUPS, min((b + 1) * CHUNK_GROUPS, n)
            sl = pairs[lo:hi]
            sl = sl[c1_first(codes[sl[:, 0].astype(int)])]
            rows, mhz = sl[:, 0].astype(int), sl[:, 1].astype(float)
            peb = build_distribution_peb(codes[rows], self.peb_dir / f"dist_batch_{b:04d}.peb",
                                         per_sample_mhz=mhz.tolist(), powerbus=POWERBUS)
            if _simulate(peb, pi_count=len(sl), mode="distribution"):
                dest = self.raw_root / "heatmaps" / f"batch_{b:04d}"
                move_pi_outputs(source_emc_dir=settings.EMC_DIR, dest_dir=dest, pi_count=len(sl))
                _write_batch_manifest(dest, [{"pi_number": i + 1, "layout_row": int(rows[i]),
                                              "freq_mhz": float(mhz[i]), "K": int((codes[rows[i]] > 0).sum())}
                                             for i in range(len(sl))])
                self.done("heatmap_batches_done", b)


def main() -> None:
    if CAMPAIGN not in ("new", "fill"):
        raise SystemExit(f"CAMPAIGN must be 'new' or 'fill', got {CAMPAIGN!r}")
    stem = settings.PLAN_DIR / f"combinations_balance30k_{CAMPAIGN}"
    codes = load_multitype_csv(stem.with_suffix(".csv")).astype(np.int64)
    freqs = np.loadtxt(f"{stem}_freqs.csv", delimiter=",", ndmin=2)
    if len(freqs) != len(codes):
        raise SystemExit(f"{len(freqs)} frequency rows != {len(codes)} placement rows")
    if N_LAYOUTS is not None:
        codes, freqs = codes[:N_LAYOUTS], freqs[:N_LAYOUTS]

    pairs = build_pairs(len(codes), freqs, PAIR_SHUFFLE_SEED)
    base = out_dir("s01_generation")
    camp = _Campaign(base / f"raw_{CAMPAIGN}", base / f"peb_{CAMPAIGN}")
    camp.peb_dir.mkdir(parents=True, exist_ok=True)
    camp.raw_root.mkdir(parents=True, exist_ok=True)
    print(f"campaign {CAMPAIGN}: {len(codes):,} placements, {len(pairs):,} maps"
          + ("" if CAMPAIGN == "fill" else f" + {len(codes):,} spectra"))
    if CAMPAIGN == "new":
        camp.run_impedance(codes)
    camp.run_heatmaps(codes, pairs)
    if settings.ECADSTAR_RUN:
        print(f"campaign complete: {camp.raw_root}")
    else:
        print(f"settings.ECADSTAR_RUN is False: PEB files and commands.txt written to {camp.peb_dir}; "
              "eCADSTAR was not started")


if __name__ == "__main__":
    main()
