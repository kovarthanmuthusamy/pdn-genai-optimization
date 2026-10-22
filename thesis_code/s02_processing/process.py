"""Turn the eCADSTAR outputs of one campaign into arrays: one map file per sample, one folder per placement.

What: every batch folder holds a batch_manifest.csv (pi_number -> layout_row, freq_mhz). Each map is
    paired with its placement, its frequency and the placement's spectrum. The frequency in the map
    file name (Z_0168.500MHz.map) is the one eCADSTAR actually solved; it is checked against the
    manifest and a mismatch stops the run, since it would attach a map to the wrong placement.
    The "fill" campaign simulated no spectra; its spectra come from the store in IMPEDANCE_FROM.
    Output (out_root):
        heatmap/sample_N.npy      (2, 64, 64): map on the board and the board mask
        PI_freq/sample_N.npy      the sample's frequency in Hz
        layouts/<design_id>/      imp.npy (231, 1), type.npy (52,) int8, occ.npy (52,) float32
        manifest.csv              sample_name, design_id, freq_label, freq_mhz, freq_hz, ..., K
        dataset_meta.json         counts, K counts, frequency range, failures
Usage: python -m thesis_code.s02_processing.process
Config keys: CAMPAIGN, RAW_ROOT, IMPEDANCE_FROM, N_WORKERS, FREQ_TOLERANCE_MHZ.
"""
from __future__ import annotations

import csv
import json
import shutil
from collections import Counter
from datetime import datetime, timezone
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from thesis_code.common import settings
from thesis_code.common.board import INPUTS
from thesis_code.common.paths import need, out_dir
from thesis_code.s01_generation.peb import load_multitype_csv
from thesis_code.s02_processing.read_outputs import (EXPECTED_IMP_LENGTH, create_heatmaps, load_mask_board,
                                                     read_impedance_file)

# =============================================================================
# CONFIGURATION — edit these before running
# =============================================================================
CAMPAIGN = "new"                                              # "new" or "fill"
RAW_ROOT = settings.RAW_SIM_ROOT / f"Raw_balance30k_{CAMPAIGN}"   # eCADSTAR outputs of the campaign
IMPEDANCE_FROM = [settings.REPO / "datasets/pdn_k1-10_balanced_v1/test_pool/raw"]   # spectra of "fill"
N_WORKERS = 8
FREQ_TOLERANCE_MHZ = 0.05           # map file name vs manifest; larger = a real mismatch
# =============================================================================

_MASK = None
_LIMIT = None


def _init_worker(frame_path: str) -> None:
    global _MASK, _LIMIT
    _MASK = load_mask_board(Path(frame_path))
    # One BLAS thread per worker: a multi-threaded solve changes the interpolated maps in the
    # last float digits (about 5e-10 ohm), so the dataset is reproducible only with one thread.
    _LIMIT = threadpool_limits(1)


def _map_freq_mhz(path: Path) -> float:
    """Z_0168.500MHz.map -> 168.5."""
    return float(path.name.split("Z_")[1].split("MHz")[0])


def read_batch_manifests(phase_dir: Path) -> list[dict]:
    rows: list[dict] = []
    for batch in sorted(p for p in phase_dir.iterdir() if p.is_dir()):
        mf = batch / "batch_manifest.csv"
        if not mf.is_file():
            print(f"  warning: {batch.name}/ has no batch_manifest.csv, skipped")
            continue
        with mf.open() as f:
            for r in csv.DictReader(f):
                r["batch_dir"] = batch
                rows.append(r)
    return rows


def build_tasks(raw_root: Path, codes: np.ndarray, design_ids: list[str], impedance_from: list[Path],
                n_layouts: int | None = None) -> list[tuple[int, dict]]:
    """(sample index, task) for every map. With ``n_layouts`` only placements with row < n_layouts
    are resolved; the others keep their place in the numbering (a finished campaign has no gaps)."""
    keep = (lambda row: True) if n_layouts is None else (lambda row: row < n_layouts)
    imp_by_row: dict[int, Path] = {}
    imp_dir = raw_root / "impedance"
    if imp_dir.is_dir():
        for r in read_batch_manifests(imp_dir):
            row = int(r["layout_row"])
            if not keep(row):
                continue
            pi_dir = r["batch_dir"] / f"PI-{int(r['pi_number'])}"
            csvs = sorted(pi_dir.rglob("*PIPinZ*.csv")) if pi_dir.is_dir() else []
            if csvs:
                imp_by_row[row] = csvs[0]

    tasks: list[tuple[int, dict]] = []
    idx = 0
    missing_map = missing_imp = 0
    for r in read_batch_manifests(raw_root / "heatmaps"):
        row = int(r["layout_row"])
        if not keep(row):
            idx += 1
            continue
        want = float(r["freq_mhz"])
        pi_dir = r["batch_dir"] / f"PI-{int(r['pi_number'])}"
        maps = sorted(pi_dir.rglob("Z_*MHz.map")) if pi_dir.is_dir() else []
        if not maps:
            missing_map += 1
            continue
        got = _map_freq_mhz(maps[0])
        if abs(got - want) > FREQ_TOLERANCE_MHZ:
            raise SystemExit(f"FREQUENCY MISMATCH in {r['batch_dir'].name}/PI-{r['pi_number']}: manifest says "
                             f"{want} MHz, simulated file is {maps[0].name} ({got} MHz).")
        if row not in imp_by_row:
            for store in impedance_from:
                npy = store / "layouts" / design_ids[row] / "imp.npy"
                if npy.is_file():
                    imp_by_row[row] = npy
                    break
        if row not in imp_by_row:
            missing_imp += 1
            continue
        tasks.append((idx, {"layout_row": row, "freq_mhz": got, "heatmap_path": str(maps[0]),
                            "impedance_path": str(imp_by_row[row]), "source_folder": r["batch_dir"].name,
                            "pi_number": int(r["pi_number"]), "type_codes": codes[row],
                            "design_id": design_ids[row]}))
        idx += 1
    if missing_map:
        print(f"  warning: {missing_map:,} map rows had no .map file")
    if missing_imp:
        print(f"  warning: {missing_imp:,} map rows had no spectrum for their placement")
    return tasks


def _process(args: tuple[int, dict, Path, Path, Path]) -> tuple[int, bool, str | None, dict | None]:
    idx, t, hm_dir, layouts_dir, pifreq_dir = args
    try:
        name = f"sample_{idx + 1}.npy"
        np.save(hm_dir / name, create_heatmaps(t["heatmap_path"], mask_board=_MASK))
        np.save(pifreq_dir / name, np.array(t["freq_mhz"] * 1e6, dtype=np.float64))
        ldir = layouts_dir / t["design_id"]
        if not ldir.is_dir():                      # one folder per placement, not per sample
            ldir.mkdir(parents=True, exist_ok=True)
            ip = str(t["impedance_path"])
            imp = (np.load(ip).astype(np.float64).reshape(-1) if ip.endswith(".npy") else
                   np.asarray(read_impedance_file(ip), dtype=np.float64))
            if imp.size != EXPECTED_IMP_LENGTH:
                return idx, False, f"impedance length {imp.size} != {EXPECTED_IMP_LENGTH}", None
            codes = np.asarray(t["type_codes"], dtype=np.int8)
            np.save(ldir / "imp.npy", imp.reshape(-1, 1))
            np.save(ldir / "type.npy", codes)
            np.save(ldir / "occ.npy", codes.astype(np.float32))
        return idx, True, None, {
            "sample_name": name, "design_id": t["design_id"], "freq_label": f"{t['freq_mhz']:g}MHz",
            "freq_mhz": t["freq_mhz"], "freq_hz": t["freq_mhz"] * 1e6, "source_folder": t["source_folder"],
            "pi_number": t["pi_number"], "decap_index": t["layout_row"], "n_decap_types": 3,
            "K": int((np.asarray(t["type_codes"]) > 0).sum())}
    except Exception as e:                          # noqa: BLE001 -- one bad row must not stop the run
        return idx, False, str(e)[:200], None


def process(raw_root: Path, codes_csv: Path, out_root: Path, n_layouts: int | None = None,
            impedance_from: list[Path] = ()) -> Path:
    """Process the campaign in ``raw_root`` planned in ``codes_csv`` (ids in <stem>_design_ids.csv)."""
    codes = load_multitype_csv(codes_csv).astype(np.int64)
    ids_csv = codes_csv.with_name(codes_csv.stem + "_design_ids.csv")
    design_ids = [ln.strip() for ln in ids_csv.read_text().splitlines() if ln.strip()]
    if len(design_ids) != len(codes) or len(set(design_ids)) != len(design_ids):
        raise SystemExit(f"{ids_csv}: need {len(codes)} unique ids, got {len(design_ids)}")
    need(raw_root, "RAW_SIM_ROOT")
    tasks = build_tasks(raw_root, codes, design_ids, list(impedance_from), n_layouts)
    if not tasks:
        raise SystemExit("no samples found")
    print(f"  {len(tasks):,} samples from {raw_root}")

    if out_root.is_dir():
        shutil.rmtree(out_root)
    hm_dir, layouts_dir, pifreq_dir = out_root / "heatmap", out_root / "layouts", out_root / "PI_freq"
    for d in (hm_dir, layouts_dir, pifreq_dir):
        d.mkdir(parents=True, exist_ok=True)

    args = [(i, t, hm_dir, layouts_dir, pifreq_dir) for i, t in tasks]
    rows: list[dict] = []
    failures: list[str] = []
    with Pool(N_WORKERS, initializer=_init_worker, initargs=(str(INPUTS / "binary_mask.npy"),)) as pool:
        for idx, ok, err, row in pool.imap_unordered(_process, args, chunksize=32):
            if ok:
                rows.append(row)
            else:
                failures.append(f"sample {idx}: {err}")

    rows.sort(key=lambda r: int(r["sample_name"].split("_")[1].split(".")[0]))
    with (out_root / "manifest.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    freqs = np.array([r["freq_mhz"] for r in rows])
    meta = {"schema_version": 1, "stage": "raw", "source_script": "thesis_code/s02_processing/process.py",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(), "raw_root": str(raw_root),
            "n_decap_types": 3, "occupancy_schema": "type_codes_0_to_T", "frequency_scheme": "per_layout_random",
            "counts": {"manifest_rows": len(rows), "unique_layouts": len({r["design_id"] for r in rows}),
                       "distinct_frequencies": int(len(np.unique(freqs)))},
            "k_counts": {str(k): v for k, v in sorted(Counter(r["K"] for r in rows).items())},
            "freq_mhz_range": [float(freqs.min()), float(freqs.max())],
            "failures": len(failures)}
    (out_root / "dataset_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"  samples {len(rows):,}, placements {meta['counts']['unique_layouts']:,}, failures {len(failures)}")
    for f in failures[:5]:
        print(f"    FAIL {f}")
    return out_root


def main() -> None:
    process(RAW_ROOT, settings.PLAN_DIR / f"combinations_balance30k_{CAMPAIGN}.csv",
            out_dir("s02_processing") / f"balance30k_{CAMPAIGN}",
            impedance_from=IMPEDANCE_FROM if CAMPAIGN == "fill" else [])


if __name__ == "__main__":
    main()
