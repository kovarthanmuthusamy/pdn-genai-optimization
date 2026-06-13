"""
Build multifreq training data for exp038_true_multi.

One row per (layout, PI_freq):
  - layouts/{design_id}/imp.npy, occ.npy — once per layout
  - heatmap/sample_N.npy, PI_freq/sample_N.npy — per anchor MHz
  - manifest.csv — maps sample_N → design_id, freq_mhz, pi_number, decap_index

After build: python scripts/Normalization.py
Legacy Imp/Occ_map cleanup: python Data_Creation/verify_layout_store.py --prune
"""
from __future__ import annotations

import csv
import os
import random
import re
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from heatmap import create_Heatmaps, load_mask_board
from impedance import read_impedance_file, EXPECTED_IMP_LENGTH
from csv_to_occupancy import create_occupancy_vector

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src_vae.others.multifreq_anchors import (  # noqa: E402
    anchors_to_freq_hz,
    anchors_to_freq_labels,
    load_anchors_mhz,
    mhz_to_heatmap_subdir,
)

# ============================================================
# FREQUENCY CONFIGURATION — edit configs/multifreq_anchors.yaml
# ============================================================

def _refresh_freq_tables(only_mhz: list[float] | None = None) -> None:
    global FREQ_LABELS, HEATMAP_SUBDIR_NAMES, FREQ_HZ
    mhz_list = [float(m) for m in (only_mhz if only_mhz is not None else load_anchors_mhz())]
    FREQ_LABELS = anchors_to_freq_labels(mhz_list)
    HEATMAP_SUBDIR_NAMES = [mhz_to_heatmap_subdir(m) for m in mhz_list]
    FREQ_HZ = anchors_to_freq_hz(mhz_list)


_refresh_freq_tables()

# ============================================================
# PATHS
# ============================================================
SCRIPT_DIR = Path(__file__).parent.absolute()
FRAME_PATH = SCRIPT_DIR.parent / "configs" / "binary_mask.npy"
TRAIN_DATA_ROOT = SCRIPT_DIR.parent / "datasets" / "data_multifreq"

_DATA_ROOT_CANDIDATES = [
    os.getenv("DATA_ROOT"),
    r"C:\Users\muthusamy\Desktop\Raw",
    "/mnt/c/Users/muthusamy/Desktop/Raw",
]
DEFAULT_DATA_ROOT = next((Path(p) for p in _DATA_ROOT_CANDIDATES if p and Path(p).exists()), None)

_PI_DIR_RE = re.compile(r"PI[-_]?([0-9]+)", flags=re.IGNORECASE)
MAX_SAMPLES: int | None = None
OVERWRITE_OUTPUT = os.getenv("DATA_OVERWRITE", "1").strip().lower() not in ("0", "false", "no")

_WORKER_MASK_BOARD = None


def _find_subdir(base: Path, candidates: list[str]) -> Path | None:
    for name in candidates:
        d = base / name
        if d.is_dir():
            return d
    name_map = {c.name.lower(): c for c in base.iterdir() if c.is_dir()}
    for name in candidates:
        d = name_map.get(name.lower())
        if d is not None:
            return d
    return None


def _list_pi_dirs(base_dir: Path) -> dict[int, Path]:
    pi_map: dict[int, Path] = {}
    try:
        with os.scandir(base_dir) as it:
            for entry in it:
                if not entry.is_dir():
                    continue
                m = _PI_DIR_RE.search(entry.name)
                if not m:
                    continue
                try:
                    pi_map[int(m.group(1))] = Path(entry.path)
                except ValueError:
                    continue
    except FileNotFoundError:
        return {}
    return pi_map


def _heatmap_map_basename(freq_label: str) -> str:
    mhz = int(FREQ_HZ[freq_label] / 1e6)
    return f"Z_{mhz:04d}.000MHz.map"


def resolve_heatmap_map(pi_dir: Path, freq_label: str) -> Path | None:
    search = pi_dir / "Power_GND"
    if not search.is_dir():
        search = pi_dir
    expected = search / _heatmap_map_basename(freq_label)
    if expected.is_file():
        return expected
    maps = sorted(search.glob("*.map"), key=lambda p: p.name.lower())
    if not maps:
        maps = sorted(pi_dir.rglob("*.map"), key=lambda p: str(p).lower())
    return maps[0] if maps else None


def resolve_impedance_ic1(pi_dir: Path) -> Path | None:
    search = pi_dir / "Power_GND"
    if not search.is_dir():
        search = pi_dir
    candidates = sorted(
        (p for p in search.glob("*.csv") if p.is_file() and "ic1" in p.name.lower()),
        key=lambda p: p.name.lower(),
    )
    if not candidates:
        candidates = sorted(
            (p for p in pi_dir.rglob("*.csv") if p.is_file() and "ic1" in p.name.lower()),
            key=lambda p: str(p).lower(),
        )
    return candidates[0] if candidates else None


def discover_dataset_roots(root: Path) -> list[Path]:
    has_imp = _find_subdir(root, ["imp", "Imp", "IMP"]) is not None
    has_hm = any((root / name).is_dir() for name in HEATMAP_SUBDIR_NAMES)
    if has_imp and has_hm:
        return [root]
    nested = sorted(
        [d for d in root.iterdir() if d.is_dir() and d.name.lower().endswith("_true")],
        key=lambda p: (int(m.group(1)) if (m := re.match(r"^(\d+)", p.name)) else 10**9, p.name.lower()),
    )
    return nested if nested else [root]


def load_decap_vectors(decap_csv: Path) -> tuple[np.ndarray, bool]:
    path = Path(decap_csv)
    if path.name.lower() == "all_combinations.csv":
        df = pd.read_csv(path, header=None)
        headerless = True
    else:
        first_line = path.read_text(encoding="utf-8", errors="replace").splitlines()[0]
        try:
            [float(x) for x in first_line.split(",")]
            df = pd.read_csv(path, header=None)
            headerless = True
        except ValueError:
            df = pd.read_csv(path)
            headerless = False
    vectors = df.select_dtypes(include=[np.number]).values.astype(np.float32)
    return vectors, headerless


def find_decap_csv(folder: Path) -> Path | None:
    preferred = folder / "decap_combinations" / "all_combinations.csv"
    if preferred.is_file():
        return preferred
    for d in (folder / "decap_combinations", folder / "decap", folder / "Decap", folder):
        if not d.is_dir():
            continue
        candidates = [
            f
            for f in d.iterdir()
            if f.is_file()
            and f.suffix.lower() == ".csv"
            and "ic1" not in f.name.lower()
            and "imp" not in f.name.lower()
        ]
        if candidates:
            return sorted(
                candidates,
                key=lambda p: (
                    0 if "all_combinations" in p.name.lower() else 1,
                    0 if "comb" in p.name.lower() else 1,
                    p.name.lower(),
                ),
            )[0]
    return None


def prepare_output_dir(output_root: Path, *, overwrite: bool = OVERWRITE_OUTPUT, verbose: bool = True) -> None:
    output_root = Path(output_root)
    if output_root.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output already exists: {output_root}\n"
                "Delete it manually or run with DATA_OVERWRITE=1 (default)."
            )
        if verbose:
            print(f"Removing existing folder: {output_root}")
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)


def design_key(source_tag: str, pi_number: int, decap_index: int) -> str:
    return f"{source_tag}_pi{pi_number}_d{decap_index}"


def iter_dataset_samples(data_root=None, verbose: bool = True):
    """Yield one record per (layout, frequency) from raw Raw tree."""
    root = Path(data_root) if data_root else DEFAULT_DATA_ROOT
    if not root or not root.exists():
        if verbose:
            print(f"Error: Dataset root not found: {root}")
        return

    dataset_roots = discover_dataset_roots(root)
    if verbose:
        layout = "flat Raw (heatmap_*MHz + imp)" if len(dataset_roots) == 1 and dataset_roots[0] == root else "nested"
        print(f"  Layout: {layout}")
        print(f"  Dataset root(s): {[str(d) for d in dataset_roots]}")

    for folder in dataset_roots:
        source_tag = folder.name if folder.name.lower() not in ("raw", ".") else "layout"
        imp_dir = _find_subdir(folder, ["imp", "Imp", "impedance", "Impedance", "IMP"])
        decap_csv = find_decap_csv(folder)

        if decap_csv is None:
            print(f"  Warning: No decap CSV in {folder}, skipping.")
            continue
        if imp_dir is None or not imp_dir.exists():
            print(f"  Warning: imp/ not found in {folder}, skipping.")
            continue

        try:
            decap_vectors, decap_headerless = load_decap_vectors(decap_csv)
        except Exception as e:
            print(f"  Error reading decap CSV {decap_csv}: {e}, skipping {folder.name}.")
            continue

        imp_pi_dirs = _list_pi_dirs(imp_dir)
        if not imp_pi_dirs:
            print(f"  Warning: No PI-* folders under {imp_dir}, skipping.")
            continue

        heatmap_pi_dirs: list[dict[int, Path]] = [dict() for _ in FREQ_LABELS]
        available_freq_indices: list[int] = []

        for freq_idx, hm_subdir_name in enumerate(HEATMAP_SUBDIR_NAMES):
            heatmap_dir = folder / hm_subdir_name
            if not heatmap_dir.is_dir():
                if verbose:
                    print(f"  Warning: {hm_subdir_name}/ missing ({FREQ_LABELS[freq_idx]})")
                continue
            heatmap_pi_dirs[freq_idx] = _list_pi_dirs(heatmap_dir)
            available_freq_indices.append(freq_idx)

        if not available_freq_indices:
            print(f"  Warning: No heatmap_*MHz folders in {folder}, skipping.")
            continue

        pi_in_any_freq = set().union(*(heatmap_pi_dirs[fi].keys() for fi in available_freq_indices))
        common_pi_numbers = sorted(pi_in_any_freq & set(imp_pi_dirs.keys()))
        num_layouts = min(len(common_pi_numbers), len(decap_vectors))

        impedance_dict: dict[int, Path] = {}
        heatmap_dicts: list[dict[int, Path]] = [dict() for _ in FREQ_LABELS]

        for pi_num in common_pi_numbers[:num_layouts]:
            imp_path = resolve_impedance_ic1(imp_pi_dirs[pi_num])
            if imp_path:
                impedance_dict[pi_num] = imp_path

            for freq_idx in available_freq_indices:
                pi_dir = heatmap_pi_dirs[freq_idx].get(pi_num)
                if pi_dir is None:
                    continue
                hm_path = resolve_heatmap_map(pi_dir, FREQ_LABELS[freq_idx])
                if hm_path:
                    heatmap_dicts[freq_idx][pi_num] = hm_path

        common_pi_numbers = [p for p in common_pi_numbers[:num_layouts] if p in impedance_dict]

        if verbose:
            print(f"\n  [{folder}]")
            hdr = "no header row" if decap_headerless else "with header row"
            print(f"    Decap CSV     : {decap_csv.name} ({len(decap_vectors)} rows, {hdr})")
            print(f"    PI in imp/    : {len(imp_pi_dirs)}")
            print(f"    Frequencies   : {[FREQ_LABELS[fi] for fi in available_freq_indices]}")
            print(f"    Layouts (PI×decap): {len(common_pi_numbers)}")
            print(f"    Rows if all MHz: {len(common_pi_numbers) * len(available_freq_indices)}")

        for i, pi_num in enumerate(common_pi_numbers):
            decap_index = i
            dk = design_key(source_tag, pi_num, decap_index)
            valid_freq_indices = [fi for fi in available_freq_indices if pi_num in heatmap_dicts[fi]]
            if not valid_freq_indices:
                continue

            for freq_idx in valid_freq_indices:
                freq_label = FREQ_LABELS[freq_idx]
                yield {
                    "design_id": dk,
                    "source_folder": source_tag,
                    "pi_number": pi_num,
                    "decap_index": decap_index,
                    "freq_label": freq_label,
                    "freq_mhz": FREQ_HZ[freq_label] / 1e6,
                    "heatmap_path": heatmap_dicts[freq_idx][pi_num],
                    "impedance_path": impedance_dict[pi_num],
                    "decap_vector": decap_vectors[i],
                }


def select_samples_balanced(all_samples: list[dict], max_samples: int, seed: int = 42) -> list[dict]:
    by_freq: dict[str, list[dict]] = defaultdict(list)
    for s in all_samples:
        by_freq[s["freq_label"]].append(s)
    for fl in by_freq:
        random.Random(seed).shuffle(by_freq[fl])

    selected: list[dict] = []
    keys = [fl for fl in FREQ_LABELS if fl in by_freq]
    idx = 0
    while len(selected) < max_samples:
        progressed = False
        for fl in keys:
            if idx < len(by_freq[fl]):
                selected.append(by_freq[fl][idx])
                progressed = True
                if len(selected) >= max_samples:
                    break
        if not progressed:
            break
        idx += 1
    return selected


def _init_worker(frame_path: str | Path) -> None:
    global _WORKER_MASK_BOARD
    _WORKER_MASK_BOARD = load_mask_board(frame_path)


def _write_layout_once(layout_sub: Path, sample: dict) -> str | None:
    """Write imp/occ under layouts/{design_id}/ if not already present. Returns error or None."""
    if (layout_sub / "imp.npy").is_file() and (layout_sub / "occ.npy").is_file():
        return None
    imp = read_impedance_file(sample["impedance_path"])
    if imp is None:
        return "Failed to read impedance"
    if len(imp) != EXPECTED_IMP_LENGTH:
        return f"Invalid impedance length {len(imp)}"
    if sample["decap_vector"] is None:
        return "Missing decap_vector for new layout"
    layout_sub.mkdir(parents=True, exist_ok=True)
    np.save(layout_sub / "imp.npy", imp.reshape(-1, 1))
    np.save(layout_sub / "occ.npy", create_occupancy_vector(sample["decap_vector"]))
    return None


def _process_sample(task: tuple) -> tuple[int, bool, str | None, dict | None]:
    idx, sample, hm_dir, layouts_dir, pifreq_dir = task
    try:
        stacked = create_Heatmaps(sample["heatmap_path"], mask_board=_WORKER_MASK_BOARD, verbose=False)
        name = f"sample_{idx + 1}.npy"
        np.save(hm_dir / name, stacked)
        np.save(pifreq_dir / name, np.array(FREQ_HZ[sample["freq_label"]], dtype=np.float64))

        err = _write_layout_once(layouts_dir / sample["design_id"], sample)
        if err:
            return idx, False, err, None

        row = {
            "sample_name": name,
            "design_id": sample["design_id"],
            "freq_label": sample["freq_label"],
            "freq_mhz": sample["freq_mhz"],
            "freq_hz": FREQ_HZ[sample["freq_label"]],
            "source_folder": sample["source_folder"],
            "pi_number": sample["pi_number"],
            "decap_index": sample["decap_index"],
        }
        return idx, True, None, row
    except Exception as e:
        return idx, False, str(e)[:200], None


def _next_sample_index(hm_dir: Path) -> int:
    mx = 0
    for p in hm_dir.glob("sample_*.npy"):
        try:
            mx = max(mx, int(p.stem.split("_")[1]))
        except (IndexError, ValueError):
            continue
    return mx


def _write_manifest(root: Path, manifest_rows: list[dict], *, append: bool) -> None:
    if not manifest_rows:
        return
    manifest_rows.sort(key=lambda r: r["sample_name"])
    manifest_path = root / "manifest.csv"
    if append and manifest_path.is_file():
        with manifest_path.open(newline="", encoding="utf-8") as f:
            existing = list(csv.DictReader(f))
        existing.extend(manifest_rows)
        manifest_rows = existing
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(manifest_rows[0].keys()))
        w.writeheader()
        w.writerows(manifest_rows)
    print(f"Wrote manifest: {manifest_path} ({len(manifest_rows)} rows)")


def create_training_dataset(
    output_root: Path | str = TRAIN_DATA_ROOT,
    data_root=None,
    max_samples: int | None = MAX_SAMPLES,
    num_workers: int | None = None,
    verbose: bool = True,
    overwrite_output: bool = OVERWRITE_OUTPUT,
    *,
    append: bool = False,
    only_freq_mhz: list[float] | None = None,
) -> int:
    """Build multifreq dataset: one row per (layout, MHz)."""
    root = Path(output_root)
    if append:
        root.mkdir(parents=True, exist_ok=True)
        if only_freq_mhz:
            _refresh_freq_tables(only_mhz=only_freq_mhz)
    else:
        prepare_output_dir(root, overwrite=overwrite_output, verbose=verbose)
        _refresh_freq_tables()

    hm_dir = root / "heatmap"
    layouts_dir = root / "layouts"
    pifreq_dir = root / "PI_freq"
    for d in (hm_dir, layouts_dir, pifreq_dir):
        d.mkdir(parents=True, exist_ok=True)

    print("Collecting (layout × frequency) samples...")
    all_samples = list(iter_dataset_samples(data_root, verbose=True))
    if only_freq_mhz is not None:
        allowed = set(anchors_to_freq_labels(only_freq_mhz))
        all_samples = [s for s in all_samples if s["freq_label"] in allowed]
        print(f"  Filtered to MHz {only_freq_mhz} → {len(all_samples)} rows")
    print(f"\nFound {len(all_samples)} total rows (all freq per layout)")

    if not all_samples:
        print("No samples found!")
        return 0

    print(f"Unique layouts (design_id): {len({s['design_id'] for s in all_samples})}")

    start_idx = _next_sample_index(hm_dir) if append else 0
    if append and start_idx > 0:
        print(f"Append mode: continuing sample index from {start_idx + 1}")

    if max_samples is not None and max_samples < len(all_samples):
        print(f"Selecting {max_samples} rows with per-MHz balancing (seed=42)...")
        selected_samples = select_samples_balanced(all_samples, max_samples, seed=42)
    else:
        selected_samples = all_samples
        random.seed(42)
        random.shuffle(selected_samples)
        print(f"Using all {len(selected_samples)} rows (shuffled, seed=42).")

    freq_counts = Counter(s["freq_label"] for s in selected_samples)
    print("\nPer-frequency row counts:")
    for fl in FREQ_LABELS:
        print(f"  {fl}: {freq_counts.get(fl, 0)}")

    tasks = [
        (start_idx + i, s, hm_dir, layouts_dir, pifreq_dir)
        for i, s in enumerate(selected_samples)
    ]
    n_workers = num_workers if num_workers and num_workers > 0 else min(48, os.cpu_count() or 1)
    print(f"\nRunning with {n_workers} workers...")

    manifest_rows: list[dict] = []
    results: list[bool] = []

    if n_workers == 1:
        _init_worker(FRAME_PATH)
        for task in tasks:
            idx, success, error, row = _process_sample(task)
            results.append(success)
            if success and row:
                manifest_rows.append(row)
            elif verbose and not success:
                print(f"  x sample_{idx + 1}: {error}")
    else:
        try:
            from multiprocessing import Pool

            with Pool(processes=n_workers, initializer=_init_worker, initargs=(str(FRAME_PATH),)) as pool:
                for i, out in enumerate(pool.imap_unordered(_process_sample, tasks)):
                    idx, success, error, row = out
                    results.append(success)
                    if success and row:
                        manifest_rows.append(row)
                    if verbose and (i + 1) % max(1, len(tasks) // 10) == 0:
                        print(f"  Progress: {i + 1}/{len(tasks)}")
        except Exception as e:
            print(f"Parallel failed ({e}); sequential fallback.")
            _init_worker(FRAME_PATH)
            for task in tasks:
                idx, success, error, row = _process_sample(task)
                results.append(success)
                if success and row:
                    manifest_rows.append(row)

    _write_manifest(root, manifest_rows, append=append)
    return sum(1 for r in results if r)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(
        description="Build multifreq dataset (layouts/ + heatmap/ + PI_freq/ + manifest.csv)",
    )
    ap.add_argument("--append", action="store_true", help="Add rows to existing output (do not wipe)")
    ap.add_argument(
        "--freqs-mhz",
        nargs="*",
        type=float,
        default=None,
        help="Only process these MHz (append). Default: all anchors in YAML.",
    )
    ap.add_argument("--data-root", type=str, default=None, help="Override Raw data root")
    ap.add_argument("--output", type=str, default=str(TRAIN_DATA_ROOT))
    args = ap.parse_args()

    if args.freqs_mhz:
        _refresh_freq_tables(only_mhz=args.freqs_mhz)

    print("=" * 60)
    print("MULTI-FREQUENCY DATASET (layout store)")
    print("=" * 60)
    print(f"Anchor MHz : {[FREQ_HZ[f] / 1e6 for f in FREQ_LABELS]}")
    print(f"Output     : {args.output}")
    print(f"Data root  : {args.data_root or DEFAULT_DATA_ROOT or 'Not found'}")
    print(f"Mode       : {'append' if args.append else 'rebuild'}")
    print(f"Max rows   : {MAX_SAMPLES if MAX_SAMPLES else 'All'}")
    print("=" * 60)

    count = create_training_dataset(
        output_root=args.output,
        data_root=args.data_root,
        max_samples=MAX_SAMPLES,
        num_workers=int(os.getenv("NUM_WORKERS", "64")),
        verbose=True,
        overwrite_output=False if args.append else OVERWRITE_OUTPUT,
        append=args.append,
        only_freq_mhz=args.freqs_mhz,
    )

    print(f"\nProcessed {count} samples")
    print(f"Next: python scripts/Normalization.py{' --append' if args.append else ''}")
