#!/usr/bin/env python3
"""Remove manifest rows + heatmap/PI_freq files for given MHz values in data_multifreq_train."""
from __future__ import annotations

import csv
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from repo_paths import REPO_ROOT, setup_path

setup_path()

from libs.dataset_meta import write_dataset_meta
from src_vae.others.multifreq_layout_store import invalidate_training_caches, manifest_path

DATA_DIR = REPO_ROOT / "datasets" / "data_multifreq_train"
FREQ_MHZ_LIST = [470.0]
EXECUTE = True


def _matches_mhz(row: dict[str, str], freq_mhz: float) -> bool:
    label = f"{int(freq_mhz) if freq_mhz == int(freq_mhz) else freq_mhz}MHz"
    if row.get("freq_label") == label:
        return True
    try:
        return abs(float(row.get("freq_mhz", -1)) - freq_mhz) < 1e-6
    except (TypeError, ValueError):
        return False


def remove_freqs(data_dir: Path, freq_list: list[float], *, execute: bool) -> dict[str, int]:
    data_dir = data_dir.resolve()
    mf = manifest_path(data_dir)
    if not mf.is_file():
        raise FileNotFoundError(f"Missing manifest: {mf}")

    hm_dir = data_dir / "heatmap"
    pf_dir = data_dir / "PI_freq"
    if not hm_dir.is_dir():
        raise FileNotFoundError(f"Missing heatmap dir: {hm_dir}")

    with mf.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    targets = {float(m) for m in freq_list}
    drop = [r for r in rows if any(_matches_mhz(r, m) for m in targets)]
    keep = [r for r in rows if r not in drop]

    counts = {
        "manifest_before": len(rows),
        "manifest_after": len(keep),
        "rows_removed": len(drop),
        "heatmap_removed": 0,
        "pi_freq_removed": 0,
        "errors": 0,
    }
    per_mhz = {int(m): sum(1 for r in drop if _matches_mhz(r, m)) for m in targets}
    counts["per_mhz"] = per_mhz

    if not execute:
        return counts

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup = data_dir / f"manifest.csv.bak_remove_{ts}"
    shutil.copy2(mf, backup)
    print(f"  manifest backup → {backup.name}")

    for row in drop:
        stem = Path(row["sample_name"]).name
        for sub, key in ((hm_dir, "heatmap_removed"), (pf_dir, "pi_freq_removed")):
            if not sub.is_dir():
                continue
            p = sub / stem
            if p.is_file():
                counts[key] += 1
                try:
                    p.unlink()
                except OSError:
                    counts["errors"] += 1

    with mf.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(keep)

    batches = data_dir / "append_batches"
    if batches.is_dir():
        for mhz in targets:
            tag = f"merged_{int(round(mhz))}"
            reg = batches / f"{tag}.json"
            if reg.is_file():
                reg.unlink()
                print(f"  removed batch registry {reg.name}")

    invalidate_training_caches(data_dir)
    write_dataset_meta(
        data_dir,
        stage="raw",
        source_script="pipelines/dataset/remove_mhz_from_train.py",
        extra={"removed_mhz": sorted(targets), "rows_removed": counts["rows_removed"]},
    )
    return counts


def main() -> None:
    mode = "EXECUTE" if EXECUTE else "DRY-RUN"
    print(f"Remove MHz {FREQ_MHZ_LIST} from {DATA_DIR} [{mode}]\n")

    counts = remove_freqs(DATA_DIR, FREQ_MHZ_LIST, execute=EXECUTE)
    print(f"  per MHz removed: {counts.get('per_mhz')}")
    print(
        f"  manifest: {counts['manifest_before']:,} → {counts['manifest_after']:,} "
        f"(-{counts['rows_removed']:,})",
    )
    if EXECUTE:
        print(
            f"  deleted: {counts['heatmap_removed']:,} heatmap, "
            f"{counts['pi_freq_removed']:,} PI_freq",
        )
        if counts["errors"]:
            print(f"  WARNING: {counts['errors']} delete errors")
    elif not EXECUTE:
        print("\nDry-run only. Set EXECUTE = True to apply.")


if __name__ == "__main__":
    main()
