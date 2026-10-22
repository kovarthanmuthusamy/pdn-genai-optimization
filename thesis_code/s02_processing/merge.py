"""Merge processed campaigns into one raw store, ready for one normalization pass.

What: writes ONE raw store from several processed ones:
        heatmap/sample_N.npy, PI_freq/sample_N.npy   renumbered 1..N across all sources
        layouts/<design_id>/                         imp.npy, type.npy, occ.npy
        manifest.csv                                 source rows, new sample_name, "source" column
        dataset_meta.json                            counts per source / K, frequency range
    Files are hard-linked (a copy across file systems): same bytes, no extra disk.
    Sources are applied in order. A source in "append" mode adds rows (more maps) to placements an
    earlier source owns; its layout files must be identical. Otherwise a design_id that appears in
    a later source replaces that design's rows from earlier sources.
    A filter keeps only the design ids listed in keep_ids_csv and drops the samples listed in
    drop_samples_csv (one name per line).
    Two presets: "balance30k" (the dataset: the first balanced set, the pool placements moved in,
    the new campaign and the fill campaign's extra maps) and "balance30k_testpool" (the rest of the
    pool, kept aside for testing).
Usage: python -m thesis_code.s02_processing.merge
    Writes OUTPUT_DIR/s02_processing/merged_<preset>/ (deleted and rebuilt).
Config keys: PRESET, PRESETS (sources, filters, modes), SAVED_CAMPAIGNS.
"""
from __future__ import annotations

import csv
import json
import os
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from thesis_code.common import settings
from thesis_code.common.paths import out_dir, plan_file

# =============================================================================
# CONFIGURATION — edit these before running
# =============================================================================
# Sources in override order. "campaign:<name>" is process.py's output for that campaign if present,
# else the saved processed store of the campaign.
PRESETS: dict[str, dict] = {
    "balance30k": {
        "sources": [
            ("v1", "datasets/pdn_k1-10_balanced_v1/raw"),
            ("v1_pool", "datasets/pdn_k1-10_balanced_v1/test_pool/raw"),
            ("balance30k_new", "campaign:balance30k_new"),
            ("balance30k_fill", "campaign:balance30k_fill"),
        ],
        "filters": {"v1_pool": {"keep_ids_csv": "balance30k_pool_ids.csv",
                                "drop_samples_csv": "balance30k_drop_samples_pool.csv"}},
        "modes": {"balance30k_fill": "append"},
    },
    "balance30k_testpool": {
        "sources": [("v1_pool", "datasets/pdn_k1-10_balanced_v1/test_pool/raw")],
        "filters": {"v1_pool": {"keep_ids_csv": "balance30k_setaside_ids.csv"}},
        "modes": {},
    },
}
SAVED_CAMPAIGNS = "datasets/_building_blocks/data_multifreq_{}"   # saved processed stores (read only)
PRESET = "balance30k"
# =============================================================================


def _link(src: Path, dst: Path) -> None:
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _read_manifest(root: Path) -> list[dict]:
    with (root / "manifest.csv").open(newline="") as fh:
        return list(csv.DictReader(fh))


def _source(spec: str) -> Path:
    if not spec.startswith("campaign:"):
        return settings.REPO / spec
    name = spec.split(":", 1)[1]
    new = Path(settings.OUTPUT_DIR) / "s02_processing" / name
    return new if (new / "manifest.csv").is_file() else settings.REPO / SAVED_CAMPAIGNS.format(name)


def select_designs(root: Path, rows: list[dict], flt: dict) -> tuple[set[str], dict]:
    """Design ids of one source kept by its filter, plus a per-K summary."""
    by_k: dict[int, list[str]] = {}
    n_unlisted = Counter()
    listed = set(Path(flt["keep_ids_csv"]).read_text().split()) if flt.get("keep_ids_csv") else None
    for d in sorted({r["design_id"] for r in rows}):
        k = int((np.load(root / "layouts" / d / "type.npy").reshape(-1) > 0).sum())
        if listed is not None and d not in listed:
            n_unlisted[k] += 1
            continue
        by_k.setdefault(k, []).append(d)
    keep: set[str] = set()
    summary = {}
    for k in sorted(by_k):
        keep.update(by_k[k])
        summary[str(k)] = {"eligible": len(by_k[k]), "kept": len(by_k[k]), "dropped_c1": 0}
    for k, n in n_unlisted.items():
        summary.setdefault(str(k), {"eligible": 0, "kept": 0, "dropped_c1": 0})["not_listed"] = n
    return keep, summary


def merge(sources: list[tuple[str, Path]], filters: dict[str, dict], modes: dict[str, str], output: Path,
          preset: str) -> Path:
    manifests = {label: _read_manifest(root) for label, root in sources}
    selection = {}
    for label, root in sources:
        if label in filters:
            keep, selection[label] = select_designs(root, manifests[label], filters[label])
            manifests[label] = [r for r in manifests[label] if r["design_id"] in keep]
            if filters[label].get("drop_samples_csv"):
                drop = set(Path(filters[label]["drop_samples_csv"]).read_text().split())
                n0 = len(manifests[label])
                manifests[label] = [r for r in manifests[label] if r["sample_name"] not in drop]
                selection[label]["dropped_samples"] = n0 - len(manifests[label])
    appending = {label for label, _ in sources if modes.get(label) == "append"}
    owner: dict[str, str] = {}               # design_id -> label of the source that wins
    for label, _ in sources:
        if label not in appending:
            for d in {r["design_id"] for r in manifests[label]}:
                owner[d] = label
    roots = dict(sources)
    for label in appending:                  # appended rows must belong to a known, identical placement
        ds_ = {r["design_id"] for r in manifests[label]}
        unknown = ds_ - set(owner)
        if unknown:
            raise SystemExit(f"{label}: {len(unknown)} designs not in any earlier source, e.g. {sorted(unknown)[:3]}")
        bad = [d for d in sorted(ds_) if not np.array_equal(
            np.load(roots[label] / "layouts" / d / "imp.npy"), np.load(roots[owner[d]] / "layouts" / d / "imp.npy"))
            or not np.array_equal(np.load(roots[label] / "layouts" / d / "type.npy").reshape(-1),
                                  np.load(roots[owner[d]] / "layouts" / d / "type.npy").reshape(-1))]
        if bad:
            raise SystemExit(f"{label}: {len(bad)} designs disagree with their owner's layout files, e.g. {bad[:3]}")
    replaced = Counter()
    for i, (label, _) in enumerate(sources):
        if label in appending:
            continue
        mine = {r["design_id"] for r in manifests[label]}
        for later, _ in sources[i + 1:]:
            if later not in appending and (clash := mine & {r["design_id"] for r in manifests[later]}):
                replaced[f"{label} <- {later}"] += len(clash)

    if output.is_dir():
        shutil.rmtree(output)
    for sub in ("heatmap", "PI_freq", "layouts"):
        (output / sub).mkdir(parents=True)
    rows_out: list[dict] = []
    per_source = Counter()
    for label, root in sources:
        for r in manifests[label]:
            if label not in appending and owner[r["design_id"]] != label:
                continue
            name = f"sample_{len(rows_out) + 1}.npy"
            _link(root / "heatmap" / r["sample_name"], output / "heatmap" / name)
            _link(root / "PI_freq" / r["sample_name"], output / "PI_freq" / name)
            rows_out.append({**r, "sample_name": name, "source": label, "source_sample": r["sample_name"]})
            per_source[label] += 1
        for d in sorted({r["design_id"] for r in manifests[label]}):
            if label in appending or owner[d] != label:
                continue
            dst = output / "layouts" / d
            dst.mkdir()
            for f in (root / "layouts" / d).iterdir():
                _link(f, dst / f.name)

    fields: list[str] = []
    for r in rows_out:
        fields += [k for k in r if k not in fields]
    with (output / "manifest.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows_out)
    freqs = np.array([float(r["freq_mhz"]) for r in rows_out])
    k_rows = Counter(int(r["K"]) for r in rows_out)
    k_layouts = Counter(int(r["K"]) for r in {r["design_id"]: r for r in rows_out}.values())
    meta = {"schema_version": 1, "stage": "raw", "source_script": "thesis_code/s02_processing/merge.py",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "sources": [{"label": label, "path": str(root), "rows_kept": per_source[label]} for label, root in sources],
            "preset": preset, "appended_sources": sorted(appending), "replaced_designs": dict(replaced),
            "source_filters": {k: {"filter": {a: str(b) for a, b in filters[k].items()}, "per_k": v}
                               for k, v in selection.items()},
            "n_decap_types": 3, "occupancy_schema": "type_codes_0_to_T", "frequency_scheme": "per_layout_random",
            "counts": {"manifest_rows": len(rows_out), "unique_layouts": len({r["design_id"] for r in rows_out}),
                       "distinct_frequencies": int(len(np.unique(freqs)))},
            "k_counts": {str(k): v for k, v in sorted(k_rows.items())},
            "k_layouts": {str(k): v for k, v in sorted(k_layouts.items())},
            "freq_mhz_range": [float(freqs.min()), float(freqs.max())]}
    (output / "dataset_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"merged {len(rows_out):,} rows / {meta['counts']['unique_layouts']:,} placements: {dict(per_source)}")
    return output


def main() -> None:
    p = PRESETS[PRESET]
    sources = [(label, _source(spec)) for label, spec in p["sources"]]
    filters = {label: {key: plan_file(name) for key, name in flt.items()} for label, flt in p["filters"].items()}
    merge(sources, filters, p["modes"], out_dir("s02_processing") / f"merged_{PRESET}", PRESET)


if __name__ == "__main__":
    main()
