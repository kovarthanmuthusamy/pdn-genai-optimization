"""Package the merged and normalized dataset as one self-describing folder.

What: writes into OUTPUT_DIR/s04_dataset/package/:
        raw/, normalized/          links to the merged raw store and its normalized copy
        test_pool/                 the pool placements left aside, merged and normalized with the
                                   dataset's own statistics (raw/, normalized/, index_*.csv)
        index_layouts.csv          one row per placement: K, parts, part class, slot C1, type codes,
                                   map count and frequencies, origin, simulation campaign, split
        index_maps.csv             one row per map: frequency, campaign, raw eCADSTAR folder, split
        split_train.txt / split_heldout.txt   the training code's own split rule (sorted design ids,
                                   seed 42, 90 % for training)
        README.md                  data card; every number computed from the files
    The campaign of a map is found by following each merged store's source list back to the
    processed campaign store that holds it.
Usage: python -m thesis_code.s04_dataset.package
Config keys: SRC_RAW, SRC_NORM, V1_INDEX, TRAIN_SPLIT, SPLIT_SEED.
"""
from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from thesis_code.common import settings
from thesis_code.common.frequency import freqs_mhz
from thesis_code.common.paths import out_dir, plan_file
from thesis_code.s02_processing import merge
from thesis_code.s03_normalization.normalize import normalize_store
from thesis_code.s04_dataset.loader import _split_indices_by_design

# =============================================================================
# CONFIGURATION — edit these before running
# =============================================================================
SRC_RAW = settings.DATA_DIR / "raw"                 # merged raw store (merge.py output for a rebuild)
SRC_NORM = settings.DATA_DIR / "normalized"         # its normalized copy (normalize.py output)
V1_INDEX = settings.REPO / "datasets/pdn_k1-10_balanced_v1/index_layouts.csv"   # origin of first-set placements
TRAIN_SPLIT, SPLIT_SEED = 0.9, 42
# =============================================================================

STEM = "balance30k"
FORM = (1, 1, 1, 8)                         # maps at 200 MHz, below 10, above 500, in 10-500 MHz
PART = {1: "100nF", 2: "47nF", 3: "22nF"}


def rows_of(store: Path) -> list[dict]:
    with (store / "manifest.csv").open(newline="") as fh:
        return list(csv.DictReader(fh))


def meta_of(store: Path) -> dict:
    p = store / "dataset_meta.json"
    return json.loads(p.read_text()) if p.is_file() else {}


def ids_file(p: Path) -> list[str]:
    return p.read_text().split() if p.is_file() else []


_LEAF: dict[Path, dict[str, tuple[Path, str]]] = {}


def leaf_map(store: Path) -> dict[str, tuple[Path, str]]:
    """sample_name -> (processed campaign store, its sample_name), following merge provenance."""
    store = store.resolve()
    if store not in _LEAF:
        m, rows = meta_of(store), rows_of(store)
        if not m.get("sources"):
            _LEAF[store] = {r["sample_name"]: (store, r["sample_name"]) for r in rows}
        else:
            src = {s["label"]: Path(s["path"]) for s in m["sources"]}
            sub = {lab: leaf_map(p) for lab, p in src.items() if any(r["source"] == lab for r in rows)}
            _LEAF[store] = {r["sample_name"]: sub[r["source"]][r["source_sample"]] for r in rows}
    return _LEAF[store]


def raw_folder(store: Path) -> str:
    rr = meta_of(store).get("raw_root")
    return Path(rr).name if rr else ""


def form_of(freqs: str) -> tuple[int, int, int, int]:
    f = [float(x) for x in freqs.split(";")]
    return (sum(x == 200.0 for x in f), sum(x < 10 for x in f), sum(x > 500 for x in f),
            sum(10 <= x <= 500 and x != 200.0 for x in f))


def split_designs(designs: list[str]) -> tuple[set[str], set[str]]:
    tr, va = _split_indices_by_design(designs, TRAIN_SPLIT, SPLIT_SEED)
    return {designs[i] for i in tr}, {designs[i] for i in va}


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def layout_table(store: Path, split: dict[str, str], origin: dict[str, str]) -> tuple[list[dict], list[dict]]:
    """Rows of index_layouts.csv and index_maps.csv for one raw store."""
    rows = rows_of(store)
    leaf = leaf_map(store)
    by_d: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_d[r["design_id"]].append(r)
    lay, maps = [], []
    for d in sorted(by_d):
        codes = np.load(store / "layouts" / d / "type.npy").astype(int).reshape(-1)
        tc = [int((codes == t).sum()) for t in (1, 2, 3)]
        first_leaf = leaf[by_d[d][0]["sample_name"]][0]
        freqs = sorted(float(r["freq_mhz"]) for r in by_d[d])
        lay.append({"design_id": d, "K": sum(tc), "n_100nF": tc[0], "n_47nF": tc[1], "n_22nF": tc[2],
                    "parts": "+".join(PART[t] for t, n in zip((1, 2, 3), tc) if n),
                    "part_class": f"{sum(x > 0 for x in tc)}-part", "slot_C1": int(codes[0] > 0),
                    "type_codes": "".join(str(c) for c in codes), "n_maps": len(freqs),
                    "freqs_mhz": ";".join(f"{f:g}" for f in freqs), "origin": origin.get(d, "existing"),
                    "spectrum_campaign": first_leaf.name, "raw_folder": raw_folder(first_leaf),
                    "split": split[d]})
        for r in by_d[d]:
            ls = leaf[r["sample_name"]][0]
            maps.append({"sample_name": r["sample_name"], "design_id": d, "K": sum(tc), "freq_mhz": r["freq_mhz"],
                         "campaign": ls.name, "raw_folder": raw_folder(ls), "split": split[d]})
    return lay, maps


def write_split_lists(raw: Path, out: Path) -> tuple[set[str], set[str]]:
    designs = sorted({r["design_id"] for r in rows_of(raw)})
    tr, va = split_designs(designs)
    out.mkdir(parents=True, exist_ok=True)
    (out / "split_train.txt").write_text("\n".join(sorted(tr)) + "\n")
    (out / "split_heldout.txt").write_text("\n".join(sorted(va)) + "\n")
    return tr, va


def write_indexes(raw: Path, out: Path, *, test_pool: bool = False) -> tuple[list[dict], list[dict]]:
    """index_layouts.csv and index_maps.csv of ``raw`` into ``out``. The dataset gets its split and
    origins; the test pool is all "test_pool" with origin "existing"."""
    designs = sorted({r["design_id"] for r in rows_of(raw)})
    if test_pool:
        split, origin = {d: "test_pool" for d in designs}, {}
    else:
        tr, _ = split_designs(designs)
        split = {d: ("train" if d in tr else "heldout") for d in designs}
        with V1_INDEX.open(newline="") as fh:
            origin = {r["design_id"]: r["origin"] for r in csv.DictReader(fh)}
        origin.update({d: "v1_pool" for d in ids_file(plan_file(f"{STEM}_pool_ids.csv"))})
        origin.update({d: "new_v2" for d in ids_file(plan_file(f"combinations_{STEM}_new_design_ids.csv"))})
    lay, maps = layout_table(raw, split, origin)
    write_csv(out / "index_layouts.csv", lay)
    write_csv(out / "index_maps.csv", maps)
    return lay, maps


def build_test_pool(final: Path) -> Path:
    """Merge the pool placements left aside and normalize them with the dataset's statistics."""
    p = merge.PRESETS[f"{STEM}_testpool"]
    sources = [(label, settings.REPO / spec) for label, spec in p["sources"]]
    filters = {label: {key: plan_file(name) for key, name in flt.items()} for label, flt in p["filters"].items()}
    raw = merge.merge(sources, filters, p["modes"], final / "test_pool/raw", f"{STEM}_testpool")
    normalize_store(raw, final / "test_pool/normalized", final / "normalized/normalization_stats.json")
    return raw


def _table(header: list[str], rows: list[list]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    return "\n".join(out + ["| " + " | ".join(str(c) for c in r) + " |" for r in rows])


def write_readme(final: Path, lay: list[dict], maps: list[dict], tp_lay: list[dict]) -> None:
    stats = json.loads((final / "normalized/normalization_stats.json").read_text())
    hm, imp = stats["Heatmap"], stats["Impedance"]
    f_axis = freqs_mhz()
    f = np.array([float(m["freq_mhz"]) for m in maps])
    per_k = []
    for k in range(1, 11):
        sel = [r for r in lay if r["K"] == k]
        if sel:
            cls = Counter(r["part_class"] for r in sel)
            per_k.append([k, f"{len(sel):,}", f"{cls['1-part']:,}", f"{cls['2-part']:,}", f"{cls['3-part']:,}",
                          f"{sum(r['n_maps'] for r in sel):,}",
                          f"{sum(r['split'] == 'train' for r in sel):,} / {sum(r['split'] == 'heldout' for r in sel):,}"])
    in_form = sum(form_of(r["freqs_mhz"]) == FORM for r in lay)
    tr = sum(r["split"] == "train" for r in lay)
    text = f"""# Balanced K = 1..10 dataset

Power-integrity dataset of one board: decap placements with 1 to 10 decaps, balanced over the decap
budget K and over the decap parts (T1 = 100 nF, T2 = 47 nF, T3 = 22 nF).

- {len(lay):,} placements, {len(maps):,} maps, one spectrum per placement.
- Spectrum: {len(f_axis)} points, {f_axis.min():g}-{f_axis.max():g} MHz.
- Map: 64 x 64 grid of the board at a frequency chosen per placement ({len(np.unique(f)):,} distinct values,
  {f.min():g}-{f.max():g} MHz). {in_form:,} of {len(lay):,} placements have the 11-map form: one map at
  200 MHz, one below 10 MHz, one above 500 MHz and eight in 10-500 MHz.

{_table(["K", "placements", "1-part", "2-part", "3-part", "maps", "train / held-out"], per_k)}

## Normalization (`normalized/`)

- Maps: z = (log(1 + ohm) - median) / IQR per frequency bin ({len(hm['anchors_mhz'])} bin centres),
  interpolated in log-frequency between bin centres; off-board pixels = {stats['background_value']}.
- Spectra: z = (ln ohm - {imp['log_mean']:.4f}) / {imp['log_std']:.4f}.

## Split

`split_train.txt` / `split_heldout.txt`: {tr:,} / {len(lay) - tr:,} placements (sorted design ids, seed
{SPLIT_SEED}, {TRAIN_SPLIT:.0%} for training). Every map of a placement is on the same side.

## Test pool (`test_pool/`)

{len(tp_lay):,} simulated placements left out to keep the dataset even, normalized with the dataset's
own statistics. It is the surplus of each class, so results on it are reported per K and per class.
"""
    (final / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    final = out_dir("s04_dataset") / "package"
    for src, sub in ((SRC_RAW, "raw"), (SRC_NORM, "normalized")):
        link = final / sub
        if not link.exists():
            link.symlink_to(Path(src).resolve())
    write_split_lists(final / "raw", final)
    lay, maps = write_indexes(final / "raw", final)
    tp_raw = build_test_pool(final)
    tp_lay, _ = write_indexes(tp_raw, final / "test_pool", test_pool=True)
    write_readme(final, lay, maps, tp_lay)
    print(f"packaged {len(lay):,} placements, {len(maps):,} maps; test pool {len(tp_lay):,} placements -> {final}")


if __name__ == "__main__":
    main()
