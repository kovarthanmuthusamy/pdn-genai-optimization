"""Layout store on disk: one row per map, one folder per placement.

What: a dataset folder holds
        heatmap/sample_N.npy     one map
        PI_freq/sample_N.npy     its frequency in Hz
        layouts/<design_id>/     imp.npy (spectrum), type.npy (52 part codes), occ.npy
        manifest.csv             sample_name -> design_id, frequency, K, ...
Usage (import only): load_manifest_index(dir) -> {"sample_N": design_id};
    validate_multifreq_dataset(dir) raises if a part is missing or the counts disagree;
    layout_imp_path / layout_occ_path / layout_type_path(dir, design_id).
"""
from __future__ import annotations

import csv
from pathlib import Path


def layout_imp_path(data_dir: Path, design_id: str) -> Path:
    return Path(data_dir) / "layouts" / design_id / "imp.npy"


def layout_occ_path(data_dir: Path, design_id: str) -> Path:
    return Path(data_dir) / "layouts" / design_id / "occ.npy"


def layout_type_path(data_dir: Path, design_id: str) -> Path:
    """Part codes 0..T per slot (0 = empty); preferred over occ.npy."""
    return Path(data_dir) / "layouts" / design_id / "type.npy"


def load_manifest_index(data_dir: Path) -> dict[str, str]:
    """Map file stem (``sample_1``) -> design_id."""
    path = Path(data_dir) / "manifest.csv"
    if not path.is_file():
        return {}
    index: dict[str, str] = {}
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            name, did = row.get("sample_name", ""), row.get("design_id", "")
            if name and did:
                index[Path(name).stem] = did
    return index


def validate_multifreq_dataset(data_dir: Path) -> None:
    """Raise if heatmap/, manifest.csv, layouts/ or PI_freq/ is missing, or the row counts disagree."""
    root = Path(data_dir)
    missing = [name for name, ok in (("heatmap/", (root / "heatmap").is_dir()),
                                     ("manifest.csv", (root / "manifest.csv").is_file()),
                                     ("layouts/", (root / "layouts").is_dir()),
                                     ("PI_freq/", (root / "PI_freq").is_dir())) if not ok]
    if missing:
        raise FileNotFoundError(f"dataset incomplete at {root} -- missing: {', '.join(missing)}")
    n_hm = len(list((root / "heatmap").glob("*.npy")))
    n_manifest = len(load_manifest_index(root))
    if n_manifest != n_hm:
        raise ValueError(f"manifest rows ({n_manifest}) != map files ({n_hm}) under {root}")
