#!/usr/bin/env python3
"""Move legacy data folders into data/ and remove empty legacy directories."""
from __future__ import annotations

import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

MOVES = [
    (ROOT / "New_heatmaps", ROOT / "data" / "heatmaps"),
    (ROOT / "Latent_opm" / "runs", ROOT / "data" / "latent_runs"),
]

ECADSTAR_FILES = [
    "inspect_raw_folder.ps1",
]

REMOVE_DIRS = [
    ROOT / "Data_Creation",
    ROOT / "scripts",
    ROOT / "New_heatmaps",
    ROOT / "Latent_opm",
]

TEXT_REPLACEMENTS = [
    ("New_heatmaps", "data/heatmaps"),
    ("Latent_opm/runs", "data/latent_runs"),
    ('repo_root / "Latent_opm" / "runs"', 'repo_root / "data" / "latent_runs"'),
    ("OUTPUT_ROOT = \"Latent_opm/runs\"", 'OUTPUT_ROOT = "data/latent_runs"'),
    ("scripts/ecadstar", "tools/ecadstar"),
    ("cd Data_Creation &&", "# legacy removed — use pipelines/data/"),
    ("Data_Creation/", "pipelines/data/"),
    ("python Latent_opm/", "python pipelines/latent/"),
    ("Latent_opm.generate_run_report", "pipelines.latent.generate_run_report"),
]


def _merge_move(src: Path, dst: Path) -> None:
    if not src.exists():
        print(f"skip missing {src}")
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        shutil.move(str(src), str(dst))
        print(f"moved {src.relative_to(ROOT)} → {dst.relative_to(ROOT)}")
        return
    # merge contents into existing dst
    for item in src.iterdir():
        target = dst / item.name
        if target.exists():
            if item.is_dir():
                _merge_move(item, target)
            else:
                print(f"skip exists {target}")
        else:
            shutil.move(str(item), str(target))
            print(f"merged {item.relative_to(ROOT)} → {target.relative_to(ROOT)}")
    if src.is_dir() and not any(src.iterdir()):
        src.rmdir()
        print(f"removed empty {src.relative_to(ROOT)}")


def main() -> None:
    # ECADStar helpers → tools/ecadstar/
    ecad_dir = ROOT / "tools" / "ecadstar"
    ecad_dir.mkdir(parents=True, exist_ok=True)
    for name in ECADSTAR_FILES:
        src = ROOT / "scripts" / name
        if src.is_file():
            shutil.move(str(src), str(ecad_dir / name))
            print(f"moved scripts/{name} → tools/ecadstar/")

    for src, dst in MOVES:
        _merge_move(src, dst)

    # Update text references
    for py in list(ROOT.rglob("*.py")) + list(ROOT.rglob("*.md")) + list(ROOT.rglob("*.ps1")):
        if "__pycache__" in str(py) or py.name == "consolidate_data_dirs.py":
            continue
        try:
            text = py.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        orig = text
        for old, new in TEXT_REPLACEMENTS:
            text = text.replace(old, new)
        # heatmaps SCRIPT_DIR pattern
        text = re.sub(
            r'Path\(__file__\)\.resolve\(\)\.parents\[2\] / "New_heatmaps"',
            'Path(__file__).resolve().parents[2] / "data" / "heatmaps"',
            text,
        )
        if text != orig:
            py.write_text(text, encoding="utf-8")
            print(f"updated {py.relative_to(ROOT)}")

    # Remove legacy dirs (only if empty or only README/__pycache__)
    for d in REMOVE_DIRS:
        if not d.exists():
            continue
        for child in list(d.rglob("*")):
            if child.is_file() and child.name in ("README.md",):
                child.unlink()
        for child in sorted(d.rglob("__pycache__"), reverse=True):
            if child.is_dir():
                shutil.rmtree(child, ignore_errors=True)
        # remove empty subdirs bottom-up
        for sub in sorted(d.rglob("*"), key=lambda p: len(p.parts), reverse=True):
            if sub.is_dir() and not any(sub.iterdir()):
                sub.rmdir()
        if d.is_dir() and not any(d.iterdir()):
            d.rmdir()
            print(f"removed directory {d.relative_to(ROOT)}")
        else:
            remaining = list(d.iterdir()) if d.exists() else []
            print(f"WARN not empty, kept: {d.relative_to(ROOT)} ({[x.name for x in remaining]})")

    # datasets/ — remove README only (data dirs stay)
    ds_readme = ROOT / "datasets" / "README.md"
    if ds_readme.is_file():
        ds_readme.unlink()
        print("removed datasets/README.md")

    # Write data/README
    data_readme = ROOT / "data" / "README.md"
    data_readme.parent.mkdir(parents=True, exist_ok=True)
    data_readme.write_text(
        "# data/\n\n"
        "| Path | Contents |\n"
        "|------|----------|\n"
        "| `heatmaps/` | PEB files, `all_combinations.csv`, decap maps |\n"
        "| `latent_runs/` | Latent optimization run outputs |\n\n"
        "Training datasets remain in [`datasets/`](../datasets/) (`data_multifreq`, etc.).\n",
        encoding="utf-8",
    )
    print("wrote data/README.md")


if __name__ == "__main__":
    main()
