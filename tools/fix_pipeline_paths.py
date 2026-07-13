#!/usr/bin/env python3
"""Fix repo-root path depth after pipelines/ migration."""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PIPELINES = ROOT / "pipelines"

REPLACEMENTS = [
    (r"Path\(__file__\)\.resolve\(\)\.parents\[1\]", "Path(__file__).resolve().parents[2]"),
    (r"Path\(__file__\)\.resolve\(\)\.parent\.parent(?!\.)", "Path(__file__).resolve().parents[2]"),
    (r"here\.parents\[1\]", "here.parents[2]"),
    (r"Path\(__file__\)\.with_name\(\"latent_optimization_impedance\.py\"\)", 'Path(__file__).with_name("optimize.py")'),
]

# optimization_loader special case
OPT_LOADER = PIPELINES / "latent" / "optimization_loader.py"

FILTER_COMBO = PIPELINES / "heatmaps" / "filter_combinations.py"
HEATMAP_CHANGE = PIPELINES / "heatmaps" / "change_frequency.py"
HEATMAP_REGEN = PIPELINES / "heatmaps" / "regenerate_mhz_pebs.py"

for py in PIPELINES.rglob("*.py"):
    if py.name == "_bootstrap.py":
        continue
    text = py.read_text(encoding="utf-8")
    orig = text
    for pat, repl in REPLACEMENTS:
        text = re.sub(pat, repl, text)
    if py == FILTER_COMBO:
        text = text.replace(
            "SCRIPT_DIR = Path(__file__).resolve().parent",
            'SCRIPT_DIR = Path(__file__).resolve().parents[2] / "New_heatmaps"',
        )
    if py in (HEATMAP_CHANGE, HEATMAP_REGEN):
        text = re.sub(
            r"SCRIPT_DIR = Path\(__file__\)\.resolve\(\)\.parent",
            'SCRIPT_DIR = Path(__file__).resolve().parents[2] / "New_heatmaps"',
            text,
        )
    if text != orig:
        py.write_text(text, encoding="utf-8")
        print(f"fixed {py.relative_to(ROOT)}")

# optimization_loader → import optimize module directly
OPT_LOADER.write_text('''"""Lazy loader for pipelines.latent.optimize."""
from __future__ import annotations

import importlib

_MOD = "pipelines.latent.optimize"


def load_optimization_module():
    return importlib.import_module(_MOD)


def resolve_run_dir(run_dir, repo_root):
    lo = load_optimization_module()
    if run_dir is None:
        return lo.resolve_run_dir(repo_root, None)
    from pathlib import Path
    path = Path(run_dir)
    return path.resolve() if path.is_absolute() else (repo_root / path).resolve()
''', encoding="utf-8")
print("fixed optimization_loader.py")

# processing paths via repo_path
for name in ("processing_single.py", "processing_multifreq.py", "processing_eval.py"):
    p = PIPELINES / "data" / name
    if not p.is_file():
        continue
    t = p.read_text(encoding="utf-8")
    t = t.replace("FRAME_PATH = SCRIPT_DIR.parent / \"configs\"", 'FRAME_PATH = repo_path("configs")')
    t = t.replace('TRAIN_DATA_ROOT = SCRIPT_DIR.parent / "datasets"', 'TRAIN_DATA_ROOT = repo_path("datasets")')
    t = t.replace("REPO_DIR = SCRIPT_DIR.parent", "REPO_DIR = Path(__file__).resolve().parents[2]")
    if "from repo_paths import repo_path" not in t:
        t = t.replace("from libs.data_creation", "from repo_paths import repo_path\nfrom libs.data_creation", 1)
    p.write_text(t, encoding="utf-8")
    print(f"fixed data/{name}")

print("done")
