"""Rules the whole thesis_code/ folder keeps: no experiment names, one file with absolute paths, no
imports of the original code, every module importable, nothing defined that nothing uses."""
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / "thesis_code"
PY = sorted(p for p in ROOT.rglob("*.py") if "runs" not in p.relative_to(ROOT).parts)
TEXT = sorted(p for p in ROOT.rglob("*") if p.suffix in (".py", ".md", ".yaml", ".json")
              and "runs" not in p.relative_to(ROOT).parts)


def test_no_experiment_ids_outside_settings():
    bad = [f"{p}:{i}" for p in TEXT if p.name != "settings.py"
           for i, l in enumerate(p.read_text().splitlines(), 1) if re.search(r"exp0\d\d", l)]
    assert not bad, bad[:10]


def test_no_absolute_paths_outside_settings():
    pat = re.compile(r"(/home/|/mnt/|C:\\\\|[A-Z]:\\)")
    bad = [f"{p}:{i}" for p in PY if p.name != "settings.py"
           for i, l in enumerate(p.read_text().splitlines(), 1) if pat.search(l)]
    assert not bad, bad[:10]


def test_no_imports_of_the_original_code():
    pat = re.compile(r"^\s*(from|import)\s+(libs|src_vae|pipelines|scrap|experiments|repo_paths|active_learning_pi)\b")
    bad = [f"{p}:{i}" for p in PY for i, l in enumerate(p.read_text().splitlines(), 1) if pat.search(l)]
    assert not bad, bad[:10]


def test_every_module_imports():
    for p in PY:
        mod = ".".join(p.relative_to(ROOT.parent).with_suffix("").parts)
        r = subprocess.run([sys.executable, "-c", f"import {mod}"], cwd=ROOT.parent, capture_output=True, text=True)
        assert r.returncode == 0, f"{mod}: {r.stderr[-500:]}"


def test_no_unreferenced_definitions():
    r = subprocess.run([sys.executable, "tests/thesis_code/tools/unreferenced.py"], cwd=ROOT.parent, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout
