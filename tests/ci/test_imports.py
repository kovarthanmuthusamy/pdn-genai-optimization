"""Every module of thesis_code/ and assistant_web/ imports without data, weights or a GPU, and every
third-party package they import is pinned in requirements.txt (the images install nothing else)."""
import ast
import importlib
import importlib.metadata
import pkgutil
import re
import sys
from pathlib import Path

import pytest

import assistant_web
import thesis_code

MODULES = sorted(m.name for pkg in (thesis_code, assistant_web)
                 for m in pkgutil.walk_packages(pkg.__path__, pkg.__name__ + "."))


def test_found_the_packages():
    assert len(MODULES) > 60
    for stage in ("s01_generation", "s05_model", "s07_training", "s09_inference", "s11_assistant", "s12_evaluation"):
        assert any(m.startswith(f"thesis_code.{stage}") for m in MODULES), stage


@pytest.mark.parametrize("name", MODULES)
def test_module_imports(name):
    importlib.import_module(name)


def test_third_party_imports_are_pinned():
    # A package that arrives only as a dependency of a dev/tracking extra (threadpoolctl via mlflow ->
    # scikit-learn) passes the venv test job but breaks the slimmer Docker image, so check it statically.
    repo = Path(thesis_code.__file__).resolve().parents[1]
    first_party = {"thesis_code", "assistant_web"}
    imported = set()
    for pkg in first_party:
        for path in (repo / pkg).rglob("*.py"):
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                if isinstance(node, ast.Import):
                    imported |= {a.name.split(".")[0] for a in node.names}
                elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                    imported.add(node.module.split(".")[0])
    third_party = imported - set(sys.stdlib_module_names) - first_party - {"__future__"}
    norm = lambda name: re.sub(r"[-_.]+", "-", name).lower()
    pinned = {norm(line.split("==")[0]) for line in (repo / "requirements.txt").read_text().splitlines()
              if line.strip() and not line.startswith("#")} | {"torch"}   # torch: installed from its own index
    dists = importlib.metadata.packages_distributions()
    missing = {mod: dists.get(mod, [mod]) for mod in third_party
               if not any(norm(d) in pinned for d in dists.get(mod, [mod]))}
    assert not missing, f"imported but not pinned in requirements.txt: {missing}"
