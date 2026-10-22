"""Every module of thesis_code/ and assistant_web/ imports without data, weights or a GPU."""
import importlib
import pkgutil

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
