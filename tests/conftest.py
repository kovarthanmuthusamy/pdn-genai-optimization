"""Pytest bootstrap: ensure repo root is on sys.path before tests import."""
from repo_paths import setup_path

# Must run before any test imports from the codebase
setup_path()
