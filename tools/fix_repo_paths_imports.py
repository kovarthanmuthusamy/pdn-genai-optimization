#!/usr/bin/env python3
"""Fix broken repo_paths import lines from migrate_all_paths.py."""
from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

BROKEN = re.compile(
    r"from repo_paths import REPO_ROOT, resolve_repo_path as (PROJECT_ROOT|_PROJECT_ROOT), setup_path\n"
)
FIXED = "from repo_paths import REPO_ROOT as PROJECT_ROOT, resolve_repo_path, setup_path\n"

BROKEN2 = re.compile(
    r"from repo_paths import REPO_ROOT, resolve_repo_path as _PROJECT_ROOT, setup_path\n"
)
FIXED2 = "from repo_paths import REPO_ROOT as _PROJECT_ROOT, resolve_repo_path, setup_path\n"

DUP_IMPORTS = re.compile(
    r"import sys\nfrom pathlib import Path\n\nimport sys\nfrom pathlib import Path\n\n",
)


def main() -> None:
    n = 0
    for py in REPO.rglob("*.py"):
        if "__pycache__" in py.parts:
            continue
        try:
            text = py.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        orig = text
        text = BROKEN.sub(FIXED, text)
        text = BROKEN2.sub(FIXED2, text)
        text = DUP_IMPORTS.sub("import sys\nfrom pathlib import Path\n\n", text)
        if text != orig:
            py.write_text(text, encoding="utf-8")
            n += 1
            print(py.relative_to(REPO))
    print(f"Fixed {n} file(s)")


if __name__ == "__main__":
    main()
