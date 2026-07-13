#!/usr/bin/env python3
"""Fix docstrings corrupted by migrate_repo_paths.py (setup_path lines inside quotes)."""
from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BAD = re.compile(
    r"\nfrom repo_paths import setup_path\nsetup_path\(\)\n",
    re.MULTILINE,
)

NEEDS_SETUP = [
    "pipelines/analysis/check_mask.py",
    "pipelines/normalize/apply_stats.py",
]


def fix_file(path: Path) -> bool:
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return False
    orig = text
    text = BAD.sub("\n", text)

    if text != orig and "setup_path()" not in text.split('"""', 2)[-1][:400]:
        # Re-insert setup_path after future imports if missing in code body
        if "from repo_paths import" in text:
            if "setup_path()" not in text:
                text = text.replace(
                    "from repo_paths import repo_path",
                    "from repo_paths import repo_path, setup_path\n\nsetup_path()",
                    1,
                )
                text = text.replace(
                    "from repo_paths import REPO_ROOT",
                    "from repo_paths import REPO_ROOT, setup_path\n\nsetup_path()",
                    1,
                )
        else:
            marker = "from __future__ import annotations\n"
            if marker in text:
                text = text.replace(
                    marker,
                    marker + "\nfrom repo_paths import setup_path\n\nsetup_path()\n",
                    1,
                )

    if text != orig:
        path.write_text(text, encoding="utf-8")
        return True
    return False


def main() -> None:
    changed = []
    for py in REPO.rglob("*.py"):
        if ".git" in py.parts:
            continue
        if fix_file(py):
            changed.append(py.relative_to(REPO))
    print(f"Fixed {len(changed)} files")


if __name__ == "__main__":
    main()
