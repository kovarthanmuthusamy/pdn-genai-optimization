#!/usr/bin/env python3
"""Insert sys.path bootstrap before ``from repo_paths import`` in entry scripts."""
from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SKIP = {"repo_paths.py", "gan_paths.py", "fix_repo_paths_bootstrap.py", "migrate_repo_paths.py"}
IMPORT_RE = re.compile(r"^from repo_paths import", re.MULTILINE)
BOOTSTRAP_MARK = "_REPO_BOOT = Path(__file__).resolve().parents["


def _bootstrap_block(depth: int) -> str:
    return (
        "import sys\n"
        "from pathlib import Path\n\n"
        f"_REPO_BOOT = Path(__file__).resolve().parents[{depth}]\n"
        "if str(_REPO_BOOT) not in sys.path:\n"
        "    sys.path.insert(0, str(_REPO_BOOT))\n\n"
    )


def patch_file(path: Path) -> bool:
    if path.name in SKIP:
        return False
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return False
    if "from repo_paths import" not in text:
        return False
    if BOOTSTRAP_MARK in text:
        return False

    m = IMPORT_RE.search(text)
    if not m:
        return False

    depth = len(path.relative_to(REPO).parts) - 1
    insert = _bootstrap_block(depth)
    new_text = text[: m.start()] + insert + text[m.start() :]
    if new_text == text:
        return False
    path.write_text(new_text, encoding="utf-8")
    return True


def main() -> None:
    changed: list[Path] = []
    for py in REPO.rglob("*.py"):
        if ".git" in py.parts or "__pycache__" in py.parts:
            continue
        if patch_file(py):
            changed.append(py.relative_to(REPO))
    print(f"Patched {len(changed)} file(s)")
    for p in sorted(changed):
        print(f"  {p}")


if __name__ == "__main__":
    main()
