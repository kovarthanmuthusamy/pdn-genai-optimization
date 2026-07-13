#!/usr/bin/env python3
"""One-shot migration: replace hard-coded repo-root detection with repo_paths imports."""
from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SKIP = {"repo_paths.py", "gan_paths.py", "migrate_repo_paths.py"}

# Block: _PROJECT_ROOT = Path(__file__).resolve().parents[2] + sys.path insert
_BOOTSTRAP_BLOCK = re.compile(
    r"# ── Bootstrap project root ─+\n"
    r"_PROJECT_ROOT = Path\(__file__\)\.resolve\(\)\.parents\[2\]\n"
    r"if str\(_PROJECT_ROOT\) not in sys\.path:\n"
    r"    sys\.path\.insert\(0, str\(_PROJECT_ROOT\)\)\n",
    re.MULTILINE,
)
_BOOTSTRAP_BLOCK2 = re.compile(
    r"_PROJECT_ROOT = Path\(__file__\)\.resolve\(\)\.parents\[2\]\n"
    r"if str\(_PROJECT_ROOT\) not in sys\.path:\n"
    r"    sys\.path\.insert\(0, str\(_PROJECT_ROOT\)\)\n",
    re.MULTILINE,
)
_PROJECT_ROOT_BOOT = (
    "import sys\n"
    "from pathlib import Path\n\n"
    "_REPO_BOOT = Path(__file__).resolve().parents[2]\n"
    "if str(_REPO_BOOT) not in sys.path:\n"
    "    sys.path.insert(0, str(_REPO_BOOT))\n\n"
    "from repo_paths import REPO_ROOT as _PROJECT_ROOT, setup_path\n"
    "setup_path()\n"
)

# Block: PROJECT_ROOT = Path(__file__).resolve().parents[2] + sys.path insert
_PROJECT_ROOT_BLOCK = re.compile(
    r"PROJECT_ROOT = Path\(__file__\)\.resolve\(\)\.parents\[2\]\n"
    r"if str\(PROJECT_ROOT\) not in sys\.path:\n"
    r"    sys\.path\.insert\(0, str\(PROJECT_ROOT\)\)\n",
    re.MULTILINE,
)
_PROJECT_ROOT_REPL = (
    "import sys\n"
    "from pathlib import Path\n\n"
    "_REPO_BOOT = Path(__file__).resolve().parents[2]\n"
    "if str(_REPO_BOOT) not in sys.path:\n"
    "    sys.path.insert(0, str(_REPO_BOOT))\n\n"
    "from repo_paths import REPO_ROOT as PROJECT_ROOT, setup_path\n"
    "setup_path()\n"
)

# sys.path.insert parents[2] lines (standalone)
_SYSPATH_INSERT = re.compile(
    r"sys\.path\.insert\(0, str\(Path\(__file__\)\.resolve\(\)\.parents\[2\]\)\)\n",
    re.MULTILINE,
)

# gan_paths → repo_paths
_GAN_PATHS_IMPORT = re.compile(r"from gan_paths import", re.MULTILINE)

# Hard-coded absolute paths
_ABS_GAN = re.compile(r"/home/ubuntu/gan")

# ROOT = Path(__file__).resolve().parents[2]
_ROOT_LINE = re.compile(
    r"^ROOT = Path\(__file__\)\.resolve\(\)\.parents\[2\]\n",
    re.MULTILINE,
)
_ROOT_REPL = (
    "from repo_paths import REPO_ROOT as ROOT, setup_path\n"
    "setup_path()\n"
)

# _REPO_ROOT = Path(__file__).resolve().parents[2]
_REPO_ROOT_VAR = re.compile(
    r"^_REPO_ROOT = Path\(__file__\)\.resolve\(\)\.parents\[2\]\n",
    re.MULTILINE,
)
_REPO_ROOT_REPL = "from repo_paths import REPO_ROOT as _REPO_ROOT, setup_path\nsetup_path()\n"

# _ROOT = Path(__file__).resolve().parents[2]
_ROOT_VAR = re.compile(
    r"^_ROOT = Path\(__file__\)\.resolve\(\)\.parents\[2\]\n",
    re.MULTILINE,
)
_ROOT_VAR_REPL = "from repo_paths import REPO_ROOT as _ROOT, setup_path\nsetup_path()\n"

# _repo_root = Path(__file__).resolve().parents[2]
_REPO_ROOT_LOWER = re.compile(
    r"^_repo_root = Path\(__file__\)\.resolve\(\)\.parents\[2\]\n",
    re.MULTILINE,
)
_REPO_ROOT_LOWER_REPL = (
    "from repo_paths import REPO_ROOT as _repo_root, setup_path\nsetup_path()\n"
)


def _ensure_repo_paths_import(text: str) -> str:
    if "from repo_paths import" in text or "import repo_paths" in text:
        return text
    # Insert after last import block in header
    lines = text.splitlines(keepends=True)
    insert_at = 0
    for i, line in enumerate(lines):
        if line.startswith(("import ", "from ")) or line.strip() == "":
            insert_at = i + 1
        elif insert_at > 0 and not line.startswith(("import ", "from ", "#", '"""', "'''")):
            break
    lines.insert(insert_at, "from repo_paths import setup_path\n")
    lines.insert(insert_at + 1, "setup_path()\n")
    return "".join(lines)


def patch_file(path: Path) -> bool:
    if path.name in SKIP:
        return False
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return False
    orig = text

    text = _BOOTSTRAP_BLOCK.sub(_PROJECT_ROOT_BOOT, text)
    text = _BOOTSTRAP_BLOCK2.sub(_PROJECT_ROOT_BOOT, text)
    text = _PROJECT_ROOT_BLOCK.sub(_PROJECT_ROOT_REPL, text)
    text = _ROOT_LINE.sub(_ROOT_REPL, text)
    text = _REPO_ROOT_VAR.sub(_REPO_ROOT_REPL, text)
    text = _ROOT_VAR.sub(_ROOT_VAR_REPL, text)
    text = _REPO_ROOT_LOWER.sub(_REPO_ROOT_LOWER_REPL, text)

    if _SYSPATH_INSERT.search(text):
        text = _SYSPATH_INSERT.sub("", text)
        text = _ensure_repo_paths_import(text)

    text = _GAN_PATHS_IMPORT.sub("from repo_paths import", text)
    text = _ABS_GAN.sub("/home/ubuntu/genai_pdn", text)

    if text != orig:
        path.write_text(text, encoding="utf-8")
        return True
    return False


def main() -> None:
    changed = []
    for py in REPO.rglob("*.py"):
        if ".git" in py.parts or "__pycache__" in py.parts:
            continue
        if patch_file(py):
            changed.append(py.relative_to(REPO))
    print(f"Patched {len(changed)} Python files:")
    for p in sorted(changed):
        print(f"  {p}")


if __name__ == "__main__":
    main()
