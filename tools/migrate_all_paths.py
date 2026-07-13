#!/usr/bin/env python3
"""Migrate stale absolute paths to repo-relative / repo_paths usage.

Fixes:
  - experiments/*/config.yaml: /home/ubuntu/gan/... and /home/ubuntu/genai_pdn/...
  - Python: parents[N] repo-root bootstraps → repo_paths bootstrap
  - Python: hard-coded /home/ubuntu/gan and /home/ubuntu/genai_pdn string prefixes
  - Shell: cd /home/ubuntu/gan → cd to repo root via $(dirname ...) or relative note

Run from repo root:
  python tools/migrate_all_paths.py
  python tools/migrate_all_paths.py --dry-run
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SKIP_PY = {
    "repo_paths.py",
    "gan_paths.py",
    "migrate_repo_paths.py",
    "migrate_all_paths.py",
    "fix_repo_paths_bootstrap.py",
}
SKIP_PARTS = {".git", "__pycache__", "node_modules"}

STALE_PREFIXES = ("/home/ubuntu/gan/", "/home/ubuntu/genai_pdn/", "/home/ubuntu/GAN/")

# parents[3] bootstrap (experiments/expNNN/codes/*.py)
_BOOTSTRAP_INLINE = (
    "import sys\n"
    "from pathlib import Path\n\n"
    "_REPO_BOOT = Path(__file__).resolve().parents[{depth}]\n"
    "if str(_REPO_BOOT) not in sys.path:\n"
    "    sys.path.insert(0, str(_REPO_BOOT))\n\n"
    "from repo_paths import REPO_ROOT as {alias}, setup_path\n"
    "setup_path()\n"
)

_BOOTSTRAP_P3 = re.compile(
    r"^(?:PROJECT_ROOT|_PROJECT_ROOT) = Path\(__file__\)\.resolve\(\)\.parents\[3\]\n"
    r"if str\((?:PROJECT_ROOT|_PROJECT_ROOT)\) not in sys\.path:\n"
    r"    sys\.path\.insert\(0, str\((?:PROJECT_ROOT|_PROJECT_ROOT)\)\)\n",
    re.MULTILINE,
)
_BOOTSTRAP_P2 = re.compile(
    r"^(?:PROJECT_ROOT|_PROJECT_ROOT) = Path\(__file__\)\.resolve\(\)\.parents\[2\]\n"
    r"if str\((?:PROJECT_ROOT|_PROJECT_ROOT)\) not in sys\.path:\n"
    r"    sys\.path\.insert\(0, str\((?:PROJECT_ROOT|_PROJECT_ROOT)\)\)\n",
    re.MULTILINE,
)

# String PROJECT_ROOT literals
_STR_PROJECT_ROOT = re.compile(
    r'^(PROJECT_ROOT|_PROJECT_ROOT)\s*=\s*["\']/home/ubuntu/(?:gan|genai_pdn)["\']\s*\n',
    re.MULTILINE,
)
_STR_PROJECT_ROOT_REPL = (
    "from repo_paths import REPO_ROOT as PROJECT_ROOT, setup_path\n"
    "setup_path()\n"
)

# _REPO_BOOT block before repo_paths (replace with bootstrap_from)
_OLD_REPO_BOOT = re.compile(
    r"import sys\n"
    r"from pathlib import Path\n\n"
    r"_REPO_BOOT = Path\(__file__\)\.resolve\(\)\.parents\[(?P<depth>\d+)\]\n"
    r"if str\(_REPO_BOOT\) not in sys\.path:\n"
    r"    sys\.path\.insert\(0, str\(_REPO_BOOT\)\)\n\n"
    r"from repo_paths import (?P<imports>[^\n]+)\n"
    r"setup_path\(\)\n",
    re.MULTILINE,
)


def _bootstrap_repl(m: re.Match[str]) -> str:
    depth = m.group("depth")
    imports = m.group("imports")
    return (
        "import sys\n"
        "from pathlib import Path\n\n"
        f"_REPO_BOOT = Path(__file__).resolve().parents[{depth}]\n"
        "if str(_REPO_BOOT) not in sys.path:\n"
        "    sys.path.insert(0, str(_REPO_BOOT))\n\n"
        f"from repo_paths import {imports}\n"
        "setup_path()\n"
    )


def _infer_alias_from_context(text: str, pos: int) -> str:
    window = text[max(0, pos - 200) : pos + 200]
    if "_PROJECT_ROOT" in window:
        return "_PROJECT_ROOT"
    return "PROJECT_ROOT"


def strip_stale_in_text(text: str) -> tuple[str, int]:
    """Replace stale absolute prefixes with repo-relative paths in quoted strings."""
    count = 0
    for prefix in STALE_PREFIXES:
        if prefix in text:
            n = text.count(prefix)
            text = text.replace(prefix, "")
            count += n
    return text, count


def patch_yaml_config(path: Path, *, dry_run: bool) -> bool:
    try:
        text = path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return False
    new_text, n = strip_stale_in_text(text)
    if n == 0 or new_text == text:
        return False
    if not dry_run:
        path.write_text(new_text, encoding="utf-8")
    return True


def patch_python(path: Path, *, dry_run: bool) -> bool:
    if path.name in SKIP_PY:
        return False
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return False
    orig = text

    text = _BOOTSTRAP_P3.sub(
        _BOOTSTRAP_INLINE.format(depth=3, alias="PROJECT_ROOT"), text
    )
    text = _BOOTSTRAP_P2.sub(
        _BOOTSTRAP_INLINE.format(depth=2, alias="PROJECT_ROOT"), text
    )
    text = _STR_PROJECT_ROOT.sub(_STR_PROJECT_ROOT_REPL, text)
    text = _OLD_REPO_BOOT.sub(_bootstrap_repl, text)
    text, _ = strip_stale_in_text(text)

    # Use shared experiment_paths in inference_vae when full pattern present
    if "def _default_data_dir()" in text and "def load_experiment_config" in text:
        text = re.sub(
            r"def _default_data_dir\(\) -> Path:\n"
            r"    if _CONFIG_PATH\.is_file\(\):\n"
            r"        cfg = load_experiment_config\(_CONFIG_PATH\)\n"
            r"        data_dir = cfg\.get\(\"data_dir\"\)\n"
            r"        if data_dir:\n"
            r"            return (?:Path\(data_dir\)|resolve_repo_path\(data_dir\))\n"
            r"    return (?:PROJECT_ROOT|_PROJECT_ROOT) / \"data_multi_norm_unbounded\"\n",
            "def _default_data_dir() -> Path:\n"
            "    from libs.experiment_paths import data_dir_from_config\n\n"
            "    return data_dir_from_config(_CONFIG_PATH)\n",
            text,
            flags=re.MULTILINE,
        )
        text = re.sub(
            r"def _norm_stats_path\(\) -> Path:\n"
            r"    p = _default_data_dir\(\) / \"normalization_stats\.json\"\n"
            r"    if p\.is_file\(\):\n"
            r"        return p\n"
            r"    legacy = (?:PROJECT_ROOT|_PROJECT_ROOT) / \"datasets\" / \"data_norm\" / \"normalization_stats\.json\"\n"
            r"    if legacy\.is_file\(\):\n"
            r"        return legacy\n"
            r"    raise FileNotFoundError\(\n"
            r"        f\"normalization_stats\.json not found\. Tried:\\n  \{p\}\\n  \{legacy\}\"\n"
            r"    \)\n",
            "def _norm_stats_path() -> Path:\n"
            "    from libs.experiment_paths import norm_stats_path\n\n"
            "    return norm_stats_path(_default_data_dir())\n",
            text,
            flags=re.MULTILINE,
        )
    text = re.sub(
        r"^REPO_DIR = Path\(__file__\)\.resolve\(\)\.parents\[2\]\n",
        "from repo_paths import REPO_ROOT as REPO_DIR  # noqa: E402\n",
        text,
        flags=re.MULTILINE,
    )

    if text != orig:
        if not dry_run:
            path.write_text(text, encoding="utf-8")
        return True
    return False


def patch_shell(path: Path, *, dry_run: bool) -> bool:
    try:
        text = path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return False
    new_text = text.replace("cd /home/ubuntu/gan", 'cd "$(git rev-parse --show-toplevel 2>/dev/null || pwd)"')
    new_text = new_text.replace("/home/ubuntu/gan/", "")
    if new_text == text:
        return False
    if not dry_run:
        path.write_text(new_text, encoding="utf-8")
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    changed_yaml: list[Path] = []
    changed_py: list[Path] = []
    changed_sh: list[Path] = []

    for path in REPO.rglob("config.yaml"):
        if any(p in SKIP_PARTS for p in path.parts):
            continue
        if path.parts[0:2] == ("experiments",) or "experiments" in path.parts:
            if patch_yaml_config(path, dry_run=args.dry_run):
                changed_yaml.append(path.relative_to(REPO))

    for py in REPO.rglob("*.py"):
        if any(p in SKIP_PARTS for p in py.parts):
            continue
        if patch_python(py, dry_run=args.dry_run):
            changed_py.append(py.relative_to(REPO))

    for sh in REPO.rglob("*.sh"):
        if any(p in SKIP_PARTS for p in sh.parts):
            continue
        if patch_shell(sh, dry_run=args.dry_run):
            changed_sh.append(sh.relative_to(REPO))

    mode = "Would patch" if args.dry_run else "Patched"
    print(f"{mode} {len(changed_yaml)} config.yaml, {len(changed_py)} .py, {len(changed_sh)} .sh")
    for p in sorted(changed_yaml):
        print(f"  yaml: {p}")
    for p in sorted(changed_py)[:80]:
        print(f"  py:   {p}")
    if len(changed_py) > 80:
        print(f"  ... and {len(changed_py) - 80} more .py files")
    for p in sorted(changed_sh):
        print(f"  sh:   {p}")


if __name__ == "__main__":
    main()
