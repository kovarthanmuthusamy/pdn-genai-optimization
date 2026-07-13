#!/usr/bin/env python3
"""Normalize scratch/*.py to use repo_paths."""
from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCRATCH = REPO / "scratch"

OLD_NEXT = re.compile(
    r"sys\.path\.insert\(0, str\(next\(p for p in Path\(__file__\)\.resolve\(\)\.parents if \(p / \"src_vae\"\)\.is_dir\(\)\)\)\)\n",
)
NEW_BOOT = "from repo_paths import setup_path\n\nsetup_path()\n"

OLD_ROOT1 = re.compile(
    r"(_ROOT = Path\(__file__\)\.resolve\(\)\.parents\[1\]\n"
    r"(?:if str\(_ROOT\) not in sys\.path:\n    )?"
    r"sys\.path\.insert\(0, str\(_ROOT\)\)\n)",
    re.MULTILINE,
)
OLD_ROOT1_ALT = re.compile(
    r"ROOT = Path\(__file__\)\.resolve\(\)\.parents\[1\]\n"
    r"sys\.path\.insert\(0, str\(ROOT\)\)\n",
)

OLD_PROJECT_NEXT = re.compile(
    r"_ROOT = Path\(__file__\)\.resolve\(\)\n"
    r"PROJECT_ROOT = next\(\n"
    r"    \(p for p in _ROOT\.parents if \(p / \"src_vae\"\)\.is_dir\(\) and \(p / \"datasets\"\)\.is_dir\(\)\),\n"
    r"    _ROOT\.parents\[1\],\n"
    r"\)\n"
    r"(?:if str\(PROJECT_ROOT\) not in sys\.path:\n    )?"
    r"sys\.path\.insert\(0, str\(PROJECT_ROOT\)\)\n",
    re.MULTILINE,
)

OLD_PROJECT_NEXT2 = re.compile(
    r"PROJECT_ROOT = next\(\n"
    r"    \(p for p in _ROOT\.parents if \(p / \"src_vae\"\)\.is_dir\(\) and \(p / \"datasets\"\)\.is_dir\(\)\),\n"
    r"    _ROOT\.parents\[1\],\n"
    r"\)\n"
    r"sys\.path\.insert\(0, str\(PROJECT_ROOT\)\)\n",
)

OLD_SMOKE = re.compile(
    r"_ROOT = Path\(__file__\)\.resolve\(\)\n"
    r"PROJECT_ROOT = next\(\n"
    r"    str\(p\) for p in _ROOT\.parents if \(Path\(p\) / \"src_vae\"\)\.is_dir\(\)\n"
    r"\)\n"
    r"if PROJECT_ROOT not in sys\.path:\n"
    r"    sys\.path\.insert\(0, PROJECT_ROOT\)\n",
)


def patch(text: str) -> str:
    text = OLD_NEXT.sub(NEW_BOOT, text)
    text = OLD_ROOT1.sub("from repo_paths import REPO_ROOT, setup_path\n\nsetup_path()\n", text)
    text = OLD_ROOT1_ALT.sub("from repo_paths import REPO_ROOT, setup_path\n\nsetup_path()\n", text)
    text = OLD_PROJECT_NEXT.sub(
        "from repo_paths import REPO_ROOT as PROJECT_ROOT, setup_path\n\nsetup_path()\n", text
    )
    text = OLD_PROJECT_NEXT2.sub(
        "from repo_paths import REPO_ROOT as PROJECT_ROOT, setup_path\n\nsetup_path()\n", text
    )
    text = OLD_SMOKE.sub(
        "from repo_paths import REPO_ROOT as PROJECT_ROOT, setup_path\n\nsetup_path()\n", text
    )
    text = text.replace("PROJECT_ROOT /", "REPO_ROOT /")
    if "from repo_paths import REPO_ROOT as PROJECT_ROOT" in text:
        text = text.replace("PROJECT_ROOT /", "REPO_ROOT /")
    return text


def main() -> None:
    for py in sorted(SCRATCH.glob("*.py")):
        try:
            orig = py.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        new = patch(orig)
        if new != orig:
            py.write_text(new, encoding="utf-8")
            print(f"  {py.name}")


if __name__ == "__main__":
    main()
