#!/usr/bin/env python3
"""One-off: add Config keys to Agent notes and normalize CONFIGURATION headers."""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

FOLDERS = [
    "active_learning_pi",
    "Data_Creation",
    "datasets",
    "experiments/exp043/codes",
    "Latent_opm",
    "New_heatmaps",
    "scrap",
    "scripts",
    "src_vae",
]

CONFIG_HEADER = "# CONFIGURATION — edit these before running:"

CONFIG_BLOCK_RE = re.compile(
    r"(# ={5,}\s*\n"
    r"# CONFIGURATION[^\n]*\n"
    r"# ={5,}\s*\n)"
    r"(.*?)"
    r"(# ={5,}\s*\n)",
    re.DOTALL,
)

ALT_CONFIG_RE = re.compile(
    r"(# ={5,}\s*\n# CONFIGURATION\s*\n# ={5,}\s*\n)(.*?)(# ={5,}\s*\n)",
    re.DOTALL,
)


def _extract_keys(block: str) -> list[str]:
    keys: list[str] = []
    for line in block.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        if "=" in s and not s.startswith("def ") and not s.startswith("class "):
            key = s.split("=", 1)[0].strip()
            if key and key.isidentifier():
                keys.append(key)
    return keys


def _find_config_keys(text: str) -> list[str] | None:
    for pat in (CONFIG_BLOCK_RE, ALT_CONFIG_RE):
        m = pat.search(text)
        if m:
            return _extract_keys(m.group(2))
    # Legacy: "# ── Config" section until blank line + def/import
    m = re.search(r"# ── Config[^\n]*\n(.*?)(?=\n\n(?:def |class |[A-Z_]+ = ))", text, re.DOTALL)
    if m:
        keys = _extract_keys(m.group(1))
        if keys:
            return keys
    return None


def _is_library(text: str, path: Path) -> bool:
    if 'if __name__' not in text:
        return True
    if "import only" in text.lower() or "library module" in text.lower():
        return True
    if path.name == "__init__.py":
        return True
    return False


def _add_config_keys_line(doc: str, keys_line: str) -> str:
    if "Config keys:" in doc:
        return doc
    # Insert after last Agent notes bullet, before closing """
    lines = doc.splitlines()
    insert_at = None
    for i, line in enumerate(lines):
        if line.strip().startswith("- ") and "Agent notes" not in line:
            insert_at = i + 1
    if insert_at is None:
        # append before end
        if lines and lines[-1].strip() == '"""':
            lines.insert(-1, keys_line)
            return "\n".join(lines)
        return doc + "\n" + keys_line
    lines.insert(insert_at, keys_line)
    return "\n".join(lines)


def _fix_run_lines(doc: str, rel: str) -> str:
    doc = re.sub(
        r"Run:\s*\n(?:\s+python[^\n]+\n)+",
        f"Run:\n    python {rel}\n",
        doc,
        count=1,
    )
    doc = re.sub(r"\s+--[\w-]+[^\n]*", "", doc)
    return doc


def process_file(path: Path) -> bool:
    text = path.read_text(encoding="utf-8")
    if '"""' not in text[:800]:
        return False
    if "Agent notes:" not in text:
        return False

    m = re.match(r'^(?:(?:#!/.*\n)|(?:""".*?"""\s*\n))', text, re.DOTALL)
    if not text.lstrip().startswith('"""'):
        dm = re.match(r'^"""[\s\S]*?"""\s*\n', text)
    else:
        dm = re.match(r'^"""[\s\S]*?"""\s*\n', text)
    if not dm:
        return False

    doc = dm.group(0).rstrip("\n")
    if doc.count('"""') < 2:
        return False

    rel = path.relative_to(ROOT).as_posix()
    keys = _find_config_keys(text)
    if keys:
        keys_line = "    - Config keys: " + ", ".join(f"``{k}``" for k in keys)
    elif _is_library(text, path):
        keys_line = "    - Config keys: none (library — import only)"
    else:
        keys_line = "    - Config keys: see CONFIGURATION block below (module-level constants)"

    new_doc = _add_config_keys_line(doc, keys_line)
    new_doc = _fix_run_lines(new_doc, rel)

    if new_doc == doc:
        return False

    new_text = text[: dm.start()] + new_doc + "\n" + text[dm.end() :]
    path.write_text(new_text, encoding="utf-8")
    return True


def main() -> int:
    updated = 0
    for folder in FOLDERS:
        base = ROOT / folder
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*.py")):
            if process_file(path):
                updated += 1
                print(f"  updated doc: {path.relative_to(ROOT)}")
    print(f"\nDone. Updated {updated} files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
