#!/usr/bin/env python3
"""Regenerate combined_all_{MHz}MHz.peb from combined_all.peb."""

from __future__ import annotations

import re
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_PEB = SCRIPT_DIR / "combined_all.peb"

ANCHORS_MHZ = [10, 80, 130, 150, 200, 230, 250, 270, 300, 330, 400, 450, 500, 550, 600]


def main() -> None:
    content = INPUT_PEB.read_text(encoding="utf-8")
    pattern = r'(<EditPIDistribution Frequency=")[^"]*(")'
    for mhz in ANCHORS_MHZ:
        out = SCRIPT_DIR / f"combined_all_{mhz}MHz.peb"
        new_content, count = re.subn(pattern, rf"\g<1>{mhz}e6\2", content)
        out.write_text(new_content, encoding="utf-8")
        print(f"{out.name}: {count} frequency replacements")


if __name__ == "__main__":
    main()
