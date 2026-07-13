"""PEB PI-Distribution frequency replacement (shared by change_frequency / regenerate_mhz_pebs)."""
from __future__ import annotations

import re
from pathlib import Path

_FREQ_PATTERN = re.compile(r'(<EditPIDistribution Frequency=")[^"]*(")')


def replace_frequency_mhz(content: str, mhz: int | float | str) -> tuple[str, int]:
    """Replace all PI-Distribution frequency tags; return (new_content, replacement_count)."""
    hz = f"{mhz}e6"
    return re.subn(_FREQ_PATTERN, rf"\g<1>{hz}\2", content)


def write_peb_at_mhz(src: Path, dst: Path, mhz: int | float | str) -> int:
    """Read *src*, set frequency to *mhz*, write *dst*; return replacement count."""
    new_content, count = replace_frequency_mhz(src.read_text(encoding="utf-8"), mhz)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(new_content, encoding="utf-8")
    return count
