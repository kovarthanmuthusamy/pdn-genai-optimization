"""Move ECADStar PI-* folders into destination (names unchanged)."""
from __future__ import annotations

import shutil
from pathlib import Path

from pipelines.dataset_sim.ecadstar import parse_pi_number, resolve_windows_path


def heatmaps_dir_name(mhz: float) -> str:
    """PI-Distribution destination folder, e.g. ``heatmaps_10MHz``."""
    return f"heatmaps_{int(round(float(mhz)))}MHz"


def _find_pi_folder(source: Path, pi_num: int) -> Path:
    for item in source.iterdir():
        if parse_pi_number(item.name) == pi_num:
            return item
    raise FileNotFoundError(f"Missing PI-{pi_num} under {source}")


def _clear_pi_children(dest_dir: Path) -> None:
    """Remove existing PI-* entries under ``dest_dir`` before a full move."""
    if not dest_dir.is_dir():
        return
    for child in dest_dir.iterdir():
        if parse_pi_number(child.name) is None:
            continue
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


def move_pi_outputs(
    *,
    source_emc_dir: str,
    dest_dir: Path,
    pi_count: int,
    clean_dest: bool,
) -> int:
    """Move PI-1..PI-{pi_count} from EMC → ``dest_dir`` (same folder names).

    ECADStar names outputs in PEB order: row 1 in the .peb → PI-1, row N → PI-N.
    Folders are moved as-is; contents are not renamed.
    """
    source = resolve_windows_path(source_emc_dir)
    if not source.is_dir():
        raise FileNotFoundError(f"EMC dir missing: {source}")

    dest_dir.mkdir(parents=True, exist_ok=True)
    if clean_dest:
        _clear_pi_children(dest_dir)

    moved = 0
    for pi_num in range(1, pi_count + 1):
        src = _find_pi_folder(source, pi_num)
        target = dest_dir / src.name
        if target.exists():
            if target.is_dir():
                shutil.rmtree(target)
            else:
                target.unlink()
        print(f"  MOVE {src.name} → {dest_dir.name}/{src.name}")
        shutil.move(str(src), str(target))
        moved += 1

    return moved
