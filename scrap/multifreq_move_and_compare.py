"""Multifreq PI-Distribution sweep — copy PEB, move CAD results, compare heatmaps.

Typical workflow
----------------
Full pipeline (recommended)::

    python scrap/run_multifreq_sweep_pipeline.py

Or step-by-step:
1. ``python scrap/generation/run_multifreq_heatmap_sweep.py``
   → writes samples + ``.peb`` (also copied to ``PEB_COPY_DEST``)
2. Batch-simulate the ``.peb`` in ECADStar (manual), or use the pipeline script.
3. ``python scrap/multifreq_move_and_compare.py``
   → move PI-* from ``.emc``, heatmap compare PNGs, HTML + markdown report.

Run
---
    python scrap/multifreq_move_and_compare.py
    python scrap/multifreq_move_and_compare.py --copy-peb-only
    python scrap/multifreq_move_and_compare.py --skip-move
    python scrap/multifreq_move_and_compare.py --skip-compare
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
from pathlib import Path, PureWindowsPath

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scrap.comparison import compare as compare_mod  # noqa: E402
from scrap.generation.run_multifreq_heatmap_sweep import (  # noqa: E402
    HEATMAP_ONLY_PEB,
    K_VALUE,
    NUM_SAMPLES,
    OUTPUT_ROOT,
    PEB_COPY_DEST,
    PEB_OUT_FILE,
    exported_freq_mhz_list,
    write_sweep_freq_manifest,
)
from scrap.build_comparison_report import build_multifreq_sweep_report  # noqa: E402
from scrap.peb_copy import copy_peb_to_folder  # noqa: E402

# ============================================================
# CONFIGURATION
# ============================================================
# ECADStar .emc directory containing PI-1, PI-2, … after batch simulate
SOURCE_EMC_DIR = r"C:\Users\muthusamy\Desktop\design\H-shape.emc"
SOURCE_EMC_DIR_OVERRIDE: str | None = None

MOVE = True
CLEAN_DEST_BEFORE_PASTE = True
PI_NAME_REGEX = re.compile(r"^PI-(\d+)(?:\..+)?$")
# ============================================================


def _resolve_source_dir(path_str: str) -> Path:
    if re.match(r"^[A-Za-z]:\\", path_str):
        win = PureWindowsPath(path_str)
        drive = win.drive.rstrip(":").lower()
        wsl_path = Path("/mnt") / drive / Path(*win.parts[1:])
        if wsl_path.exists():
            return wsl_path
        return Path(path_str)
    return Path(path_str)


def _parse_pi_number(name: str) -> int | None:
    m = PI_NAME_REGEX.match(name)
    return int(m.group(1)) if m else None


def _suffix(name: str) -> str:
    return Path(name).suffix


def _extract_map_mhz(folder: Path) -> float | None:
    for path in sorted(folder.rglob("Z_*MHz.map")):
        m = re.search(r"Z_(\d+(?:\.\d+)?)MHz", path.name, re.IGNORECASE)
        if m:
            return float(m.group(1))
    return None


def _pis_per_sample() -> int:
    return 1 if HEATMAP_ONLY_PEB else 2


def _build_pi_slots(
    freq_list: list[int],
    *,
    k_value: int,
    num_samples: int,
) -> list[tuple[int, int, int]]:
    """Return (target_mhz, local_sample_i, global_pi_index) in PEB order."""
    slots: list[tuple[int, int, int]] = []
    global_idx = 0
    for mhz in freq_list:
        for _local_i in range(num_samples):
            slots.append((int(mhz), _local_i, global_idx))
            global_idx += 1
    return slots


def _rename_heatmap_outputs(dest_dir: Path, *, num_samples: int, sample_offset: int) -> None:
    """Rename PI-* in dest_dir to Heatmap_real_{i} (heatmap-only sweep)."""
    by_num: dict[int, list[Path]] = {}
    for item in dest_dir.iterdir():
        pi_num = _parse_pi_number(item.name)
        if pi_num is None:
            continue
        by_num.setdefault(pi_num, []).append(item)

    pps = _pis_per_sample()
    expected: list[int] = []
    for i in range(num_samples):
        g = sample_offset + i
        if pps == 1:
            expected.append(g + 1)
        else:
            expected.extend([2 * g + 1, 2 * g + 2])

    missing = [n for n in expected if n not in by_num]
    if missing:
        raise SystemExit(
            "Missing expected PI outputs in DEST_DIR.\n"
            f"  Missing PI numbers: {missing}\n"
            f"  DEST_DIR: {dest_dir}\n"
            f"  (heatmap-only sweep expects 1 PI per sample; "
            f"PEB order is freq outer, sample inner)"
        )

    for i in range(num_samples):
        g = sample_offset + i
        if pps == 1:
            pi_nums = [g + 1]
        else:
            pi_nums = [2 * g + 1, 2 * g + 2]
        if pps == 1:
            for item in sorted(by_num[pi_nums[0]], key=lambda p: p.name):
                target = dest_dir / f"Heatmap_real_{i}{_suffix(item.name)}"
                if target.exists():
                    raise SystemExit(f"Target already exists: {target}")
                item.rename(target)
        else:
            for item in sorted(by_num[pi_nums[0]], key=lambda p: p.name):
                target = dest_dir / f"Heatmap_real_{i}{_suffix(item.name)}"
                if target.exists():
                    raise SystemExit(f"Target already exists: {target}")
                item.rename(target)
            for item in sorted(by_num[pi_nums[1]], key=lambda p: p.name):
                target = dest_dir / f"Imp_Real{i}{_suffix(item.name)}"
                if target.exists():
                    raise SystemExit(f"Target already exists: {target}")
                item.rename(target)


def move_sweep_pi_outputs(
    *,
    source_emc_dir: str,
    freq_list: list[int],
    base_generated_dir: Path,
    k_value: int = K_VALUE,
    num_samples: int = NUM_SAMPLES,
) -> None:
    """Move PI-* from ECADStar into freq_*MHz/K{k}/Real/ for this sweep only.

    PEB order (matches run_multifreq_heatmap_sweep.py):
      for mhz in freqs:
        for sample in 0..num_samples-1:
          one PI entry (heatmap-only) or two (distribution + spectrum)
    """
    source_dir = _resolve_source_dir(source_emc_dir)
    if not source_dir.is_dir():
        raise SystemExit(f"Source .emc directory does not exist:\n  {source_dir}")

    pps = _pis_per_sample()
    slots = _build_pi_slots(freq_list, k_value=k_value, num_samples=num_samples)

    pi_to_dest: dict[int, tuple[Path, int, int]] = {}
    dest_dirs: set[Path] = set()
    for mhz, local_i, g_idx in slots:
        dest_dir = base_generated_dir / f"freq_{mhz}MHz" / f"K{k_value}" / "Real"
        dest_dirs.add(dest_dir)
        if pps == 1:
            pi_nums = [g_idx + 1]
        else:
            pi_nums = [2 * g_idx + 1, 2 * g_idx + 2]
        for pi_num in pi_nums:
            pi_to_dest[pi_num] = (dest_dir, local_i, mhz)

    for dest_dir in dest_dirs:
        dest_dir.mkdir(parents=True, exist_ok=True)
        if CLEAN_DEST_BEFORE_PASTE:
            for child in dest_dir.iterdir():
                if child.is_dir() and not child.is_symlink():
                    shutil.rmtree(child)
                else:
                    child.unlink()

    moved = 0
    for item in sorted(source_dir.iterdir(), key=lambda p: p.name):
        pi_num = _parse_pi_number(item.name)
        if pi_num is None or pi_num not in pi_to_dest:
            continue
        dest_dir, local_i, mhz = pi_to_dest[pi_num]
        target = dest_dir / item.name
        action = "MOVE" if MOVE else "COPY"
        print(
            f"{action}: {item} -> {target}   "
            f"(freq_{mhz}MHz/K{k_value}, sample={local_i})"
        )
        if MOVE:
            shutil.move(str(item), str(target))
        else:
            if item.is_dir():
                shutil.copytree(item, target)
            else:
                shutil.copy2(item, target)
        moved += 1

    if moved == 0:
        print("\n⚠ No PI-* files found in source — skipping rename.")
        return

    dir_offsets: dict[Path, int] = {}
    for _mhz, _local_i, g_idx in slots:
        dest_dir = base_generated_dir / f"freq_{_mhz}MHz" / f"K{k_value}" / "Real"
        dir_offsets.setdefault(dest_dir, g_idx)
        dir_offsets[dest_dir] = min(dir_offsets[dest_dir], g_idx)

    for dest_dir, offset in sorted(dir_offsets.items(), key=lambda x: x[0].as_posix()):
        n = sum(1 for mhz, _, _ in slots if dest_dir == base_generated_dir / f"freq_{mhz}MHz" / f"K{k_value}" / "Real")
        _rename_heatmap_outputs(dest_dir, num_samples=n, sample_offset=offset)

    print(
        f"\n✓ Moved {moved} PI item(s) into {len(dest_dirs)} Real/ folder(s) "
        f"({len(freq_list)} freq(s), K={k_value}, {pps} PI/sample)."
    )


def salvage_sweep_heatmap_outputs(
    *,
    freq_list: list[int],
    base_generated_dir: Path,
    k_value: int = K_VALUE,
    num_samples: int = NUM_SAMPLES,
) -> None:
    """Re-group misplaced Heatmap_real_* / PI-* by MHz from .map files."""
    seen: set[Path] = set()
    candidates: list[tuple[float, int, Path]] = []

    for real_dir in sorted(base_generated_dir.glob(f"freq_*MHz/K{k_value}/Real")):
        folder_items = [p for p in real_dir.iterdir() if not p.name.startswith("_")]
        for item in folder_items:
            if not (
                item.name.startswith("Heatmap_real_")
                or item.name.startswith("Imp_Real")
                or _parse_pi_number(item.name) is not None
            ):
                continue
            resolved = item.resolve()
            if resolved in seen:
                continue
            mhz = _extract_map_mhz(item)
            if mhz is None:
                print(f"  skip (no .map MHz): {item}")
                continue
            if item.name.startswith("Imp_Real") and any(
                not x.name.startswith("Imp_Real") and _extract_map_mhz(x) == mhz
                for x in folder_items
            ):
                continue
            seen.add(resolved)
            prio = 0 if item.name.startswith("Heatmap_real_") else 1 if _parse_pi_number(item.name) else 2
            candidates.append((mhz, prio, item))

    if not candidates:
        raise SystemExit("Salvage found no heatmap folders with Z_*MHz.map under the sweep Real/ dirs.")

    by_target: dict[int, list[tuple[float, int, Path]]] = {int(m): [] for m in freq_list}
    for mhz, prio, item in candidates:
        target = min(freq_list, key=lambda f: abs(float(f) - mhz))
        by_target[int(target)].append((mhz, prio, item))

    plan: list[tuple[Path, Path]] = []
    for mhz in freq_list:
        dest_dir = base_generated_dir / f"freq_{mhz}MHz" / f"K{k_value}" / "Real"
        items = sorted(by_target[int(mhz)], key=lambda t: (t[0], t[1], t[2].name))
        if len(items) > num_samples:
            print(f"  warning: freq_{mhz}MHz has {len(items)} folder(s); using first {num_samples}")
            items = items[:num_samples]
        if len(items) < num_samples:
            print(f"  warning: freq_{mhz}MHz has only {len(items)}/{num_samples} folder(s)")
        for i, (sim_mhz, _, src) in enumerate(items):
            plan.append((src, dest_dir / f"Heatmap_real_{i}{_suffix(src.name)}"))
            print(f"  plan: {src.parent.parent.parent.name}/{src.name} "
                  f"({sim_mhz:.3f} MHz) -> freq_{mhz}MHz/Heatmap_real_{i}")

    temp_root = base_generated_dir / "_salvage_all"
    if temp_root.exists():
        shutil.rmtree(temp_root)
    temp_root.mkdir(parents=True, exist_ok=True)

    temp_paths: list[tuple[Path, Path]] = []
    for i, (src, final_dst) in enumerate(plan):
        tmp = temp_root / f"item_{i:03d}{_suffix(src.name)}"
        shutil.move(str(src), str(tmp))
        temp_paths.append((tmp, final_dst))

    for mhz in freq_list:
        dest_dir = base_generated_dir / f"freq_{mhz}MHz" / f"K{k_value}" / "Real"
        dest_dir.mkdir(parents=True, exist_ok=True)
        for child in dest_dir.iterdir():
            if child.is_dir() and not child.is_symlink():
                shutil.rmtree(child)
            else:
                child.unlink()

    for tmp, final_dst in temp_paths:
        final_dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(tmp), str(final_dst))

    if temp_root.exists():
        shutil.rmtree(temp_root, ignore_errors=True)

    print(f"\n✓ Salvage complete for {len(freq_list)} Real/ folder(s).")


def copy_sweep_peb() -> Path:
    """Copy the multifreq sweep PEB to ``PEB_COPY_DEST`` (Windows PEB folder)."""
    if not PEB_COPY_DEST:
        raise SystemExit("PEB_COPY_DEST is not set in run_multifreq_heatmap_sweep.py")
    peb = _PROJECT_ROOT / PEB_OUT_FILE
    target = copy_peb_to_folder(peb, PEB_COPY_DEST)
    print(f"Copied PEB → {target}")
    return target


def _live_sweep_config():
    """Read K/OUTPUT_ROOT from sweep module after pipeline config apply."""
    import scrap.generation.run_multifreq_heatmap_sweep as sweep  # noqa: E402

    return sweep.K_VALUE, sweep.OUTPUT_ROOT, sweep.NUM_SAMPLES, sweep.HEATMAP_ONLY_PEB


def run_move(freq_list: list[int] | None = None) -> None:
    """Route PI-* outputs from ECADStar into per-frequency Real/ folders."""
    import scrap.generation.run_multifreq_heatmap_sweep as sweep  # noqa: E402

    k_value, output_root, num_samples, heatmap_only = _live_sweep_config()
    source = SOURCE_EMC_DIR_OVERRIDE or SOURCE_EMC_DIR
    if freq_list is None:
        freq_list = sweep.exported_freq_mhz_list()
    print(f"\n=== Move PI outputs ({len(freq_list)} freq(s), K={k_value}) ===")
    print(f"  Source: {source}")
    print(f"  Target: {_PROJECT_ROOT / output_root}")
    print(
        f"  PEB mode: {'heatmap-only (1 PI/sample)' if heatmap_only else 'distribution+spectrum (2 PI/sample)'}"
    )
    move_sweep_pi_outputs(
        source_emc_dir=source,
        freq_list=freq_list,
        base_generated_dir=_PROJECT_ROOT / output_root,
        k_value=k_value,
        num_samples=num_samples,
    )


def run_compare() -> None:
    """Generated vs real heatmap plots for each frequency (fixed K)."""
    k_value, _, _, _ = _live_sweep_config()
    print(f"\n=== Compare heatmaps (K={k_value}) ===")
    compare_mod.main()


def run_report() -> None:
    """Build comparison_report.html + comparison_summary.md under OUTPUT_ROOT."""
    print(f"\n=== Build comparison report ===")
    build_multifreq_sweep_report()


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Multifreq sweep: copy PEB to Windows, move PI outputs, compare heatmaps",
    )
    ap.add_argument(
        "--copy-peb-only",
        action="store_true",
        help="Only copy the sweep .peb to PEB_COPY_DEST and exit",
    )
    ap.add_argument(
        "--copy-peb",
        action="store_true",
        help="Also copy .peb to PEB_COPY_DEST before move/compare",
    )
    ap.add_argument("--skip-move", action="store_true", help="Skip move step")
    ap.add_argument("--skip-compare", action="store_true", help="Skip compare step")
    ap.add_argument("--skip-report", action="store_true", help="Skip HTML/markdown report")
    ap.add_argument(
        "--salvage-misplaced",
        action="store_true",
        help="Re-group existing Heatmap_real_* / PI-* by MHz from .map files (skip ECAD move)",
    )
    ap.add_argument("--source-emc", default=None, help="Override SOURCE_EMC_DIR")
    ap.add_argument(
        "--mhz",
        nargs="+",
        type=float,
        default=None,
        metavar="MHZ",
        help="Override sweep MHz list (else sweep_frequencies_mhz.json / generate_summary.csv)",
    )
    args = ap.parse_args()

    global SOURCE_EMC_DIR_OVERRIDE
    if args.source_emc:
        SOURCE_EMC_DIR_OVERRIDE = args.source_emc

    os.chdir(_PROJECT_ROOT)

    if args.copy_peb_only:
        copy_sweep_peb()
        return

    if args.copy_peb:
        copy_sweep_peb()

    freq_override: list[int] | None = None
    if args.mhz is not None:
        freq_override = [int(round(m)) for m in args.mhz]
        write_sweep_freq_manifest([float(m) for m in freq_override])

    out_root = _PROJECT_ROOT / OUTPUT_ROOT
    if args.salvage_misplaced:
        fl = freq_override if freq_override is not None else exported_freq_mhz_list()
        print(f"\n=== Salvage misplaced heatmaps ({len(fl)} freq(s), K={K_VALUE}) ===")
        salvage_sweep_heatmap_outputs(
            freq_list=fl,
            base_generated_dir=out_root,
            k_value=K_VALUE,
            num_samples=NUM_SAMPLES,
        )
    elif not args.skip_move:
        run_move(freq_override)
    else:
        print("Skipping move step.")

    if not args.skip_compare:
        run_compare()
    else:
        print("Skipping compare step.")

    if not args.skip_report:
        if args.skip_compare:
            print("Warning: building report without compare — PNGs may be missing.")
        run_report()
    else:
        print("Skipping report step.")

    print("\n✓ Done.")


if __name__ == "__main__":
    main()
