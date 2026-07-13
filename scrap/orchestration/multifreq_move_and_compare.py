"""Multifreq Move, Compare, and Report.

Run: python scrap/multifreq_move_and_compare.py"""
from __future__ import annotations

import os
import re
import shutil
import sys
from pathlib import Path, PureWindowsPath

from repo_paths import REPO_ROOT as _PROJECT_ROOT, setup_path
setup_path()

from scrap.comparison import compare as compare_mod  # noqa: E402
from scrap.generation.run_multifreq_heatmap_sweep import (  # noqa: E402
    HEATMAP_ONLY_PEB,
    K_VALUE,
    NUM_SAMPLES,
    OUTPUT_ROOT,
    PEB_COPY_DEST,
    PEB_OUT_FILE,
    exported_freq_mhz_list,
    exported_k_values,
    write_sweep_freq_manifest,
)
from scrap.orchestration.build_comparison_report import build_multifreq_sweep_report  # noqa: E402
from scrap.peb_copy import copy_peb_to_folder  # noqa: E402

# =============================================================================
# CONFIGURATION — edit these before running
# =============================================================================

# ECADStar .emc directory containing PI-1, PI-2, … after batch simulate
SOURCE_EMC_DIR = r"C:\Users\muthusamy\Desktop\design\H-shape.emc"
SOURCE_EMC_DIR_OVERRIDE: str | None = None

MOVE = True
CLEAN_DEST_BEFORE_PASTE = True
PI_NAME_REGEX = re.compile(r"^PI-(\d+)(?:\..+)?$")

# COPY_PEB_ONLY = True  # only copy .peb to PEB_COPY_DEST and exit
COPY_PEB_ONLY = False

# SKIP_MOVE = True
SKIP_MOVE = False
# SKIP_COMPARE = True
SKIP_COMPARE = False
# SKIP_REPORT = True
SKIP_REPORT = False

# METRICS_ONLY = True  # only sim_compare_metrics.* (no move/plots/report)
METRICS_ONLY = False

# Run mode (single PI kind per sample; set by the sweep pipeline). Mutually exclusive:
#   "heatmap"   → moved PI folders renamed Heatmap_real_{i} (compare heatmap .map)
#   "impedance" → moved PI folders renamed Imp_Real{i}       (compare PIPinZ CSV)
RUN_MODE = "heatmap"

# Which comparisons to run + report (derived from RUN_MODE by the pipeline). Heatmap &
# impedance each get their own PNGs and a *separate* HTML report.
RUN_HEATMAP_COMPARE = True
RUN_IMPEDANCE_COMPARE = False

# SALVAGE_MISPLACED = True  # re-group misplaced Real/ by .map MHz
SALVAGE_MISPLACED = False

# MHZ_OVERRIDE = [10, 70, 120]  # override sweep MHz list (else manifest / CSV)
MHZ_OVERRIDE: list[float] | None = None

# Heatmap compare overrides (applied in run_compare; set by sweep pipeline)
_COMPARE_OVERRIDES: dict[str, object] = {}

# =============================================================================


def set_compare_overrides(**kwargs: object) -> None:
    """Push heatmap compare settings into compare.py before plotting."""
    global _COMPARE_OVERRIDES
    _COMPARE_OVERRIDES = dict(kwargs)


def _apply_compare_overrides() -> None:
    for key, val in _COMPARE_OVERRIDES.items():
        if hasattr(compare_mod, key):
            setattr(compare_mod, key, val)
        else:
            print(f"  Warning: compare.py has no attribute {key!r} — skipped")


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


def _impedance_run() -> bool:
    from scrap.generation.run_multifreq_heatmap_sweep import uses_flat_k_layout  # noqa: E402

    return str(RUN_MODE).strip().lower() == "impedance" or uses_flat_k_layout()


def _real_target_stem(i: int) -> str:
    """Renamed Real/ item stem compare.py expects for the active run mode."""
    return f"Imp_Real{i}" if _impedance_run() else f"Heatmap_real_{i}"


def _k_real_dir(base_generated_dir: Path, mhz: int, k: int) -> Path:
    """Generated sample Real/ parent for one (MHz, K) slot."""
    if _impedance_run():
        return base_generated_dir / f"K{k}" / "Real"
    return base_generated_dir / f"freq_{mhz}MHz" / f"K{k}" / "Real"


def _build_pi_slots(
    freq_list: list[int],
    *,
    k_values: list[int],
    num_samples: int,
) -> list[tuple[int, int, int, int]]:
    """Return (target_mhz, k, local_sample_i, global_pi_index) in PEB order.

    Heatmap (PI-Distribution): for mhz in freqs → for k → for sample.
    Impedance (PI-Spectrum): PI frequencies ignored — for k → for sample only.
    """
    slots: list[tuple[int, int, int, int]] = []
    global_idx = 0
    if _impedance_run():
        for k in k_values:
            for local_i in range(num_samples):
                slots.append((0, int(k), local_i, global_idx))
                global_idx += 1
        return slots
    for mhz in freq_list:
        for k in k_values:
            for local_i in range(num_samples):
                slots.append((int(mhz), int(k), local_i, global_idx))
                global_idx += 1
    return slots


def _rename_heatmap_outputs(dest_dir: Path, *, num_samples: int, sample_offset: int) -> None:
    """Rename PI-* in dest_dir for the active run mode (1 PI/sample).

    heatmap mode → ``Heatmap_real_{i}`` (.map), impedance mode → ``Imp_Real{i}`` (PIPinZ CSV).
    The folder structure is identical across modes; only the item name (and its file) differs.
    """
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
                target = dest_dir / f"{_real_target_stem(i)}{_suffix(item.name)}"
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
    k_values: list[int] | None = None,
    num_samples: int = NUM_SAMPLES,
) -> None:
    """Move PI-* from ECADStar into freq_*MHz/K{k}/Real/ for this sweep only."""
    ks = k_values if k_values is not None else exported_k_values(base_generated_dir)
    source_dir = _resolve_source_dir(source_emc_dir)
    if not source_dir.is_dir():
        raise SystemExit(f"Source .emc directory does not exist:\n  {source_dir}")

    pps = _pis_per_sample()
    slots = _build_pi_slots(freq_list, k_values=ks, num_samples=num_samples)
    max_pi = max(g_idx + 1 for *_, g_idx in slots) if pps == 1 else max(2 * g_idx + 2 for *_, g_idx in slots)
    print(f"  PI slot map: {len(slots)} entries → PI-1..PI-{max_pi} (K={ks[0]}–{ks[-1]}, {len(ks)} values)")

    pi_to_dest: dict[int, tuple[Path, int, int, int]] = {}
    dest_dirs: set[Path] = set()
    for mhz, k, local_i, g_idx in slots:
        dest_dir = _k_real_dir(base_generated_dir, mhz, k)
        dest_dirs.add(dest_dir)
        if pps == 1:
            pi_nums = [g_idx + 1]
        else:
            pi_nums = [2 * g_idx + 1, 2 * g_idx + 2]
        for pi_num in pi_nums:
            pi_to_dest[pi_num] = (dest_dir, local_i, mhz, k)

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
        dest_dir, local_i, mhz, k = pi_to_dest[pi_num]
        target = dest_dir / item.name
        action = "MOVE" if MOVE else "COPY"
        print(
            f"{action}: {item} -> {target}   "
            f"({'K' if _impedance_run() else f'freq_{mhz}MHz/'}K{k}, sample={local_i})"
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
    for _mhz, _k, _local_i, g_idx in slots:
        dest_dir = _k_real_dir(base_generated_dir, _mhz, _k)
        dir_offsets.setdefault(dest_dir, g_idx)
        dir_offsets[dest_dir] = min(dir_offsets[dest_dir], g_idx)

    for dest_dir, offset in sorted(dir_offsets.items(), key=lambda x: x[0].as_posix()):
        n = sum(
            1
            for mhz, k, _, _ in slots
            if dest_dir == _k_real_dir(base_generated_dir, mhz, k)
        )
        _rename_heatmap_outputs(dest_dir, num_samples=n, sample_offset=offset)

    k_label = "_".join(str(k) for k in ks)
    freq_note = "PI freqs ignored" if _impedance_run() else f"{len(freq_list)} freq(s)"
    print(
        f"\n✓ Moved {moved} PI item(s) into {len(dest_dirs)} Real/ folder(s) "
        f"({freq_note}, K={k_label}, {pps} PI/sample)."
    )


def salvage_pi_by_global_index(
    *,
    freq_list: list[int],
    base_generated_dir: Path,
    k_values: list[int] | None = None,
    num_samples: int = NUM_SAMPLES,
) -> None:
    """Re-route ``PI-N`` folders using global batch index (fixes wrong K×freq slot maps).

    Use after a move that assumed K values not actually present in the PEB (e.g. K=1 dropped
    from val split but manifest still listed K=1..20).
    """
    ks = k_values if k_values is not None else exported_k_values(base_generated_dir)
    slots = _build_pi_slots(freq_list, k_values=ks, num_samples=num_samples)
    pi_to_slot: dict[int, tuple[int, int, int]] = {
        g_idx + 1: (mhz, k, local_i) for mhz, k, local_i, g_idx in slots
    }

    sources: list[tuple[Path, int]] = []
    for real_dir in sorted(base_generated_dir.glob("freq_*MHz/K*/Real")):
        for item in sorted(real_dir.iterdir()):
            pi_num = _parse_pi_number(item.name)
            if pi_num is None or pi_num not in pi_to_slot:
                continue
            sources.append((item.resolve(), pi_num))

    if not sources:
        raise SystemExit(
            "Salvage-by-PI-index found no PI-N folders under freq_*MHz/K*/Real.\n"
            "  Expected PI-1..PI-N from ECADStar batch."
        )

    print(f"\n=== Salvage PI by global index ({len(sources)} PI folder(s), K={ks[0]}–{ks[-1]}) ===")
    temp_root = base_generated_dir / "_salvage_pi_index"
    if temp_root.exists():
        shutil.rmtree(temp_root)
    temp_root.mkdir(parents=True, exist_ok=True)

    staged: list[tuple[Path, Path]] = []
    for i, (src, pi_num) in enumerate(sorted(sources, key=lambda t: t[1])):
        mhz, k, local_i = pi_to_slot[pi_num]
        dest_dir = _k_real_dir(base_generated_dir, mhz, k)
        final_name = f"{_real_target_stem(local_i)}{_suffix(src.name)}"
        final_path = dest_dir / final_name
        tmp = temp_root / f"pi_{pi_num:04d}{_suffix(src.name)}"
        print(
            f"  PI-{pi_num} ({src.parent.parent.parent.name}/{src.parent.parent.name}) "
            f"→ freq_{mhz}MHz/K{k}/{final_name}"
        )
        shutil.move(str(src), str(tmp))
        staged.append((tmp, final_path))

    for mhz in freq_list:
        for k in ks:
            dest_dir = _k_real_dir(base_generated_dir, mhz, k)
            dest_dir.mkdir(parents=True, exist_ok=True)
            for child in dest_dir.iterdir():
                if child.is_dir() and not child.is_symlink():
                    shutil.rmtree(child)
                else:
                    child.unlink()

    for tmp, final_path in staged:
        final_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(tmp), str(final_path))

    if temp_root.exists():
        shutil.rmtree(temp_root, ignore_errors=True)

    print(f"\n✓ Salvage-by-PI-index complete ({len(staged)} item(s) → {len(freq_list) * len(ks)} slots).")


def salvage_sweep_heatmap_outputs(
    *,
    freq_list: list[int],
    base_generated_dir: Path,
    k_values: list[int] | None = None,
    num_samples: int = NUM_SAMPLES,
) -> None:
    """Re-group misplaced Heatmap_real_* / PI-* by MHz from .map files."""
    ks = k_values if k_values is not None else exported_k_values(base_generated_dir)
    seen: set[Path] = set()
    candidates: list[tuple[float, int, Path, int]] = []

    for k in ks:
        for real_dir in sorted(base_generated_dir.glob(f"freq_*MHz/K{k}/Real")):
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
                prio = (
                    0
                    if item.name.startswith("Heatmap_real_")
                    else 1
                    if _parse_pi_number(item.name)
                    else 2
                )
                candidates.append((mhz, prio, item, k))

    if not candidates:
        raise SystemExit("Salvage found no heatmap folders with Z_*MHz.map under the sweep Real/ dirs.")

    by_target: dict[tuple[int, int], list[tuple[float, int, Path]]] = {
        (int(m), k): [] for m in freq_list for k in ks
    }
    for mhz, prio, item, k in candidates:
        target = min(freq_list, key=lambda f: abs(float(f) - mhz))
        by_target[(int(target), k)].append((mhz, prio, item))

    plan: list[tuple[Path, Path]] = []
    for mhz in freq_list:
        for k in ks:
            dest_dir = base_generated_dir / f"freq_{mhz}MHz" / f"K{k}" / "Real"
            items = sorted(by_target[(int(mhz), k)], key=lambda t: (t[0], t[1], t[2].name))
            if len(items) > num_samples:
                print(
                    f"  warning: freq_{mhz}MHz/K{k} has {len(items)} folder(s); "
                    f"using first {num_samples}"
                )
                items = items[:num_samples]
            if len(items) < num_samples:
                print(
                    f"  warning: freq_{mhz}MHz/K{k} has only {len(items)}/{num_samples} folder(s)"
                )
            for i, (sim_mhz, _, src) in enumerate(items):
                plan.append((src, dest_dir / f"{_real_target_stem(i)}{_suffix(src.name)}"))
                print(
                    f"  plan: {src.parent.parent.parent.name}/{src.name} "
                    f"({sim_mhz:.3f} MHz) -> freq_{mhz}MHz/K{k}/Heatmap_real_{i}"
                )

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
        for k in ks:
            dest_dir = base_generated_dir / f"freq_{mhz}MHz" / f"K{k}" / "Real"
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

    print(f"\n✓ Salvage complete for {len(freq_list) * len(ks)} Real/ folder(s).")


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

    return (
        exported_k_values(sweep.OUTPUT_ROOT),
        sweep.OUTPUT_ROOT,
        sweep.NUM_SAMPLES,
        sweep.HEATMAP_ONLY_PEB,
    )


def run_move(freq_list: list[int] | None = None) -> None:
    """Route PI-* outputs from ECADStar into per-frequency Real/ folders."""
    import scrap.generation.run_multifreq_heatmap_sweep as sweep  # noqa: E402

    k_values, output_root, num_samples, heatmap_only = _live_sweep_config()
    source = SOURCE_EMC_DIR_OVERRIDE or SOURCE_EMC_DIR
    if freq_list is None:
        freq_list = sweep.exported_freq_mhz_list()
    if _impedance_run():
        freq_list = []
    k_label = "_".join(str(k) for k in k_values)
    freq_note = "PI freqs ignored" if _impedance_run() else f"{len(freq_list)} freq(s)"
    print(f"\n=== Move PI outputs ({freq_note}, K={k_label}) ===")
    print(f"  Source: {source}")
    print(f"  Target: {_PROJECT_ROOT / output_root}")
    print(
        f"  PEB mode: {'heatmap-only (1 PI/sample)' if heatmap_only else 'distribution+spectrum (2 PI/sample)'}"
    )
    move_sweep_pi_outputs(
        source_emc_dir=source,
        freq_list=freq_list,
        base_generated_dir=_PROJECT_ROOT / output_root,
        k_values=k_values,
        num_samples=num_samples,
    )


def run_sim_metrics_only() -> Path:
    """Simulated-real vs generated metrics only (no comparison PNGs)."""
    k_values, _, _, _ = _live_sweep_config()
    k_label = "_".join(str(k) for k in k_values)
    print(f"\n=== Sim metrics only (K={k_label}) ===")
    _apply_compare_overrides()
    return compare_mod.run_sim_metrics_only()


def run_compare() -> None:
    """Generated vs real heatmap and/or impedance plots for each frequency and K."""
    k_values, _, _, _ = _live_sweep_config()
    k_label = "_".join(str(k) for k in k_values)
    kinds = [
        name
        for name, on in (("heatmap", RUN_HEATMAP_COMPARE), ("impedance", RUN_IMPEDANCE_COMPARE))
        if on
    ]
    print(f"\n=== Compare {', '.join(kinds) or 'nothing'} (K={k_label}) ===")
    _apply_compare_overrides()
    compare_mod.main()


def run_report() -> None:
    """Build separate heatmap / impedance comparison HTML + markdown under OUTPUT_ROOT."""
    print(f"\n=== Build comparison report(s) ===")
    build_multifreq_sweep_report(
        run_heatmap=RUN_HEATMAP_COMPARE,
        run_impedance=RUN_IMPEDANCE_COMPARE,
    )


def main() -> None:
    os.chdir(_PROJECT_ROOT)

    if COPY_PEB_ONLY:
        copy_sweep_peb()
        return

    if METRICS_ONLY:
        run_sim_metrics_only()
        print("\n✓ Done.")
        return

    freq_override: list[int] | None = None
    if MHZ_OVERRIDE is not None:
        freq_override = [int(round(m)) for m in MHZ_OVERRIDE]
        write_sweep_freq_manifest([float(m) for m in freq_override])

    out_root = _PROJECT_ROOT / OUTPUT_ROOT
    if SALVAGE_MISPLACED:
        fl = freq_override if freq_override is not None else exported_freq_mhz_list()
        ks = exported_k_values(out_root)
        k_label = "_".join(str(k) for k in ks)
        print(f"\n=== Salvage misplaced PI outputs ({len(fl)} freq(s), K={k_label}) ===")
        salvage_pi_by_global_index(
            freq_list=fl,
            base_generated_dir=out_root,
            k_values=ks,
            num_samples=NUM_SAMPLES,
        )
    elif not SKIP_MOVE:
        run_move(freq_override)
    else:
        print("Skipping move step.")

    if not SKIP_COMPARE:
        run_compare()
    else:
        print("Skipping compare step.")

    if not SKIP_REPORT:
        if SKIP_COMPARE:
            print("Warning: building report without compare — PNGs may be missing.")
        run_report()
    else:
        print("Skipping report step.")

    print("\n✓ Done.")


if __name__ == "__main__":
    main()
