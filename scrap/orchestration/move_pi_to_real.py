"""Move ECADStar PI Outputs to Real/ Folders.

Run: python scrap/move_pi_to_real.py  (or import move_pi_outputs_for_k / move_pi_outputs_for_k_range)"""
from __future__ import annotations

import re
import shutil
import sys
from bisect import bisect_right
from pathlib import Path, PureWindowsPath

from repo_paths import REPO_ROOT as _PROJECT_ROOT, setup_path
setup_path()

# =============================================================================
# CONFIGURATION — edit these before running: python scrap/move_pi_to_real.py
# =============================================================================
# "run_all_k" | "multifreq_heatmap_sweep"
WORKFLOW = "run_all_k"

SOURCE_EMC_DIR = r"C:\Users\muthusamy\Desktop\design\H-shape.emc"

# Which generated folder to target when running this file as a script.
# The callable function `move_pi_outputs_for_k()` takes this as an argument.
K_VALUE = 10

# If you generated one combined .peb (K=1..52 in order), set this True and
# configure the range below. For multifreq_heatmap_sweep leave False (single fixed K).
PROCESS_K_RANGE = True

# Override which frequencies to process (run_all_k only; None = use run_all_k list)
PI_FREQ_OVERRIDE: int | list[int] | None = None

if WORKFLOW == "multifreq_heatmap_sweep":
    from scrap.generation.run_multifreq_heatmap_sweep import (  # noqa: E402
        HEATMAP_ONLY_PEB,
        K_VALUE as _SWEEP_K,
        NUM_SAMPLES as _RUN_NUM_SAMPLES,
        OUTPUT_ROOT as _OUTPUT_ROOT,
        exported_freq_mhz_list,
    )

    K_MIN = K_MAX = _SWEEP_K
    BASE_GENERATED_DIR = _OUTPUT_ROOT
    HEATMAP_ONLY_PI = HEATMAP_ONLY_PEB
    _FREQ_LIST: list[int | None] = []  # use exported_freq_mhz_list() at move time
elif WORKFLOW == "run_all_k":
    from scrap.generation.run_all_k import (  # noqa: E402
        K_MIN,
        K_MAX,
        NUM_SAMPLES as _RUN_NUM_SAMPLES,
        OUTPUT_ROOT as _OUTPUT_ROOT,
        PI_FREQ_MHZ as _RUN_PI_FREQ_MHZ,
    )

    BASE_GENERATED_DIR = _OUTPUT_ROOT
    HEATMAP_ONLY_PI = False
    _freq_src = PI_FREQ_OVERRIDE if PI_FREQ_OVERRIDE is not None else _RUN_PI_FREQ_MHZ
    if _freq_src is None:
        _FREQ_LIST = [None]
    elif isinstance(_freq_src, list):
        _FREQ_LIST = list(_freq_src)
    else:
        _FREQ_LIST = [int(_freq_src)]
else:
    raise SystemExit(f"Unknown WORKFLOW={WORKFLOW!r}; use 'run_all_k' or 'multifreq_heatmap_sweep'")


def _base_dir_for_freq(mhz: int | None) -> str:
    """Return the effective base directory for a given PI frequency."""
    if mhz is None:
        return BASE_GENERATED_DIR
    return f"{BASE_GENERATED_DIR}/freq_{mhz}MHz"

MOVE = True  # True = cut/move, False = copy
OVERWRITE = False  # if False, auto-rename on name collision
DRY_RUN = False

# If True, delete everything currently in DEST_DIR before pasting new PI-* items.
# This is skipped when DRY_RUN=True.
CLEAN_DEST_BEFORE_PASTE = True

SEARCH_RECURSIVE = False  # True = search all descendants, False = only direct children

# If True, only move PI outputs needed for the current run.
LIMIT_TO_EXPECTED_PI = True

# After moving, rename PI outputs to Heatmap_real_{i} (and Imp_Real{i} if not heatmap-only).
RENAME_PI_TO_MATCH_SAMPLES = True

# Matches: PI-1, PI-2, ... optionally with extension (PI-1.csv, PI-2.arv, etc.)
PI_NAME_REGEX = r"^PI-\d+(?:\..+)?$"


def _parse_pi_number(name: str) -> int | None:
    m = re.match(r"^PI-(\d+)(?:\..+)?$", name)
    if not m:
        return None
    return int(m.group(1))


def _suffix(name: str) -> str:
    """Return extension suffix (including dot) if present, else empty string."""
    # If this is a directory named PI-1, suffix is empty.
    p = Path(name)
    return p.suffix


# When set by move_pi_outputs_for_k_list(...), overrides WORKFLOW defaults.
_PIS_PER_SAMPLE_OVERRIDE: int | None = None
_PI_OUTPUT_KIND_OVERRIDE: str | None = None  # "heatmap" | "impedance" when pps == 1


def _pis_per_sample() -> int:
    """1 for single PI group per sample; 2 for distribution + spectrum."""
    if _PIS_PER_SAMPLE_OVERRIDE is not None:
        return int(_PIS_PER_SAMPLE_OVERRIDE)
    return 1 if HEATMAP_ONLY_PI else 2


def _single_pi_output_kind() -> str:
    """Rename target for one PI per sample: Heatmap_real_* or Imp_Real*."""
    if _PI_OUTPUT_KIND_OVERRIDE in ("heatmap", "impedance"):
        return _PI_OUTPUT_KIND_OVERRIDE
    return "heatmap" if HEATMAP_ONLY_PI else "impedance"


def _infer_num_samples(generated_k_dir: Path) -> int:
    """Infer N from consecutive data_sample_0..data_sample_{N-1} folders."""
    sample_dirs = [p for p in generated_k_dir.glob("data_sample_*") if p.is_dir()]
    if not sample_dirs:
        raise SystemExit(f"No data_sample_* folders found in: {generated_k_dir}")

    def _idx(p: Path) -> int | None:
        m = re.match(r"^data_sample_(\d+)$", p.name)
        return int(m.group(1)) if m else None

    indices = sorted(i for i in (_idx(p) for p in sample_dirs) if i is not None)
    if not indices:
        raise SystemExit(f"Found data_sample_* entries but none matched expected naming in: {generated_k_dir}")

    expected = list(range(0, max(indices) + 1))
    if indices != expected:
        raise SystemExit(
            "data_sample_* folders are not consecutive starting at 0.\n"
            f"  Found:    {indices[:20]}{' ...' if len(indices) > 20 else ''}\n"
            f"  Expected: {expected[:20]}{' ...' if len(expected) > 20 else ''}"
        )

    return len(expected)


def _rename_pi_outputs(dest_dir: Path, num_samples: int) -> None:
    """Rename PI outputs in dest_dir to match sample ordering."""
    by_num: dict[int, list[Path]] = {}
    for item in dest_dir.iterdir():
        pi_num = _parse_pi_number(item.name)
        if pi_num is None:
            continue
        by_num.setdefault(pi_num, []).append(item)

    pps = _pis_per_sample()
    expected_nums = list(range(1, pps * num_samples + 1))
    missing = [n for n in expected_nums if n not in by_num]
    if missing:
        raise SystemExit(
            "Missing expected PI outputs in DEST_DIR.\n"
            f"  Missing PI numbers: {missing[:20]}{' ...' if len(missing) > 20 else ''}\n"
            f"  DEST_DIR: {dest_dir}"
        )

    for i in range(num_samples):
        if pps == 1:
            pi_num = i + 1
            kind = _single_pi_output_kind()
            prefix = "Heatmap_real_" if kind == "heatmap" else "Imp_Real"
            for item in sorted(by_num[pi_num], key=lambda p: p.name):
                new_name = f"{prefix}{i}{_suffix(item.name)}"
                target = dest_dir / new_name
                if target.exists():
                    raise SystemExit(f"Target already exists while renaming: {target}")
                item.rename(target)
        else:
            pi_heat = 2 * i + 1
            pi_imp = 2 * i + 2
            for item in sorted(by_num[pi_heat], key=lambda p: p.name):
                new_name = f"Heatmap_real_{i}{_suffix(item.name)}"
                target = dest_dir / new_name
                if target.exists():
                    raise SystemExit(f"Target already exists while renaming: {target}")
                item.rename(target)
            for item in sorted(by_num[pi_imp], key=lambda p: p.name):
                new_name = f"Imp_Real{i}{_suffix(item.name)}"
                target = dest_dir / new_name
                if target.exists():
                    raise SystemExit(f"Target already exists while renaming: {target}")
                item.rename(target)


def _rename_pi_outputs_with_offset(dest_dir: Path, *, num_samples: int, sample_offset: int) -> None:
    """Rename PI outputs when PI numbering is global across a combined PEB."""
    by_num: dict[int, list[Path]] = {}
    for item in dest_dir.iterdir():
        pi_num = _parse_pi_number(item.name)
        if pi_num is None:
            continue
        by_num.setdefault(pi_num, []).append(item)

    pps = _pis_per_sample()
    expected = []
    for i in range(num_samples):
        j = sample_offset + i
        if pps == 1:
            expected.append(j + 1)
        else:
            expected.extend([2 * j + 1, 2 * j + 2])

    missing = [n for n in expected if n not in by_num]
    if missing:
        raise SystemExit(
            "Missing expected PI outputs in DEST_DIR.\n"
            f"  Missing PI numbers: {missing[:20]}{' ...' if len(missing) > 20 else ''}\n"
            f"  DEST_DIR: {dest_dir}"
        )

    for i in range(num_samples):
        j = sample_offset + i
        if pps == 1:
            pi_num = j + 1
            kind = _single_pi_output_kind()
            prefix = "Heatmap_real_" if kind == "heatmap" else "Imp_Real"
            for item in sorted(by_num[pi_num], key=lambda p: p.name):
                new_name = f"{prefix}{i}{_suffix(item.name)}"
                target = dest_dir / new_name
                if target.exists():
                    raise SystemExit(f"Target already exists while renaming: {target}")
                item.rename(target)
        else:
            pi_heat = 2 * j + 1
            pi_imp = 2 * j + 2
            for item in sorted(by_num[pi_heat], key=lambda p: p.name):
                new_name = f"Heatmap_real_{i}{_suffix(item.name)}"
                target = dest_dir / new_name
                if target.exists():
                    raise SystemExit(f"Target already exists while renaming: {target}")
                item.rename(target)
            for item in sorted(by_num[pi_imp], key=lambda p: p.name):
                new_name = f"Imp_Real{i}{_suffix(item.name)}"
                target = dest_dir / new_name
                if target.exists():
                    raise SystemExit(f"Target already exists while renaming: {target}")
                item.rename(target)


def _count_pi_names(directory: Path, *, recursive: bool) -> int:
    pattern = re.compile(PI_NAME_REGEX)
    n = 0
    if recursive:
        for item in directory.rglob("*"):
            if pattern.match(item.name):
                n += 1
    else:
        for item in directory.iterdir():
            if pattern.match(item.name):
                n += 1
    return n


def resolve_pi_source_dir(path_str: str, *, recursive: bool = True) -> Path:
    """Pick the directory that actually contains PI-* outputs (top-level or subfolder)."""
    root = _resolve_source_dir(path_str)
    if not root.exists():
        raise SystemExit(
            f"Source path does not exist:\n  {root}\n"
            "Set LATENT_SOURCE_EMC_DIR to the folder that contains PI-1, PI-2, … after ECADStar batch."
        )
    if not root.is_dir():
        raise SystemExit(f"SOURCE path must be a directory: {root}")

    if _count_pi_names(root, recursive=False) > 0:
        return root

    for child in sorted(p for p in root.iterdir() if p.is_dir()):
        if _count_pi_names(child, recursive=recursive) > 0:
            print(f"Using PI subfolder: {child}")
            return child

    if recursive and _count_pi_names(root, recursive=True) > 0:
        print(f"Using recursive PI search under: {root}")
        return root

    sample = sorted(root.iterdir())[:15]
    listing = "\n  ".join(p.name for p in sample)
    extra = f"\n  … ({len(list(root.iterdir()))} entries total)" if len(list(root.iterdir())) > 15 else ""
    raise SystemExit(
        "No PI-* items found under source path.\n"
        f"  Path: {root}\n"
        f"  Top-level entries:\n  {listing}{extra}\n\n"
        "Run ECADStar batch on latent_run.peb first, then point LATENT_SOURCE_EMC_DIR "
        "at the folder that contains PI-1, PI-2, … (often inside the .emc project folder)."
    )


def _resolve_source_dir(path_str: str) -> Path:
    """Resolve a potentially Windows-style path to a usable Path on this machine."""
    # If it's a Windows path (e.g. C:\Users\...), try WSL mapping first.
    if re.match(r"^[A-Za-z]:\\", path_str):
        win = PureWindowsPath(path_str)
        drive = win.drive.rstrip(":").lower()
        wsl = Path("/mnt") / drive
        # win.parts is like ('C:\\', 'Users', ...)
        wsl_path = wsl.joinpath(*win.parts[1:])
        if wsl_path.exists():
            return wsl_path
        # Fall back to the raw string (useful if running on Windows).
        return Path(path_str)

    return Path(path_str)


def _iter_candidates(source_dir: Path):
    if SEARCH_RECURSIVE:
        yield from source_dir.rglob("*")
    else:
        yield from source_dir.iterdir()


def _unique_dest_path(dest_dir: Path, name: str) -> Path:
    """Create a non-colliding destination path by appending _<n> if needed."""
    base = dest_dir / name
    if not base.exists():
        return base

    stem, suffix = base.stem, base.suffix
    for i in range(1, 10_000):
        candidate = dest_dir / f"{stem}_{i}{suffix}"
        if not candidate.exists():
            return candidate

    raise RuntimeError(f"Could not find free destination name for {name}")


def move_pi_outputs_for_k(
    k_value: int,
    *,
    source_emc_dir: str = SOURCE_EMC_DIR,
    base_generated_dir: str | Path = BASE_GENERATED_DIR,
) -> Path:
    """Move/copy PI-* items into `.../K{k_value}/Real/` and rename them per sample.

    Args:
        k_value: Which generated folder to target, e.g. 5 -> .../K5/
        source_emc_dir: Path to the `.emc` directory containing PI-* outputs
        base_generated_dir: Parent folder that contains `K{n}/` directories

    Returns:
        Path to the destination Real/ directory.
    """
    source_dir = _resolve_source_dir(source_emc_dir)
    generated_k_dir = Path(base_generated_dir) / f"K{k_value}"
    dest_dir = generated_k_dir / "Real"

    if not generated_k_dir.exists() or not generated_k_dir.is_dir():
        raise SystemExit(f"Generated K directory not found: {generated_k_dir}")

    # Ensure destination exists (create Real/ if missing)
    if dest_dir.exists() and not dest_dir.is_dir():
        raise SystemExit(f"DEST_DIR exists but is not a directory: {dest_dir}")
    dest_dir.mkdir(parents=True, exist_ok=True)

    num_samples = _infer_num_samples(generated_k_dir)
    expected_pi_numbers = set(range(1, _pis_per_sample() * num_samples + 1))

    if not source_dir.exists():
        raise SystemExit(
            "Source .emc directory does not exist:\n"
            f"  {source_dir}\n\n"
            "If this is a Windows path, run this on Windows/WSL or update SOURCE_EMC_DIR to a Linux-accessible path."
        )

    if not source_dir.is_dir():
        raise SystemExit(f"SOURCE_EMC_DIR must be a directory, got: {source_dir}")

    # Optional cleanup: ensure Real/ is empty before pasting new results.
    if CLEAN_DEST_BEFORE_PASTE and not DRY_RUN:
        # Safety guard: never allow deleting '/' or an empty path.
        dest_resolved = dest_dir.resolve()
        if str(dest_resolved) in ("/", ""):
            raise SystemExit(f"Refusing to clean unsafe DEST_DIR: {dest_dir}")

        for child in dest_dir.iterdir():
            if child.is_dir() and not child.is_symlink():
                shutil.rmtree(child)
            else:
                child.unlink()

    pattern = re.compile(PI_NAME_REGEX)

    moved = 0
    for item in _iter_candidates(source_dir):
        if not item.exists():
            continue
        if not pattern.match(item.name):
            continue

        pi_num = _parse_pi_number(item.name)
        if LIMIT_TO_EXPECTED_PI and (pi_num is None or pi_num not in expected_pi_numbers):
            continue

        target = dest_dir / item.name
        if target.exists() and not OVERWRITE:
            target = _unique_dest_path(dest_dir, item.name)

        action = "MOVE" if MOVE else "COPY"
        print(f"{action}: {item} -> {target}")

        if DRY_RUN:
            moved += 1
            continue

        if MOVE:
            if OVERWRITE and target.exists():
                if target.is_dir():
                    shutil.rmtree(target)
                else:
                    target.unlink()
            shutil.move(str(item), str(target))
        else:
            if item.is_dir():
                if OVERWRITE and target.exists():
                    shutil.rmtree(target)
                shutil.copytree(item, target)
            else:
                if OVERWRITE and target.exists():
                    target.unlink()
                shutil.copy2(item, target)

        moved += 1

    if moved == 0:
        print(f"\n⚠ No PI-* files found in source – skipping rename for: {dest_dir}")
    elif RENAME_PI_TO_MATCH_SAMPLES and not DRY_RUN:
        _rename_pi_outputs(dest_dir, num_samples=num_samples)

    print(f"\n✓ Done. {moved} item(s) {'moved' if MOVE else 'copied'} into: {dest_dir}")
    return dest_dir


def move_pi_outputs_for_k_list(
    k_values: list[int],
    *,
    source_emc_dir: str = SOURCE_EMC_DIR,
    base_generated_dir: str | Path = BASE_GENERATED_DIR,
    pis_per_sample: int | None = None,
    pi_output_kind: str | None = None,
) -> list[Path]:
    """Split PI-* outputs across listed K folders (combined-PEB order = sorted K list).

    Matches `latent_run.peb` / `generate_peb_for_run` when K folders are only the
    exported K values in increasing order (not necessarily every integer in between).

    pis_per_sample : 1 = one PI group per sample; 2 = distribution + spectrum.
    pi_output_kind : when pis_per_sample==1, rename PI-* to ``Heatmap_real_*`` or
        ``Imp_Real*`` (``"heatmap"`` | ``"impedance"``). None = workflow default.
    """
    global _PIS_PER_SAMPLE_OVERRIDE, _PI_OUTPUT_KIND_OVERRIDE
    if pis_per_sample is not None and pis_per_sample not in (1, 2):
        raise SystemExit("pis_per_sample must be 1 or 2")
    if pi_output_kind is not None and pi_output_kind not in ("heatmap", "impedance"):
        raise SystemExit('pi_output_kind must be "heatmap" or "impedance"')
    old_pps_override = _PIS_PER_SAMPLE_OVERRIDE
    old_kind_override = _PI_OUTPUT_KIND_OVERRIDE
    if pis_per_sample is not None:
        _PIS_PER_SAMPLE_OVERRIDE = int(pis_per_sample)
    if pi_output_kind is not None:
        _PI_OUTPUT_KIND_OVERRIDE = pi_output_kind
    try:
        return _move_pi_outputs_for_k_list_impl(
            k_values,
            source_emc_dir=source_emc_dir,
            base_generated_dir=base_generated_dir,
        )
    finally:
        _PIS_PER_SAMPLE_OVERRIDE = old_pps_override
        _PI_OUTPUT_KIND_OVERRIDE = old_kind_override


def _move_pi_outputs_for_k_list_impl(
    k_values: list[int],
    *,
    source_emc_dir: str,
    base_generated_dir: str | Path,
) -> list[Path]:
    ks = sorted({int(k) for k in k_values})
    if not ks or not all(0 <= k <= 52 for k in ks):
        raise SystemExit("k_values must be non-empty integers in [0, 52]")

    source_dir = resolve_pi_source_dir(source_emc_dir, recursive=SEARCH_RECURSIVE)
    base_dir = Path(base_generated_dir)
    per_k_num_samples: list[int] = []
    per_k_dest: list[Path] = []
    for k in ks:
        generated_k_dir = base_dir / f"K{k}"
        if not generated_k_dir.exists() or not generated_k_dir.is_dir():
            raise SystemExit(f"Generated K directory not found: {generated_k_dir}")

        num_samples = _infer_num_samples(generated_k_dir)
        per_k_num_samples.append(num_samples)

        dest_dir = generated_k_dir / "Real"
        if dest_dir.exists() and not dest_dir.is_dir():
            raise SystemExit(f"DEST_DIR exists but is not a directory: {dest_dir}")
        dest_dir.mkdir(parents=True, exist_ok=True)
        per_k_dest.append(dest_dir)

    # Optional cleanup: clear each Real/ folder.
    if CLEAN_DEST_BEFORE_PASTE and not DRY_RUN:
        for dest_dir in per_k_dest:
            dest_resolved = dest_dir.resolve()
            if str(dest_resolved) in ("/", ""):
                raise SystemExit(f"Refusing to clean unsafe DEST_DIR: {dest_dir}")
            for child in dest_dir.iterdir():
                if child.is_dir() and not child.is_symlink():
                    shutil.rmtree(child)
                else:
                    child.unlink()

    pps = _pis_per_sample()
    cumulative = [0]
    for n in per_k_num_samples:
        cumulative.append(cumulative[-1] + n)

    pattern = re.compile(PI_NAME_REGEX)
    moved = 0

    for item in _iter_candidates(source_dir):
        if not item.exists():
            continue
        if not pattern.match(item.name):
            continue

        pi_num = _parse_pi_number(item.name)
        if pi_num is None:
            continue

        sample_index = (pi_num - 1) // pps
        # Find block idx such that cumulative[idx] <= sample_index < cumulative[idx+1]
        idx = bisect_right(cumulative, sample_index) - 1
        if idx < 0 or idx >= len(ks):
            # PI-* outside the expected K-range; ignore.
            continue

        k = ks[idx]
        block_start = cumulative[idx]
        local_i = sample_index - block_start
        if not (0 <= local_i < per_k_num_samples[idx]):
            continue

        if LIMIT_TO_EXPECTED_PI:
            start_pi = pps * block_start + 1
            end_pi = pps * (block_start + per_k_num_samples[idx])
            if not (start_pi <= pi_num <= end_pi):
                continue

        dest_dir = per_k_dest[idx]
        target = dest_dir / item.name
        if target.exists() and not OVERWRITE:
            target = _unique_dest_path(dest_dir, item.name)

        action = "MOVE" if MOVE else "COPY"
        print(f"{action}: {item} -> {target}   (K={k}, sample={local_i})")

        if DRY_RUN:
            moved += 1
            continue

        if MOVE:
            if OVERWRITE and target.exists():
                if target.is_dir():
                    shutil.rmtree(target)
                else:
                    target.unlink()
            shutil.move(str(item), str(target))
        else:
            if item.is_dir():
                if OVERWRITE and target.exists():
                    shutil.rmtree(target)
                shutil.copytree(item, target)
            else:
                if OVERWRITE and target.exists():
                    target.unlink()
                shutil.copy2(item, target)

        moved += 1

    # Rename per folder with the correct global offset.
    if moved == 0:
        print(f"\n⚠ No PI-* files found in source – skipping rename for K={ks}.")
    elif RENAME_PI_TO_MATCH_SAMPLES and not DRY_RUN:
        kind = _single_pi_output_kind()
        print(f"\nRenaming PI-* in Real/ (pps={pps}, output={kind!r}) …")
        for idx, dest_dir in enumerate(per_k_dest):
            try:
                _rename_pi_outputs_with_offset(
                    dest_dir,
                    num_samples=per_k_num_samples[idx],
                    sample_offset=cumulative[idx],
                )
            except SystemExit as e:
                print(f"  ⚠ K{ks[idx]} rename: {e}")
                _fallback_rename_pi_to_impedance(dest_dir, per_k_num_samples[idx])

    print(
        f"\n✓ Done. {moved} item(s) {'moved' if MOVE else 'copied'} into {len(per_k_dest)} Real/ folders "
        f"(K={ks})."
    )
    return per_k_dest


def ensure_real_impedance_renamed(dest_dir: Path, num_samples: int) -> None:
    """Ensure PI-Spectrum outputs under Real/ are named Imp_Real{i}* (for compare.py)."""
    _fallback_rename_pi_to_impedance(dest_dir, num_samples)


def _fallback_rename_pi_to_impedance(dest_dir: Path, num_samples: int) -> None:
    """If offset-based rename fails, map sorted PI-* → Imp_Real0.. for single-sample K folders."""
    if any(dest_dir.glob("Imp_Real*")):
        return
    pi_items = sorted(
        (p for p in dest_dir.iterdir() if _parse_pi_number(p.name) is not None),
        key=lambda p: _parse_pi_number(p.name) or 0,
    )
    for i, src in enumerate(pi_items[:num_samples]):
        dst = dest_dir / f"Imp_Real{i}{_suffix(src.name)}"
        if dst.exists():
            continue
        src.rename(dst)
        print(f"  fallback rename: {src.name} → {dst.name}")


def move_pi_outputs_for_k_range(
    k_min: int,
    k_max: int,
    *,
    source_emc_dir: str = SOURCE_EMC_DIR,
    base_generated_dir: str | Path = BASE_GENERATED_DIR,
) -> list[Path]:
    """Split PI-* outputs across K{k_min}..K{k_max} assuming one combined-PEB run."""
    if not (0 <= k_min <= 52 and 0 <= k_max <= 52 and k_min <= k_max):
        raise SystemExit("Expected 0 <= k_min <= k_max <= 52")
    return move_pi_outputs_for_k_list(
        list(range(k_min, k_max + 1)),
        source_emc_dir=source_emc_dir,
        base_generated_dir=base_generated_dir,
    )


def main() -> None:
    multi_freq = _FREQ_LIST != [None]
    if WORKFLOW == "multifreq_heatmap_sweep" or (multi_freq and PROCESS_K_RANGE):
        move_pi_outputs_multi_freq(
            k_min=K_MIN,
            k_max=K_MAX,
            num_samples=_RUN_NUM_SAMPLES,
        )
        return

    for mhz in _FREQ_LIST:
        base_dir = _base_dir_for_freq(mhz)
        label = f"freq_{mhz}MHz" if mhz is not None else "(no freq subfolder)"
        print(f"\n{'='*60}")
        print(f"Processing {label}  →  {base_dir}")
        print(f"{'='*60}")
        if PROCESS_K_RANGE:
            move_pi_outputs_for_k_range(K_MIN, K_MAX, base_generated_dir=base_dir)
        else:
            move_pi_outputs_for_k(K_VALUE, base_generated_dir=base_dir)


def _extract_map_mhz(folder: Path) -> float | None:
    """Read simulated MHz from ECADStar Z_*MHz.map under a PI or Heatmap_real folder."""
    for path in sorted(folder.rglob("Z_*MHz.map")):
        m = re.search(r"Z_(\d+(?:\.\d+)?)MHz", path.name, re.IGNORECASE)
        if m:
            return float(m.group(1))
    return None


def _salvage_item_priority(name: str) -> int:
    if name.startswith("Heatmap_real_"):
        return 0
    if _parse_pi_number(name) is not None:
        return 1
    return 2


def salvage_heatmap_real_by_map_mhz(
    *,
    freq_list: list[int],
    k_value: int,
    num_samples: int,
    base_generated_dir: str | Path = BASE_GENERATED_DIR,
) -> list[Path]:
    """Re-group misplaced Heatmap_real_* / PI-* folders by MHz read from .map files.

    Use when a prior move assumed 2 PI outputs per sample but the PEB was heatmap-only.
    """
    base_dir = Path(base_generated_dir)
    seen: set[Path] = set()
    candidates: list[tuple[float, int, Path]] = []

    for real_dir in sorted(base_dir.glob(f"freq_*MHz/K{k_value}/Real")):
        if not real_dir.is_dir():
            continue
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
            if item.name.startswith("Imp_Real"):
                dup = any(
                    not x.name.startswith("Imp_Real")
                    and _extract_map_mhz(x) == mhz
                    for x in folder_items
                )
                if dup:
                    continue
            seen.add(resolved)
            candidates.append((mhz, _salvage_item_priority(item.name), item))

    if not candidates:
        raise SystemExit(
            "Salvage found no Heatmap_real_* or PI-* folders with Z_*MHz.map under the sweep Real/ dirs."
        )

    by_target: dict[int, list[tuple[float, int, Path]]] = {int(mhz): [] for mhz in freq_list}
    for mhz, prio, item in candidates:
        target = min(freq_list, key=lambda f: abs(float(f) - mhz))
        by_target[int(target)].append((mhz, prio, item))

    plan: list[tuple[Path, Path]] = []
    updated: list[Path] = []
    for mhz in freq_list:
        dest_dir = base_dir / f"freq_{mhz}MHz" / f"K{k_value}" / "Real"
        items = sorted(by_target[int(mhz)], key=lambda t: (t[0], t[1], t[2].name))
        if len(items) > num_samples:
            print(
                f"  warning: freq_{mhz}MHz has {len(items)} heatmap folder(s) "
                f"(expected {num_samples}); using first {num_samples}"
            )
            items = items[:num_samples]
        if len(items) < num_samples:
            print(
                f"  warning: freq_{mhz}MHz has only {len(items)}/{num_samples} "
                "heatmap folder(s) after salvage"
            )
        for i, (sim_mhz, _, src) in enumerate(items):
            plan.append((src, dest_dir / f"Heatmap_real_{i}{_suffix(src.name)}"))
            print(f"  plan: {src.parent.parent.parent.name}/{src.name} "
                  f"({sim_mhz:.3f} MHz) -> freq_{mhz}MHz/Heatmap_real_{i}")

    temp_root = base_dir / "_salvage_all"
    if temp_root.exists():
        shutil.rmtree(temp_root)
    temp_root.mkdir(parents=True, exist_ok=True)

    temp_paths: list[tuple[Path, Path]] = []
    for i, (src, final_dst) in enumerate(plan):
        tmp = temp_root / f"item_{i:03d}{_suffix(src.name)}"
        shutil.move(str(src), str(tmp))
        temp_paths.append((tmp, final_dst))

    for freq_tag in {f"freq_{mhz}MHz" for mhz in freq_list}:
        dest_dir = base_dir / freq_tag / f"K{k_value}" / "Real"
        if not dest_dir.exists():
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

    for mhz in freq_list:
        updated.append(base_dir / f"freq_{mhz}MHz" / f"K{k_value}" / "Real")

    print(f"\n✓ Salvage complete for {len(updated)} Real/ folder(s).")
    return updated


def move_pi_outputs_multi_freq(
    *,
    source_emc_dir: str = SOURCE_EMC_DIR,
    freq_list: list[int] | None = None,
    k_min: int = K_MIN,
    k_max: int = K_MAX,
    num_samples: int = _RUN_NUM_SAMPLES,
    base_generated_dir: str | Path = BASE_GENERATED_DIR,
    pis_per_sample: int | None = None,
    pi_output_kind: str | None = None,
) -> None:
    """Move PI-* outputs when PEB was generated with multiple PI frequencies.

    ECADStar numbers PI outputs globally in the order entries appear in the PEB.
    The PEB order mirrors run_all_k.py: outer loop = frequency, inner loop = K.
    So for freqs=[10,100,200], K=6..7, num_samples=1:
      PI-1,2  → freq_10MHz/K6/Real
      PI-3,4  → freq_10MHz/K7/Real
      PI-5,6  → freq_100MHz/K6/Real
      PI-7,8  → freq_100MHz/K7/Real
      PI-9,10 → freq_200MHz/K6/Real
      PI-11,12→ freq_200MHz/K7/Real
    """
    if freq_list is None:
        if WORKFLOW == "multifreq_heatmap_sweep":
            freq_list = list(exported_freq_mhz_list())
        else:
            freq_list = [mhz for mhz in _FREQ_LIST if mhz is not None]  # type: ignore

    global _PIS_PER_SAMPLE_OVERRIDE, _PI_OUTPUT_KIND_OVERRIDE
    if pis_per_sample is not None and pis_per_sample not in (1, 2):
        raise SystemExit("pis_per_sample must be 1 or 2")
    if pi_output_kind is not None and pi_output_kind not in ("heatmap", "impedance"):
        raise SystemExit('pi_output_kind must be "heatmap" or "impedance"')
    old_pps_override = _PIS_PER_SAMPLE_OVERRIDE
    old_kind_override = _PI_OUTPUT_KIND_OVERRIDE
    if pis_per_sample is not None:
        _PIS_PER_SAMPLE_OVERRIDE = int(pis_per_sample)
    if pi_output_kind is not None:
        _PI_OUTPUT_KIND_OVERRIDE = pi_output_kind

    try:
        _move_pi_outputs_multi_freq_impl(
            source_emc_dir=source_emc_dir,
            freq_list=freq_list,
            k_min=k_min,
            k_max=k_max,
            num_samples=num_samples,
            base_generated_dir=base_generated_dir,
        )
    finally:
        _PIS_PER_SAMPLE_OVERRIDE = old_pps_override
        _PI_OUTPUT_KIND_OVERRIDE = old_kind_override


def _move_pi_outputs_multi_freq_impl(
    *,
    source_emc_dir: str,
    freq_list: list[int],
    k_min: int,
    k_max: int,
    num_samples: int,
    base_generated_dir: str | Path,
) -> None:
    source_dir = _resolve_source_dir(source_emc_dir)
    if not source_dir.exists() or not source_dir.is_dir():
        raise SystemExit(
            f"Source .emc directory does not exist:\n  {source_dir}\n\n"
            "If this is a Windows path, run on Windows/WSL or update SOURCE_EMC_DIR."
        )

    base_dir = Path(base_generated_dir)
    ks = list(range(k_min, k_max + 1))

    # Build ordered list of (freq_tag, k, local_sample_offset) matching PEB order
    # and compute which global PI numbers belong to each slot.
    pps = _pis_per_sample()
    slots: list[tuple[str, int, int, int]] = []  # (freq_tag, k, local_i, global_idx)
    global_idx = 0
    for mhz in freq_list:
        freq_tag = f"freq_{mhz}MHz"
        for k in ks:
            for local_i in range(num_samples):
                slots.append((freq_tag, k, local_i, global_idx))
                global_idx += 1

    pi_to_dest: dict[int, tuple[Path, int]] = {}
    dest_dirs_seen: set[Path] = set()
    for freq_tag, k, local_i, g_idx in slots:
        dest_dir = base_dir / freq_tag / f"K{k}" / "Real"
        if pps == 1:
            pi_nums = [g_idx + 1]
        else:
            pi_nums = [2 * g_idx + 1, 2 * g_idx + 2]
        for pi_num in pi_nums:
            pi_to_dest[pi_num] = (dest_dir, local_i)
        dest_dirs_seen.add(dest_dir)

    # Ensure all dest dirs exist
    for d in dest_dirs_seen:
        d.mkdir(parents=True, exist_ok=True)

    # Optional clean
    if CLEAN_DEST_BEFORE_PASTE and not DRY_RUN:
        for d in dest_dirs_seen:
            d_resolved = d.resolve()
            if str(d_resolved) in ("/", ""):
                raise SystemExit(f"Refusing to clean unsafe DEST_DIR: {d}")
            for child in d.iterdir():
                if child.is_dir() and not child.is_symlink():
                    shutil.rmtree(child)
                else:
                    child.unlink()

    pattern = re.compile(PI_NAME_REGEX)
    moved = 0

    for item in _iter_candidates(source_dir):
        if not item.exists():
            continue
        if not pattern.match(item.name):
            continue
        pi_num = _parse_pi_number(item.name)
        if pi_num is None or pi_num not in pi_to_dest:
            continue

        dest_dir, local_i = pi_to_dest[pi_num]
        target = dest_dir / item.name
        if target.exists() and not OVERWRITE:
            target = _unique_dest_path(dest_dir, item.name)

        action = "MOVE" if MOVE else "COPY"
        freq_tag = target.parent.parent.parent.name  # .../freq_Xmhz/KY/Real
        k_tag    = target.parent.parent.name
        print(f"{action}: {item} -> {target}   ({freq_tag}/{k_tag}, sample={local_i})")

        if DRY_RUN:
            moved += 1
            continue

        if MOVE:
            if OVERWRITE and target.exists():
                shutil.rmtree(target) if target.is_dir() else target.unlink()
            shutil.move(str(item), str(target))
        else:
            if item.is_dir():
                if OVERWRITE and target.exists():
                    shutil.rmtree(target)
                shutil.copytree(item, target)
            else:
                if OVERWRITE and target.exists():
                    target.unlink()
                shutil.copy2(item, target)
        moved += 1

    # Rename per dest_dir
    if moved == 0:
        print("\n⚠ No PI-* files found in source – skipping rename.")
    elif RENAME_PI_TO_MATCH_SAMPLES and not DRY_RUN:
        # Group global_idx by dest_dir to find per-folder sample offset
        dir_to_slots: dict[Path, list[tuple[int, int]]] = {}  # dest_dir → [(local_i, global_idx)]
        for freq_tag, k, local_i, g_idx in slots:
            dest_dir = base_dir / freq_tag / f"K{k}" / "Real"
            dir_to_slots.setdefault(dest_dir, []).append((local_i, g_idx))

        for dest_dir, slot_list in dir_to_slots.items():
            # sample_offset = first global_idx in this folder
            min_g_idx = min(g_idx for _, g_idx in slot_list)
            n = len(set(local_i for local_i, _ in slot_list))
            _rename_pi_outputs_with_offset(dest_dir, num_samples=n, sample_offset=min_g_idx)

    print(f"\n✓ Done. {moved} item(s) {'moved' if MOVE else 'copied'} across"
          f" {len(dest_dirs_seen)} Real/ folders ({len(freq_list)} freq(s), K{k_min}..K{k_max}).")


if __name__ == "__main__":
    main()
