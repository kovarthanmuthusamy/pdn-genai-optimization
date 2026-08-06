---
title: ecadstar
type: code
path: pipelines/dataset_sim/ecadstar.py
group: pipelines/dataset_sim
loc: 522
tags: [code, pipelines]
---

# ecadstar

> ECADStar batch automation helpers for combinations simulation.

**Source:** `pipelines/dataset_sim/ecadstar.py` · 522 lines

## Constants

| Name | Value |
|------|-------|
| `_PI_RE` | `re.compile('^PI-(\\d+)(?:\\..+)?$', re.IGNORECASE)` |
| `_MAP_MHZ_RE` | `re.compile('Z_([\\d.]+)MHz', re.IGNORECASE)` |
| `_KILL_PS` | `"$titles=@('*eCADSTAR*PI/EMI*','*PI/EMI Analysis*');$procs=Get-Process \| Where-Object { $…` |

## Functions

- **`_is_windows_abs(path_str: str)`**
- **`_wsl_path_from_windows(path_str: str)`**
- **`_windows_path_from_wsl(path: Path)`**
- **`_windows_path_from_mangled_wsl(path: Path)`** — Recover C:\... when a Windows path was resolved under the Linux cwd.
- **`windows_path_str(path: Path | str)`**
- **`resolve_windows_path(path_str: str)`**
- **`parse_pi_number(name: str)`**
- **`kill_ecadstar_windows(*, verbose: bool=True)`** — Force-close any running ECADStar PI/EMI instance (matched by window title).
- **`clear_ecadstar_lock(erf_path: str)`**
- **`stage_peb_for_ecadstar(peb_path: Path, *, erf_path: str, peb_copy_dest: str | None)`** — Copy generated PEB beside .erf and optional PEB folder.
- **`run_ecadstar_batch_headless(peb_path: Path, *, erf_path: str, engineer_exe: str, impulse_port: int | None=None, timeout_sec: int=172800, verbose: bool=True)`** — Run a PI/EMI batch fully headless via ``engineer.exe --batch --batch-auto-exit``.
- **`_pi_folder_index(emc_dir: Path)`**
- **`_pi_folder(emc_dir: Path, pi_num: int)`**
- **`_fmt_ts(ts: float)`**
- **`_map_matches_mhz(path: Path, expected_mhz: float)`**
- **`_pi_map_files(pi_dir: Path, expected_mhz: float | None=None)`**
- **`_pi_map_mtime(pi_dir: Path, expected_mhz: float | None=None)`**
- **`_pi_csv_mtime(pi_dir: Path)`**
- **`_pi_ready_distribution(pi_index: dict[int, Path], pi_num: int, since_ts: float, expected_mhz: float | None=None)`**
- **`_pi_ready_impedance(pi_index: dict[int, Path], pi_num: int, since_ts: float)`**
- **`_inspect_pi_folder(folder: Path | None, *, since_ts: float, expected_mhz: float | None)`**
- **`_summarize_wait_state(pi_index: dict[int, Path], pi_count: int, since_ts: float, mode: str, expected_mhz: float | None)`**
- **`_print_wait_debug(*, emc: Path, pi_index: dict[int, Path], pi_count: int, batch_start_ts: float, mode: str, ready_count: int, expected_mhz: float | None, poll_index: int)`**
- **`wait_for_pi_outputs(emc_output_dir: str, *, pi_count: int, batch_start_ts: float, mode: str, timeout_sec: int, poll_sec: int, expected_mhz: float | None=None)`** — Poll until PI-1..PI-N are ready (distribution=.map, impedance=.csv).

## Imported by

- [[active_learning_pi.al.ecadstar]]
- [[append_legacy_19k_multifreq]]
- [[append_merged_combinations_multifreq]]
- [[append_restore_49k_legacy_multifreq]]
- [[ingest_labels]]
- [[move_outputs]]
- [[pipelines.dataset_sim.paths]]
- [[run_combinations_sim_pipeline]]
- [[run_multifreq_sweep_pipeline]]
- [[run_multitype_sim_pipeline]]
