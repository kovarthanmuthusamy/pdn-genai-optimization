# eCADSTAR PI/EMI headless batch (native CLI) — replaces AutoHotkey

This is the elegant, background-safe replacement for the AutoHotkey GUI automation.
`engineer.exe` (the PI/EMI Analysis app) exposes a **native batch command line** that
opens the design, runs a `.peb` batch, writes the PI outputs, and **quits itself** —
with no GUI window, no window focus, and no AutoHotkey.

## 📝 Summary of Changes

- Discovered `engineer.exe`'s built-in batch CLI (no macro recording, no COM needed).
- Added `run_ecadstar_batch_headless()` to `pipelines/dataset_sim/ecadstar.py`.
- Converted **every** pipeline that previously drove ECADStar via AutoHotkey to the
  headless CLI, then **removed AutoHotkey entirely** (no fallback):

  | Script | ECADStar driver |
  |--------|-----------------|
  | `pipelines/dataset_sim/run_multitype_sim_pipeline.py` | headless only |
  | `pipelines/dataset_sim/run_combinations_sim_pipeline.py` | headless only |
  | `active_learning_pi/al/ecadstar.py` (`al/pipeline.py`) | headless only |
  | `scrap/orchestration/run_multifreq_sweep_pipeline.py` | headless only |

  All resolve the engine via `ENGINEER_EXE` (module const) or `ecadstar.engineer_exe`
  (AL config), defaulting to
  `C:\Program Files\eCADSTAR\eCADSTAR 2023.0\Analysis\bin\engineer.exe`.

### AutoHotkey removal (complete)
Deleted:
- `tools/ecadstar/ecadstar_piemi_batch.ahk`
- `tools/ecadstar/capture_ecadstar_tools_coords.ahk`
- `tools/ecadstar/run_ecadstar_piemi_batch.ps1` (the AHK launcher)
- `run_ecadstar_batch()` + `wait_for_batch_started()` (and AHK-only helpers) from
  `pipelines/dataset_sim/ecadstar.py`
- All `ECADSTAR_AHK_EXE` / `ECADSTAR_SKIP_OPEN_ERF*` / `ECADSTAR_USE_HEADLESS_CLI`
  config and the `_skip_open_erf_for_distribution()` helpers from the pipelines.

Remaining `tools/ecadstar/` files are PowerShell/sh helpers only (`kill_ecadstar.ps1`,
`ecadstar_piemi_close.ps1`, `inspect_raw_folder.ps1`, `watch_resume.sh`).

## 🚀 Implementation Details

### The command
```
engineer.exe <design.erf> --batch <file.peb> --batch-auto-exit
```

Full option table (extracted from `engineer.exe`):

| Short | Long                | Meaning |
|-------|---------------------|---------|
| `-b`  | `--batch <file>`    | Execute the given batch file directly after load |
| `-be` | `--batch-auto-exit` | Quit directly after batch execution |
| `-p`  | `--impulse-port <n>`| Port for the IMPULSE model-library server |
| `-h`  | `--help`            | Show help and exit (GUI dialog) |
| `-v`  | `--version`         | Show version and exit (GUI dialog) |
| `-n`  | `--installation`    | Report install paths and exit (GUI dialog) |
| `-i`  | `--issue`           | Report dependent libraries and exit |

Positional argument is the design (`riffile` = `.erf`/`.rif`).

Install path (this machine):
```
C:\Program Files\eCADSTAR\eCADSTAR 2023.0\Analysis\bin\engineer.exe
```

### Why this is better than AutoHotkey / macro playback
- **No GUI focus dependency** — runs even if the window is not foreground / you are
  on another desktop or disconnected from RDP. True background operation.
- **No stale-window race** — the "Target window not found" AHK crash cannot happen.
- **Self-terminating** — `--batch-auto-exit` releases the design lock on exit, so
  there is no orphaned `.rlk` to clean between phases.
- **No manual macro to record** (unlike the `-playback` option).

### Runner API
`run_ecadstar_batch_headless(peb_path, *, erf_path, engineer_exe, impulse_port=None,
timeout_sec=172_800, verbose=True) -> int`

- Launches via PowerShell `Start-Process ... -PassThru -NoNewWindow` and blocks on
  `WaitForExit(timeout_ms)`. Process self-exit is the ground-truth completion signal.
- Returns `0` on clean self-exit, `124` on timeout (process force-killed), `-1` if
  PowerShell is unavailable.
- The pipeline still runs `wait_for_pi_outputs(...)` afterward as a fast correctness
  check that the expected `PI-1..N` outputs are present and fresh before harvesting.

### Pipeline config (`run_multitype_sim_pipeline.py`)
```python
ENGINEER_EXE = r"C:\Program Files\eCADSTAR\eCADSTAR 2023.0\Analysis\bin\engineer.exe"
ECADSTAR_IMPULSE_PORT: int | None = None   # None worked in validation
```
AutoHotkey has been removed entirely — the headless CLI is the only ECADStar driver
(there is no `ECADSTAR_USE_HEADLESS_CLI` toggle anymore).

## 🛠️ Verification & Execution Results

Validated 2026-08-04 with a 2-layout impedance batch (`test_cli_impedance.peb`:
row0 = one Type-1 decap in slot 0, row1 = one Type-2 decap in slot 5), design
`H-shape.erf`, with the PCB Editor left open in the background.

Raw CLI run:
- `engineer.exe ... --batch ... --batch-auto-exit` → **self-exited in ~6 s**.
- `PI-1/Power_GND/1-PIPinZ_IC1_Port1.csv` produced (impedance spectrum result).
- `PI-1/log.txt`: `Batch file ... read successfully` → `Perform batch step: 1` →
  `Impedance spectrum calculation needed 0.524 seconds` → `PI Analysis completed`.
- Top-level `log.txt`: `Batch is terminated.`
- No PI/EMI window ever became foreground; PCB Editor was untouched.

Integrated pipeline functions (`run_ecadstar_batch_headless` + `wait_for_pi_outputs`,
`mode="impedance"`, `pi_count=2`):
- `=> rc = 0`
- `✓ All 2 PI outputs ready.` (fresh `.csv` for PI-1 and PI-2)

**Status: PASS for the impedance (PI-Spectrum) path.**

### Distribution (PI-Distribution `.map`) — PASS (measured 2026-08-04)
Real pipeline resume at **200 MHz** (15,000-layout `multitype_dist_200MHz.peb`, built
by the pipeline, driven through `_simulate_peb`'s headless branch):
- `engineer.exe` launched headless (no window, ~3.5 GB working set), read the
  213 MB PEB (`Batch file ... multitype_dist_200MHz.peb read successfully`), ran
  `Perform batch step: 1`, `Plane pair generation completed`.
- PI folders grew 19 → 142 in ~40 s (**≈3 layouts/s** → ≈80 min for the full 15k at
  this frequency), each with the harvested artifact `Power_GND/Z_0200.000MHz.map`.
- No AutoHotkey, no GUI focus; PCB Editor untouched in the background.

Both phases (impedance + distribution) now run through the native CLI. AutoHotkey has
been removed entirely; there is no fallback path.
