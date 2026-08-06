---
title: docs_update_ahk_removal
type: concept
source: docs/docs_update_ahk_removal.md
tags: [concept, thesis]
---

> [!info] Mirror of `docs/docs_update_ahk_removal.md` — edit the source file, then re-run `tools/build_vault.py`.

# Documentation update: AutoHotkey removal → headless CLI

Pass to bring all documentation in line with the code change that **removed
AutoHotkey entirely** and drives eCADSTAR PI/EMI through the native headless CLI
(`engineer.exe --batch --batch-auto-exit`). See
[[ecadstar_headless_cli|`ecadstar_headless_cli.md`]] for the mechanism.

## 📝 Summary of Changes

Active docs updated to the headless model:

- `docs/ecadstar_headless_cli.md` — removed the last two lines that implied AHK was
  kept as an inert fallback (`ECADSTAR_USE_HEADLESS_CLI`); it is now stated as fully
  removed with no fallback.
- `docs/data-pipeline.md` — pipeline diagram + section 5 rewritten: simulation is
  headless `engineer.exe --batch`; no RDP-foreground / AutoHotkey constraints; lock is
  self-released on exit.
- `docs/limitations-and-validity.md` — "RDP/AHK automation failures" row reworded to
  "ECADStar automation failures (now headless)".
- `pipelines/dataset_sim/README.md` — config table drops `ECADSTAR_AHK_EXE`,
  `ECADSTAR_SKIP_OPEN_ERF`, `ECADSTAR_BATCH_START_*`; adds `ENGINEER_EXE` /
  `ECADSTAR_IMPULSE_PORT`. Flow/"how it knows it's finished"/requirements now describe
  the headless engine and self-exit.
- `scrap/orchestration/ECADSTAR_LOCK_FILE.md`,
  `scrap/orchestration/PEB_FULL_PATH_LOAD_BATCH.md` — SUPERSEDED banners added.

AHK-only docs marked obsolete (kept as history):

- `docs/ecadstar_persistence_and_locking.md` — SUPERSEDED banner (persistence/locking/
  foreground concerns no longer apply under the headless CLI).
- `docs/ecadstar_ahk_winactivate_fix.md` — OBSOLETE banner (the `WinActivate`
  crash class cannot occur without GUI automation).

Thesis vault regenerated:

- `tools/build_vault.py` re-run — it owns and rewrites `10-Concepts`, `12-Archive`,
  `30-Code`, etc., from the updated `docs/*.md` and code. This propagated all of the
  above into the vault mirrors (e.g. the AL `ecadstar` code note now reads
  "native headless CLI, engineer.exe --batch"; stale `SKIP_OPEN_ERF` config tables and
  the removed `run_ecadstar_batch`/`wait_for_batch_started`/`_skip_open_erf_for_distribution`
  listings are gone).

Left intentionally unchanged (historical records):

- `docs/_archive/*` and `thesis_vault/12-Archive/*` — past AHK/RDP bug-fix notes
  (`HEATMAP_63MHZ_PIEMI_ACTIVATION_FIX`, `sim-pipeline-auto-append`, archive READMEs).
  These document what happened at the time and are preserved as-is.

## 🚀 Implementation Details

- Method used: targeted `StrReplace` edits on active docs; top-of-file "SUPERSEDED"/
  "OBSOLETE" blockquote banners on the AHK-only docs (pointing to
  `ecadstar_headless_cli.md`) rather than deleting them, so the troubleshooting history
  is retained.
- Vault mirrors are generated, so they were refreshed via `python tools/build_vault.py`
  instead of hand-patching each auto-generated note. The generator parses the updated
  code docstrings/CONFIG and mirrors `docs/*.md`, so the concept and code-index notes
  now match the source.

## 🛠️ Verification & Execution Results

- `python tools/build_vault.py` → exit 0: `646 code notes, 21 concept notes, 772 .md`,
  `parse errors: 0`, mirrored 776 files.
- Post-update `rg` for `AutoHotkey|AHK|.ahk|WinActivate|SKIP_OPEN_ERF|skipopen` across
  `*.md`: remaining hits are only (a) `_archive/` + `12-Archive/` history, (b) the
  banner-marked superseded/obsolete docs, and (c) correct "no AutoHotkey" phrasing in
  the headless docs. No active doc still instructs the reader to use AutoHotkey.
