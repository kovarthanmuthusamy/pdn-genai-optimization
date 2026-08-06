---
title: tools (code index)
type: index
tags: [index, code, tools]
---

# tools/ — code index

17 modules.

## `tools/`

- [[add_config_keys_to_docstrings]] *(runnable)* — One-off: add Config keys to Agent notes and normalize CONFIGURATION headers.
- [[build_vault]] *(runnable)* — Build an Obsidian knowledge-graph vault from this repository.
- [[check_mhz_manifest]]
- [[clean_agent_notes]] *(runnable)* — Remove CLI/argparse wording from module docstrings and user-facing messages.
- [[consolidate_data_dirs]] — Move legacy data folders into data/ and remove empty legacy directories.
- [[debug_fix]]
- [[fix_docstring_paths]] — Fix docstrings corrupted by migrate_repo_paths.py (setup_path lines inside quotes).
- [[fix_evaluate_vae_main]]
- [[fix_pipeline_paths]] — Fix repo-root path depth after pipelines/ migration.
- [[fix_repo_paths_bootstrap]] — Insert sys.path bootstrap before ``from repo_paths import`` in entry scripts.
- [[fix_repo_paths_imports]] — Fix broken repo_paths import lines from migrate_all_paths.py.
- [[fix_scratch_paths]] — Normalize scratch/*.py to use repo_paths.
- [[improve_agent_notes]] — Apply improved Agent notes (What / Usage / Config keys) to entry scripts.
- [[migrate_all_paths]] — Migrate stale absolute paths to repo-relative / repo_paths usage.
- [[migrate_repo_paths]] — One-shot migration: replace hard-coded repo-root detection with repo_paths imports.
- [[reorganize_scripts]] — One-time layout migration: move scripts into pipelines/ + libs/, leave compat shims.
- [[smoke_dataset_meta]] — One-off smoke test for libs.dataset_meta (not part of main pipelines).
