---
title: build_vault
type: code
path: tools/build_vault.py
group: tools
loc: 1318
tags: [code, tools, runnable, uncommitted]
---

# build_vault

> Build an Obsidian knowledge-graph vault from this repository.

**Source:** `tools/build_vault.py` · 1318 lines
**Git:** uncommitted — not yet tracked
**Runnable:** CONFIG-only script — edit constants at top, then `python tools/build_vault.py`

## Purpose

```text
Build an Obsidian knowledge-graph vault from this repository.

Purpose:
    Convert the codebase (Python modules, docs, experiment notes, AL run
    results, dataset references) into a linked Obsidian vault at
    ``thesis_vault/`` so the thesis can be written against a navigable graph
    instead of raw source trees.

Run:
    python tools/build_vault.py

Agent notes:
    What
        Walks every tracked ``*.py`` file, parses it with ``ast``, and emits one
        note per module carrying its docstring, top-level classes/functions,
        CONFIG constants, and internal imports rendered as ``[[wikilinks]]``.
        Also mirrors ``docs/*.md`` as concept notes, builds one note per
        ``experiments/*`` (config + notes.md + lineage), one per
        ``active_learning_pi/runs/*`` (decision ledger), and one per referenced
        dataset directory.
    Usage
        Edit the CONFIG block below, then ``python tools/build_vault.py``.
        Re-run after code changes; generated folders are rewritten wholesale.
    Config keys
        VAULT_DIR       — vault output directory (repo-relative)
        OWNED_DIRS      — folders the generator rewrites; everything else is
                          left alone so hand-written notes survive re-runs
        LINEAGE_HINT    — pins order only for non-numeric experiment names
        CURRENT_TRACK   — experiments marked as the active model track
        CONCEPT_CODE    — curated concept-doc -> source-path links
```

## Constants

| Name | Value |
|------|-------|
| `_REPO` | `Path(__file__).resolve().parents[1]` |
| `VAULT_DIR` | `'thesis_vault'` |
| `WINDOWS_MIRROR` | `'/mnt/c/Users/muthusamy/thesis_vault'` |
| `OWNED_DIRS` | `('10-Concepts', '12-Archive', '20-Experiments', '30-Code', '40-Datasets', '50-Results')` |
| `HANDWRITTEN` | `('00-Thesis', '05-Literature', '15-Ideas', 'README.md')` |
| `MIRROR_SEED_ONCE` | `('.obsidian',)` |
| `SYNC_STATE_NAME` | `'.vault_sync.json'` |
| `LINEAGE_HINT` | `['exp029_heat_private', 'exp037_lat_change', 'exp038_true_multi', 'exp039_improved_heatma…` |
| `SUMMARY_ONLY_BELOW` | `37` |
| `CURRENT_TRACK` | `{'exp059_capacity_freq': 'current', 'exp060_multitype_occ': 'exploratory'}` |
| `LEGACY` | `{'exp057_structured_graph', 'exp058_asymmetric_kl'}` |
| `CONCEPT_CODE` | `{'framework-overview': ['pipelines/latent/optimize.py', 'pipelines/active_learning/run.py…` |
| `KEY_HPARAMS` | `['latent_dim', 'heatmap_private_dim', 'cond_dim', 'freq_fourier_features', 'use_multiscal…` |
| `VAULT` | `REPO_ROOT / VAULT_DIR` |
| `STDLIB_HINT` | `{'os', 'sys', 'json', 'math', 're', 'csv', 'time', 'shutil', 'pathlib', 'typing', 'datacl…` |
| `_GAN_RE` | `re.compile('discriminator\|adversarial\|wgan\|gan_loss\|generator_loss\|net_?[dg]\\b', re.I)` |
| `_VAE_RE` | `re.compile('\\bvae\\b\|reparameter\|logvar\|kl_div\|elbo\|posterior', re.I)` |

## Classes

- **`ModuleInfo`**

## Functions

- **`python_files()`** — Every .py file git would consider part of the project.
- **`_no_links(text: str)`** — Neutralise stray ``[ [`` in prose so docstrings cannot forge graph edges.
- **`_first_line(doc: str | None)`**
- **`_sig(node: ast.FunctionDef | ast.AsyncFunctionDef)`**
- **`_const_value(node: ast.AST, limit: int=90)`**
- **`parse_module(rel: str)`**
- **`assign_note_names(modules: list[ModuleInfo])`** — Short stem when globally unique, else the full dotted path.
- **`sanitize(name: str)`**
- **`yaml_list(values)`**
- **`write_note(rel_path: str, content: str)`**
- **`load_jsonlike(path: Path)`** — Experiment config.yaml files are JSON with '#' comment lines.
- **`build_code_notes(modules: list[ModuleInfo], index: dict[str, ModuleInfo])`**
- **`experiment_dirs()`** — All experiment directories, ordered oldest -> newest by numeric prefix.
- **`is_summary_only(exp: str)`** — True when an experiment should get a summary note but no per-module notes.
- **`detect_era(exp: str)`** — Classify an experiment as gan / vae / unknown.
- **`build_area_indexes(modules: list[ModuleInfo])`** — One index note per top-level code area, so no module is unreachable.
- **`build_archive_index()`**
- **`concept_note_name(stem: str)`** — docs/README.md would collide with the vault's own README — rename it.
- **`archive_note_name(stem: str)`** — Archive mirrors several filenames that also exist in docs/; suffix them.
- **`build_concept_notes(index_by_path: dict[str, ModuleInfo])`**
- **`build_experiment_notes(modules: list[ModuleInfo])`**
- **`build_lineage_note(order: list[str])`** — Narrative index of the whole experiment arc, grouped by detected era.
- **`build_result_notes()`**
- **`build_dataset_notes()`** — Datasets are untracked; derive notes from every config that references them.
- **`build_home(modules: list[ModuleInfo], experiments: list[str], results: list[str], datasets: list[str], areas: list[str])`**
- **`write_obsidian_config()`** — Graph colour groups so the vault is readable on first open.
- **`_sha(path: Path)`**
- **`_handwritten_files(root: Path)`** — Map vault-relative path -> file, for every hand-written note under *root*.
- **`sync_handwritten(mirror: Path)`** — Three-way sync of hand-written notes between repo and mirror.
- **`save_sync_state()`** — Record hashes of hand-written notes so the next run can detect real edits.
- **`push_to_mirror(mirror: Path)`** — Mirror the repo vault onto the Windows filesystem for Obsidian.
- **`main()`**

## Imports

- [[repo_paths]]

## External dependencies

`repo_paths`
