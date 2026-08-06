---
title: README
type: moc
tags: [moc]
---

# thesis_vault

Obsidian vault generated from the `genai_pdn` codebase. Start at [[Home]].

## Where to open it

**Open `C:\Users\muthusamy\thesis_vault` — not this WSL copy.**

Obsidian cannot open a vault over `\\wsl.localhost`: its file watcher fails with
`EISDIR: illegal operation on a directory, watch ...` because WSL's 9P filesystem emits
no Windows change notifications. `tools/build_vault.py` therefore mirrors the vault onto
the Windows filesystem, and Obsidian opens the mirror.

Hand-written notes (`00-Thesis/`, `15-Ideas/`, this README) are synced **both ways** by
comparing each side against the hashes recorded at the last run, so an edit made in
Obsidian and an edit made in the repo cannot overwrite each other. If the same note
changed on both sides, the Obsidian version is kept beside the repo version as
`<name>.conflict-from-obsidian.md` and the run prints a warning — nothing is discarded.

Generated folders flow one way only (repo → mirror) and are replaced wholesale.
`.obsidian/` is seeded into a new mirror once so graph colours are preconfigured, then
never written again — your window layout and open tabs survive rebuilds, including
rebuilds run while Obsidian is open.

## Layout

| Folder | Owner | Contents |
|--------|-------|----------|
| `00-Thesis/` | **you** | [[Thesis Outline]], [[Claim Ledger]], [[Reading Paths]] |
| `15-Ideas/` | **you** | your own drafting notes — start at [[_start-here]] |
| `10-Concepts/` | generator | mirrors of `docs/*.md`, linked to the code that implements them |
| `12-Archive/` | generator | mirrors of `docs/_archive/*.md` — not for citation |
| `20-Experiments/` | generator | one note per `experiments/*`, with config, notes, lineage, modules |
| `30-Code/` | generator | one note per Python module: docstring, classes, functions, constants, imports |
| `40-Datasets/` | generator | one note per referenced dataset directory |
| `50-Results/` | generator | one note per `active_learning_pi/runs/*`, incl. decision ledgers |

## Regenerating

```bash
python tools/build_vault.py
```

The generator **deletes and rewrites** `10-Concepts`, `12-Archive`, `20-Experiments`,
`30-Code`, `40-Datasets`, `50-Results`, plus `Home.md`. It never touches `00-Thesis/`,
`15-Ideas/`, or this README — write there freely.

Because `10-Concepts/` is a mirror, **edit `docs/*.md` in the repo**, not the note. The
mirror carries a callout reminding you of this.

## Note naming

A module note is named after its filename when that filename is unique in the repo
(`gp_error_surrogate`, `optimize`). Where a name repeats across experiment forks — 24
copies of `train_vae_simple.py`, 16 of `vae_poe_freq.py` — the note takes the full dotted
path instead (`experiments.exp059_capacity_freq.codes.vae_poe_freq`), so links stay
unambiguous. Frontmatter always carries the exact `path:`.

## Coverage

Built from 646 Python files (561 tracked + 85 uncommitted, the latter tagged
`#uncommitted`), 18 concept docs, 44 archived docs, 25 experiments, 7 AL runs,
10 dataset references — 754 notes, ~4250 links.

Nothing under `datasets/`, `checkpoints/`, or `*.pt` is read; those are referenced by
path only, as they are untracked and large.
