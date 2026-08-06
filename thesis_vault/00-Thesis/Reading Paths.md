---
title: Reading Paths
type: moc
tags: [moc, thesis]
---

# Reading Paths

Guided tours through the graph. Each path is a sequence of notes that answers one question
end-to-end — useful when drafting a chapter or re-orienting after time away.

---

## Path A — "How does a placement become a predicted spectrum?"

1. [[dataset]] — what a sample is (occupancy `[52]`, spectrum `[231]`, heatmap `[64×64]`)
2. [[normalization-and-losses]] — how those tensors are normalized
3. [[experiments.exp059_capacity_freq.codes.dataloader_multifreq]] — batching and conditioning
4. [[experiments.exp059_capacity_freq.codes.graph_occ]] — occupancy encoder over the slot graph
5. [[model-architecture]] — PoE fusion into the structured latent
6. [[experiments.exp059_capacity_freq.codes.vae_poe_freq]] — the model itself
7. [[experiments.exp059_capacity_freq.codes.train_core]] — the loop that fits it

## Path B — "How is a design proposed?" (Stage 2)

1. [[latent-optimization]] — objective, STE, selection rule
2. [[optimize]] — the optimizer, CONFIG block and all loss weights
3. [[impedance-surrogate]] — the forward model the optimizer queries
4. [[find_feasible]] — feasibility sampling
5. [[generate_run_report]] — how a run becomes a report

⚠️ This path currently runs on exp038 checkpoints, not [[exp059_capacity_freq]].

## Path C — "How does active learning close the loop?"

1. [[active-learning]] — the 8-stage cycle
2. [[candidates]] → [[inference_pool]] → [[per_k_acquire]] — propose and score
3. [[gp-error-surrogate]] + [[gp_error_surrogate]] — residual GP, UCB, `score_mode: mu`
4. [[peb_batch]] → [[active_learning_pi.al.ecadstar]] — export and simulate on the Windows host
5. [[ingest_labels]] → [[robust_normalize]] → [[build_overlay]] — labels back into a dataset
6. [[finetune_run]] — fine-tune the surrogate
7. [[decision_report]] → [[al_exp059_gp_error_001]] — verdicts
8. [[Claim Ledger]] — what may actually be written

## Path D — "Why is the model the way it is?" (lineage)

Walk the lineage chain, reading only each experiment's *Notes* section:

[[exp054_K_30]] → [[exp055_hard_occ]] → [[exp056_graph_vae]] → [[exp057_structured_graph]]
→ [[exp058_asymmetric_kl]] → [[exp059_capacity_freq]] → [[exp060_multitype_occ]]

Then [[experiment-lineage]] for the consolidated narrative and [[gnn-rationale]] for the
architecture argument.

## Path E — "What can I defend in a viva?"

1. [[Claim Ledger]] — verdicts and the one caution
2. [[limitations-and-validity]] — threats to validity
3. [[al_exp059_gp_error_001]] — the numbers themselves
4. [[acquisition_ab_validation_exp059]] — the equal-budget A/B

---

## Graph-view tips

The generator writes `.obsidian/graph.json` with colour groups:

| Colour group | Tag |
|--------------|-----|
| Concepts | `#concept` |
| Experiments | `#experiment` |
| Results | `#result` |
| Datasets | `#dataset` |
| Maps of content | `#moc` |
| Archived notes | `#archive` |
| Empty modules | `#stub` |

Useful graph filters:

- `-tag:#stub -tag:#archive` — hide empty `__init__.py` modules and archived chatter
- `tag:#exp059_capacity_freq` — the current model's code cluster only
- `tag:#concept OR tag:#experiment OR tag:#result` — the thesis-level graph, no source files
- `tag:#uncommitted` — code that exists on disk but is not yet in git
- `tag:#runnable` — CONFIG-only entry-point scripts
