---
title: sweep_latent_opt_rules
type: code
path: scrap/generation/sweep_latent_opt_rules.py
group: scrap/generation
loc: 477
tags: [code, scrap]
---

# sweep_latent_opt_rules

> Hard rules aligning multifreq sweep QC with ``pipelines/latent/optimize.py``.

**Source:** `scrap/generation/sweep_latent_opt_rules.py` · 477 lines

## Purpose

```text
Hard rules aligning multifreq sweep QC with ``pipelines/latent/optimize.py``.

The sweep is a *quality check* of the trained VAE — not a random layout sampler.
These rules mirror the production latent-optimization → heatmap path:

1. **Occupancy / impedance** come from a fixed layout (val set or latent-run export),
   never from marginal z sampling (unless ``ALLOW_RANDOM_LAYOUT=True``).
2. **Layout encode** uses ``PI_REF_MHZ``; **heatmap decode** changes only ``PI_freq``
   per sweep MHz (matches ``model.inference`` layout mode and post-opt ``decode``).
3. **``latent_z`` mode** — load ``best_latent.npy``, decode heatmaps at each MHz without
   re-encoding (matches heatmaps after latent optimization).
4. **``layout_hybrid`` mode** — ``encode_layout_latent_full(occ, imp)`` for private dims,
   then ``z[:, :shared] = z_opt[:, :shared]`` (recommended post-opt variant).
5. **Top-K occupancy** for PEB uses hard top-K (same as optimize ``_topk_occ``).
```

## Constants

| Name | Value |
|------|-------|
| `QC_INFERENCE_MODES` | `frozenset({'layout_qc', 'latent_z', 'layout_hybrid'})` |
| `RANDOM_LAYOUT_MODES` | `frozenset({'marginal', 'layout', 'anchor_blend', 'encode'})` |
| `LATENT_RUN_FILES` | `{'z': 'best_latent.npy', 'occ_topk': 'best_occupancy_topk.npy', 'occ_prob': 'best_occupan…` |

## Classes

- **`SweepQCConfig`**
- **`SweepQCError(ValueError)`** — Raised when sweep config violates latent-opt alignment rules.

## Functions

- **`validate_sweep_qc(cfg: SweepQCConfig)`** — Enforce QC rules before generate.
- **`topk_occ_binary(occ_prob: torch.Tensor, k: int)`** — Hard top-K mask — matches ``optimize._topk_occ``.
- **`_imp_log_to_norm(imp_log: torch.Tensor, imp_log_mean: float, imp_log_std: float)`** — Convert log-Ω impedance to model normalized space (B, 1, 231).
- **`latent_run_k_dir(run_dir: Path, k: int)`** — ``.../K30`` folder inside a latent optimization run.
- **`load_latent_run_bundle(run_dir: Path | str, k: int, device: torch.device, *, imp_log_mean: float, imp_log_std: float)`** — Load optimize.py exports for one K.
- **`load_explicit_npy_bundle(*, z_npy: str | None, occ_npy: str | None, imp_npy: str | None, k: int, device: torch.device, imp_log_mean: float, imp_log_std: float)`** — Load explicit .npy paths (latent-run style layout).
- **`_sweep_val_loader_key(data_dir: Path | str, experiment_cfg: dict[str, Any])`**
- **`k_values_in_val_loader(val_ld: DataLoader)`** — Return K values present in the val split (one pass over ``val_ld``).
- **`filter_k_for_layout_qc(k_values: list[int], val_ld: DataLoader)`** — Drop K not in val; raise if none remain.
- **`get_sweep_val_loader(*, data_dir: Path | str, experiment_cfg: dict[str, Any], device: torch.device)`** — Shared val ``DataLoader`` for sweep/QC — built once per dataset + split config.
- **`load_val_layout_samples(*, data_dir: Path | str, experiment_cfg: dict[str, Any], k_value: int, num_samples: int, seed: int, device: torch.device, with_ground_truth: bool=False, val_ld: DataLoader | None=None)`** — Collect ``num_samples`` real val layouts with fixed K (matches eval_real_data_sweep).
- **`load_val_anchor_samples(*, data_dir: Path | str, experiment_cfg: dict[str, Any], k_value: int, anchor_mhz: float, num_samples: int, seed: int, device: torch.device, mhz_tol: float=2.0, val_ld: DataLoader | None=None)`** — Val layouts with real heatmaps at a training-anchor PI_freq.
- **`build_hybrid_z(model: torch.nn.Module, z_opt: torch.Tensor, occ_prob: torch.Tensor, imp_norm: torch.Tensor, k: torch.Tensor, pi_ref_mhz: float, device: torch.device)`** — Private dims from layout head; shared dims from optimized z.
- **`decode_heatmap_sweep_mhz(model: torch.nn.Module, z: torch.Tensor, k: torch.Tensor, mhz_list: list[float], device: torch.device, *, occ_for_decode: torch.Tensor | None=None)`** — Decode heatmaps at each MHz — z fixed (latent-opt path). PI_freq only changes.
- **`qc_manifest_extra(cfg: SweepQCConfig)`** — Extra fields for sweep_freq_manifest.json.

## Imports

- [[experiments.exp038_true_multi.codes.dataloader_multifreq]]
- [[pi_freq_utils]]

## Imported by

- [[run_multifreq_heatmap_sweep]]
- [[sweep_qc_eval]]

## External dependencies

`experiments`, `numpy`, `src_vae`, `torch`
