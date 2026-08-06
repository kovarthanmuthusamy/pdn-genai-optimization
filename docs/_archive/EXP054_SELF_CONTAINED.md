# exp054: Self-contained training layout

exp054 no longer imports from `experiments/exp038_true_multi`. The training loop, dataloader base, and physics loss are vendored under `experiments/exp054_K_30/codes/`.

---

## 📝 Summary of Changes

- Added vendored modules: `train_core.py`, `dataloader_base.py`, `physics_loss.py`, `eval_off_anchor.py`
- `train_vae_simple.py` now patches `train_core` (not exp038)
- Removed runtime monkey-patches: `_patch_spatial_eval`, early-stop stripping, `importlib` hacks
- `train_core` default `experiment_dir` → `experiments/exp054_K_30`
- Off-anchor eval interval/config wired via `eval_off_anchor.set_off_anchor_config(c)` in `_apply_yaml_exp054`

---

## 🚀 Implementation Details

### Module layout

| File | Role |
|------|------|
| `train_vae_simple.py` | Entry point: exp054 `Config`, loss hooks, yaml apply, calls `train_core.train_vae()` |
| `train_core.py` | Full training loop (config, dataloaders, epoch loop, checkpointing, plots) |
| `dataloader_base.py` | Base multifreq dataset / collate (from exp038, now local) |
| `dataloader_multifreq.py` | exp054 cross-freq pairs on top of `dataloader_base` |
| `physics_loss.py` | Physics RI/critic/AR losses |
| `eval_off_anchor.py` | Interval-gated wrapper → `eval_spatial_metrics` → `metrics/off_anchor_eval.csv` |
| `run_epoch_encode.py` | Per-epoch train/val with layout path, distill, cross-freq |

### External imports (intentional)

- `src_vae.others.vae_logger` — logging only
- `experiments/exp043/codes/freq_conditioning` — FiLM freq conditioning in VAE backbone
- `repo_paths` — repo root resolution

### Removed from runtime

- `_patch_spatial_eval()` (exp038 `eval_cross_freq` monkey-patch)
- `_strip_early_stop_from_train_vae()` regex patch (early-stop block removed in `train_core` source)
- `importlib` dynamic imports for eval

---

## 🛠️ Verification & Execution Results

```bash
# Syntax + import check
python3 -c "from experiments.exp054_K_30.codes import train_vae_simple"

# 30s smoke run (no Traceback / AttributeError / ModuleNotFoundError)
timeout 30 python experiments/exp054_K_30/codes/train_vae_simple.py
```

Both passed. Full 400-epoch run not re-verified in this step.

`rg exp038 experiments/exp054_K_30/` — only a comment in `train_vae_simple.py` (Config docstring).
