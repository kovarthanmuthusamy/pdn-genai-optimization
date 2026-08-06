---
title: sampler_curriculum
type: code
path: experiments/exp051_new_datas_appended/codes/sampler_curriculum.py
group: experiments/exp051_new_datas_appended/codes
experiment: exp051_new_datas_appended
loc: 134
tags: [code, exp051_new_datas_appended]
---

# sampler_curriculum

> Per-epoch train sampling and layout-path curriculum for exp051.

**Source:** `experiments/exp051_new_datas_appended/codes/sampler_curriculum.py` · 134 lines
**Experiment:** [[exp051_new_datas_appended]]

## Classes

- **`CurriculumWeightedSampler(Sampler[int])`** — Weighted random sampler; call ``set_epoch`` before each training epoch.

## Functions

- **`_lerp(a: float, b: float, t: float)`**
- **`schedule_boost_frac(epoch_1based: int, *, boost_frac: float, natural_frac: float, start_epoch: int, peak_end_epoch: int, decay_end_epoch: int)`** — Rise to ``boost_frac`` by ``peak_end_epoch``, decay to ``natural_frac`` by ``decay_end_epoch``.
- **`tag_alpha_for_target_frac(target_frac: float, n_tag: int, n_other: int)`** — Relative weight multiplier for tagged rows to hit ``target_frac`` draw probability.
- **`load_append_tags_for_dataset(data_dir: Path, heatmap_files: Sequence[Path])`** — ``append_tag`` per dataset index (aligned with sorted heatmap stems).
- **`build_tag_boost_weights(train_indices: Sequence[int], row_tags: Sequence[str], tag_value: str, target_frac: float)`**
- **`apply_layout_path_curriculum(c, epoch_1based: int)`** — Update ``layout_train_prob`` and ``cross_freq_layout_mix_prob`` on config.

## Imported by

- [[experiments.exp051_new_datas_appended.codes.dataloader_multifreq]]
- [[experiments.exp051_new_datas_appended.codes.train_vae_simple]]

## External dependencies

`torch`
