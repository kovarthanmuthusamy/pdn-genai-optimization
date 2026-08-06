---
title: csv_to_occupancy
type: code
path: libs/data_creation/csv_to_occupancy.py
group: libs/data_creation
loc: 183
tags: [code, libs]
---

# csv_to_occupancy

> 52-d decap vector → binary occupancy vector conversion utilities.

**Source:** `libs/data_creation/csv_to_occupancy.py` · 183 lines

## Purpose

```text
52-d decap vector → binary occupancy vector conversion utilities.

Run:
    Import only — ``from libs.data_creation.csv_to_occupancy import create_occupancy_vector``.
```

## Constants

| Name | Value |
|------|-------|
| `DECAP_VECTOR_LENGTH` | `52` |

## Functions

- **`create_occupancy_vector(decap_vector)`** — Create a normalised float32 occupancy vector from a raw decap vector.
- **`visualize_occupancy_vector(vector_or_path)`** — Return the active capacitor labels (value > 0.5) from an occupancy vector.
- **`read_decap_csv(csv_path, max_samples=None, verbose=True)`** — Read decap vector CSV file.
- **`convert_csv_to_occupancy(csv_path, output_dir, max_samples=None, start_index=1, verbose=True)`** — Convert CSV rows to occupancy grid samples.
- **`get_existing_sample_count(output_dir)`** — Count existing sample_*.npy files in directory.

## Imported by

- [[experiments.exp037_lat_change.codes.inference_vae]]
- [[experiments.exp038_true_multi.codes.inference_vae]]
- [[experiments.exp039_improved_heatmap.codes.codes.inference_vae]]
- [[experiments.exp040.codes.codes.inference_vae]]
- [[experiments.exp041.codes.codes.inference_vae]]
- [[experiments.exp043.codes.inference_vae]]
- [[experiments.exp044.codes.inference_vae]]
- [[experiments.exp045.codes.inference_vae]]
- [[experiments.exp046.codes.inference_vae]]
- [[experiments.exp047.codes.inference_vae]]
- [[experiments.exp048.codes.inference_vae]]
- [[experiments.exp049.codes.inference_vae]]
- [[experiments.exp050.codes.inference_vae]]
- [[experiments.exp051_new_datas_appended.codes.inference_vae]]
- [[experiments.exp052_unbounded_pearson.codes.inference_vae]]
- [[experiments.exp053_peak_log1p_losses.codes.inference_vae]]
- [[experiments.exp054_K_30.codes.inference_vae]]
- [[experiments.exp055_hard_occ.codes.inference_vae]]
- [[experiments.exp056_graph_vae.codes.inference_vae]]
- [[experiments.exp057_structured_graph.codes.inference_vae]]
- [[experiments.exp058_asymmetric_kl.codes.inference_vae]]
- [[experiments.exp059_capacity_freq.codes.inference_vae]]
- [[experiments.exp060_multitype_occ.codes.inference_vae]]
- [[optimize]]
- [[processing_eval]]
- [[processing_multifreq]]
- [[visualize_sample]]

## External dependencies

`numpy`, `pandas`, `tqdm`
