# thesis_code — the code behind the thesis model

This folder is a trimmed, runnable copy of the code that produced the reference model of the thesis
(the multi-modal VAE surrogate trained for 700 epochs, plus its placement-only encoder) and every
number of Chapter 4. It is ordered by pipeline stage, from the sampling plan of the dataset to the
evaluation scripts. Only what the thesis setting executes is kept; the defaults are the thesis
settings. Appendix D of the thesis lists every file with its purpose and the section it implements.

All scripts are run as modules from the repository root and take no command-line arguments: each
begins with a `CONFIGURATION` block that is edited before the run.

```bash
python -m thesis_code.<stage>.<module>
```

## Stages

| Folder | What it does | Entry points |
|---|---|---|
| `common/` | settings, output folders, board mask, placement formats, frequency grid, shipped inputs (`inputs/`) | `common.make_targets` (target spectra) |
| `s01_generation/` | sampling plan of the 30,000 placements, PEB files, eCADSTAR driver | `s01_generation.sample_plan`, `s01_generation.simulate` |
| `s02_processing/` | eCADSTAR outputs to spectrum and map arrays; merging of the simulation campaigns | `s02_processing.process`, `s02_processing.merge` |
| `s03_normalization/` | per-frequency normalization and its statistics | `s03_normalization.normalize` |
| `s04_dataset/` | packaging (indexes, split lists, test pool), layout store, dataset, data loader | `s04_dataset.package` |
| `s05_model/` | the VAE, its blocks and graph networks | (import only) |
| `s06_losses/` | all loss terms of the training objective | (import only) |
| `s07_training/` | phase 1 (the VAE) and phase 2 (the placement-only encoder) | `s07_training.train`, `s07_training.train_phase2` |
| `s08_monitoring/` | metric files, cross-frequency check during training, run log | (import only) |
| `s09_inference/` | loading the model, predictions, placement-only encoder, its safety margin | `s09_inference.margin` |
| `s10_search/` | guided latent search, latent statistics per budget | `s10_search.latent_stats`, `s10_search.search` |
| `s11_assistant/` | check of a proposed placement and the nearest passing placement | `s11_assistant.suggest` |
| `s12_evaluation/` | the Chapter 4 evaluations, one script per evaluation | `s12_evaluation.run_search`, `s12_evaluation.check_*` |

## External inputs

The dataset, the trained weights and the eCADSTAR board project are **not** included. Their
locations are the settings in `common/settings.py`, the only file with absolute paths:

| Setting | What it points to |
|---|---|
| `DATA_DIR` | the packaged dataset (`raw/`, `normalized/`, `test_pool/`, split lists, indexes) |
| `WEIGHTS_DIR`, `CHECKPOINT`, `HEAD` | the trained VAE, the distiller and the fine-tuned head (`vae_route_finetune/`), the head's margin file, `latent_cov_per_k.json` |
| `INDEPENDENT_DIR` | the simulations of the 480 independent placements |
| `SAVED_RESULTS_DIR` | the Chapter 4 results as quoted in the thesis (read when a check needs another check's result that this run has not made) |
| `SIM_STORE_DIR` | the eCADSTAR simulations of the Chapter 4 checks (read only) |
| `PLAN_DIR`, `RAW_SIM_ROOT` | inputs of the sampling plan; raw eCADSTAR outputs of the simulation campaigns |
| `ECADSTAR_RUN`, `ENGINEER_EXE`, `ERF_PATH`, `EMC_DIR` | the eCADSTAR installation and board project on the Windows host |

Small inputs ship in `common/inputs/`: the board mask, the 231 frequency points, the slot positions,
the three target spectra (also rebuilt by `common.make_targets`), the bare-board spectrum, the
dataset's normalization statistics and metadata, and the training configuration (`config.yaml`,
JSON with `#` comment lines).

eCADSTAR is started only when `ECADSTAR_RUN = True` (on the Windows host, when no other simulation
runs). Otherwise every step that needs new simulations writes its PEB files and a `commands.txt`
and stops.

## Run order

sample_plan → simulate → process → merge → normalize → package → train → train_phase2 → margin →
latent_stats → search / suggest → run_search → check_* (check_accuracy first; check_search_ecad
after run_search and check_random_surrogate).

The dataset of the thesis is a merge of several simulation campaigns. This folder covers the last
sampling plan and the generic steps (simulation, processing, merge, normalization, packaging) that
every campaign went through, not the plans of the earlier campaigns.

## Outputs

Every output goes to `runs/` (`settings.OUTPUT_DIR`, or the environment variable
`THESIS_CODE_OUTPUT_DIR`), one subfolder per stage. Nothing is written elsewhere.

## Verification

In the full repository, `pytest tests/thesis_code` runs each stage of this copy and the original
code on the same inputs and compares their outputs (the Chapter 4 checks against the saved results
the thesis quotes).
