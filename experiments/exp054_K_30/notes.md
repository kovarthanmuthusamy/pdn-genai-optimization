# Experiment: exp054_K_30

## Goal
400-epoch heatmap + impedance training on unified multifreq dataset (K≈30 focus via experiment design, not loss reweighting).

## Schedule (400 epochs)
| Epoch | Event |
|------:|-------|
| 20 | Cross-freq loss on |
| 80 | Impedance peak loss ramp + separate impedance LR |
| 80–119 | Peak/dual-topk impedance terms ramp to full |
| 240 | β (KL) anneal complete |
| 260 | Heatmap focus phase + protect heatmap in modality dropout |

## Loss stack
- tier_a: Pearson + grad + peak_loc + valley_loc
- log1p peak/valley (hotspot, max/min, centroid)
- Impedance: full spectrum + peak terms (ramped from ep 80)
- No physics losses or KAN spline regularization
- No decoder freeze, K/MHz sample weighting, append-tag / synthetic-blend / layout curriculum

## Off-anchor eval
- Single file: `metrics/off_anchor_eval.csv` (append per eval)
- MHz: 100, 270, 400 | 12 val batches | every 50 epochs

## Run
```bash
./experiments/exp054_K_30/run_train_gpu1.sh
```

## Results
(TBD)
