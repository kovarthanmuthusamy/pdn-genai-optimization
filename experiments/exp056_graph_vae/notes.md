# Experiment: exp056_hard_occ

## Goal
exp054 successor: **binary occupancy before heatmap decode** so inference/QC matches CAD discrete layouts.
Fresh 400-epoch training on unified multifreq dataset (K≈30 focus).

## vs exp054
| Change | exp054 | exp056 |
|--------|--------|--------|
| Decode occupancy | soft probs OK in decoder cond | **top-K binary** before decode (`occupancy_binary_decode`) |
| Cross-freq encode skips | off | off (same) |
| Training | 800ep resume path | **fresh 400ep** |

## Binary occupancy
- `occupancy_binary.py`: `topk_occ_binary`, `occupancy_for_heatmap_decode`
- Model `decode()` binarizes when `binary_occupancy_decode=True` (default)
- Off-anchor eval uses same binary occ for encode_cross + layout_cross
- PEB/CAD sim should use the **same** binary vector as decode conditioning

## Schedule (400 epochs)
| Epoch | Event |
|------:|-------|
| 20 | Cross-freq loss on |
| 80 | Impedance peak loss ramp |
| 260 | Heatmap focus phase |

## Run
```bash
./experiments/exp056_hard_occ/run_train_gpu1.sh
# or DDP:
./experiments/exp056_hard_occ/run_train_ddp.sh
```

## Results
(TBD)
