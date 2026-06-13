# VAE novelty test summary

Generated at: 2026-04-12 18:56:18

## Parameters

- checkpoint: experiments/exp027_sigma_reg_tuning/checkpoints/checkpoint_epoch_400.pt
- latent_dim: 32
- K: 5
- N: 50
- shared_temp: 1.5
- dataset_root: datasets/data_norm
- hm_pool: 16
- baseline_n: 200
- max_train: None
- force_cpu: False
- out_dir: scrap/generated_samples/K5_noveltyN50

## Report output

```
gen-dir:       scrap/generated_samples/K5_noveltyN50
dataset-root:  datasets/data_norm
K filter:      5
hm-pool:       16x16
Scanned 2000/30000 occupancy files...
Scanned 4000/30000 occupancy files...
Scanned 6000/30000 occupancy files...
Scanned 8000/30000 occupancy files...
Scanned 10000/30000 occupancy files...
Scanned 12000/30000 occupancy files...
Scanned 14000/30000 occupancy files...
Scanned 16000/30000 occupancy files...
Scanned 18000/30000 occupancy files...
Scanned 20000/30000 occupancy files...
Scanned 22000/30000 occupancy files...
Scanned 24000/30000 occupancy files...
Scanned 26000/30000 occupancy files...
Scanned 28000/30000 occupancy files...
Scanned 30000/30000 occupancy files...
Loading dataset: N=608 samples (dataset_root=datasets/data_norm)
Train N=608 | Generated M=50

Nearest-neighbor distance summary
  train→train baseline (combined): min=0.04559 p10=0.1079 med=0.1568 p90=0.1942 max=0.2496
  gen→train (combined):           min=4.288 p10=4.373 med=4.503 p90=4.735 max=6.396
  gen→train heatmap MSE:          min=4.168 p10=4.279 med=4.394 p90=4.631 max=6.246
  gen→train impedance MSE:        min=0.004694 p10=0.005794 med=0.00994 p90=0.01813 max=0.03554
  gen→train occupancy hamming:    min=2 med=5 max=8
  gen→train occupancy exact:      0/50

Wrote: scrap/generated_samples/K5_noveltyN50/novelty_report.csv
```
