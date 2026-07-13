# Simulated Real vs Generated — Heatmap Metrics

- **Rows**: 66 (one per freq × K × sample)
- **Source**: ECADStar `.map` in `K*/Real/` vs `data_sample_*/heatmap_physical.npy`
- **CSV**: `sim_compare_metrics.csv`
- **JSON**: `sim_compare_metrics.json`

**Primary QC** (tables below): `pearson_r`, `max_diff_ohm` (real−gen; − = gen higher), `pattern_mae`, `max_ratio`.
Use `mae_ohm` / `mape_pct` from CSV with care — often misleading at 10 MHz (FG Ω ≈ 0).

## Per-sample metrics (primary)

```
mhz | k  | sample | pearson_r | max_diff_ohm | pattern_mae | max_ratio | peak_loc_err_px | real_max_ohm | gen_max_ohm
----|----|--------|-----------|--------------|-------------|-----------|-----------------|--------------|------------
20  | 7  | 0      | 0.9685    | 0.4717       | 0.01688     | 0.3373    | 0               | 0.7118       | 0.2401     
20  | 8  | 0      | 0.9807    | 0.4756       | 0.0165      | 0.3374    | 0               | 0.7178       | 0.2422     
20  | 9  | 0      | 0.9739    | 0.4472       | 0.02777     | 0.3233    | 0               | 0.6609       | 0.2136     
20  | 10 | 0      | 0.9874    | 0.4664       | 0.01521     | 0.3273    | 0               | 0.6933       | 0.2269     
20  | 11 | 0      | 0.9831    | 0.4529       | 0.00967     | 0.3166    | 63              | 0.6627       | 0.2098     
20  | 12 | 0      | 0.9758    | 0.4458       | 0.01654     | 0.3398    | 0               | 0.6753       | 0.2295     
20  | 13 | 0      | 0.9818    | 0.4873       | 0.01809     | 0.294     | 0               | 0.6902       | 0.2029     
20  | 14 | 0      | 0.9689    | 0.435        | 0.01704     | 0.3322    | 0               | 0.6513       | 0.2164     
20  | 15 | 0      | 0.9823    | 0.4573       | 0.02013     | 0.3136    | 63              | 0.6661       | 0.2089     
20  | 16 | 0      | 0.97      | 0.4151       | 0.0198      | 0.3332    | 63              | 0.6225       | 0.2074     
20  | 17 | 0      | 0.9882    | 0.4558       | 0.0169      | 0.304     | 0               | 0.6549       | 0.1991     
50  | 7  | 0      | 0.9798    | -0.5036      | 0.00784     | 1.261     | 0               | 1.93         | 2.434      
50  | 8  | 0      | 0.9814    | -0.6123      | 0.005529    | 1.33      | 0               | 1.853        | 2.466      
50  | 9  | 0      | 0.9874    | -0.5561      | 0.003152    | 1.328     | 0               | 1.696        | 2.252      
50  | 10 | 0      | 0.9958    | -0.5167      | 0.0002271   | 1.29      | 0               | 1.78         | 2.297      
50  | 11 | 0      | 0.9838    | -0.4613      | 0.001135    | 1.272     | 63              | 1.698        | 2.159      
50  | 12 | 0      | 0.9833    | -0.6118      | 0.002931    | 1.355     | 0               | 1.722        | 2.333      
50  | 13 | 0      | 0.9871    | -0.3082      | 0.004129    | 1.176     | 0               | 1.752        | 2.06       
50  | 14 | 0      | 0.9647    | -0.6496      | 0.01194     | 1.391     | 0               | 1.663        | 2.313      
50  | 15 | 0      | 0.9855    | -0.5012      | 0.004251    | 1.293     | 0               | 1.713        | 2.214      
50  | 16 | 0      | 0.9665    | -0.5703      | 0.01342     | 1.363     | 63              | 1.571        | 2.141      
50  | 17 | 0      | 0.9893    | -0.4236      | 0.002297    | 1.254     | 0               | 1.67         | 2.094      
140 | 7  | 0      | 0.9335    | 3.321        | 0.078       | 0.6795    | 0               | 10.36        | 7.039      
140 | 8  | 0      | 0.9698    | 0.8746       | 0.03575     | 0.898     | 0               | 8.575        | 7.701      
140 | 9  | 0      | 0.9263    | 0.2817       | 0.03729     | 0.9569    | 63              | 6.532        | 6.25       
140 | 10 | 0      | 0.9929    | -0.5038      | 0.000262    | 1.079     | 0               | 6.392        | 6.895      
140 | 11 | 0      | 0.9822    | -1.594       | 0.004936    | 1.284     | 0               | 5.616        | 7.209      
140 | 12 | 0      | 0.9923    | -1.622       | 0.002689    | 1.268     | 0               | 6.05         | 7.672      
140 | 13 | 0      | 0.9933    | -0.3304      | 0.000733    | 1.054     | 0               | 6.164        | 6.494      
140 | 14 | 0      | 0.9588    | -1.259       | 0.01497     | 1.241     | 0               | 5.231        | 6.49       
140 | 15 | 0      | 0.984     | -0.6063      | 0.006863    | 1.109     | 0               | 5.578        | 6.184      
140 | 16 | 0      | 0.9665    | -0.9562      | 0.0209      | 1.203     | 63              | 4.704        | 5.66       
140 | 17 | 0      | 0.9766    | -0.8745      | 0.01562     | 1.184     | 0               | 4.754        | 5.629      
260 | 7  | 0      | 0.7215    | 8.897        | 0.1207      | 0.547     | 60.03           | 19.64        | 10.75      
260 | 8  | 0      | 0.6786    | 7.374        | 0.07207     | 0.6651    | 0               | 22.02        | 14.65      
260 | 9  | 0      | 0.9439    | 5.851        | 0.03066     | 0.7494    | 0               | 23.35        | 17.49      
260 | 10 | 0      | 0.8668    | 1.855        | 0.06856     | 0.8499    | 81.34           | 12.36        | 10.5       
260 | 11 | 0      | 0.7465    | -9.496       | 0.106       | 2.332     | 62              | 7.132        | 16.63      
260 | 12 | 0      | 0.9865    | -0.5351      | 0.02        | 1.05      | 0               | 10.68        | 11.21      
260 | 13 | 0      | 0.9574    | 17.95        | 0.04565     | 0.4593    | 0               | 33.2         | 15.25      
260 | 14 | 0      | 0.9156    | -1.379       | 0.02917     | 1.104     | 0               | 13.22        | 14.6       
260 | 15 | 0      | 0.9373    | 11.87        | 0.03972     | 0.597     | 4.123           | 29.45        | 17.58      
260 | 16 | 0      | 0.9008    | -0.9349      | 0.1246      | 1.047     | 63              | 19.71        | 20.64      
260 | 17 | 0      | 0.9294    | 9.212        | 0.07098     | 0.6895    | 85.56           | 29.66        | 20.45      
440 | 7  | 0      | 0.4737    | 12.1         | 0.1181      | 0.5074    | 88.39           | 24.56        | 12.46      
440 | 8  | 0      | 0.6699    | 5.853        | 0.08908     | 0.6226    | 64.03           | 15.51        | 9.657      
440 | 9  | 0      | 0.9       | 3.459        | 0.0316      | 0.7628    | 88.39           | 14.58        | 11.12      
440 | 10 | 0      | 0.8815    | 6.864        | 0.0299      | 0.6143    | 0               | 17.8         | 10.93      
440 | 11 | 0      | 0.9315    | 0.5299       | 0.01898     | 0.9567    | 62              | 12.24        | 11.71      
440 | 12 | 0      | 0.4762    | -5.925       | 0.1156      | 1.871     | 32              | 6.802        | 12.73      
440 | 13 | 0      | 0.9346    | 0.2895       | 0.01948     | 0.9706    | 88.39           | 9.834        | 9.545      
440 | 14 | 0      | 0.8609    | 1.13         | 0.05681     | 0.8954    | 63              | 10.8         | 9.671      
440 | 15 | 0      | 0.9444    | 0.5761       | 0.02375     | 0.9509    | 62              | 11.72        | 11.15      
440 | 16 | 0      | 0.9293    | -0.97        | 0.01971     | 1.09      | 0               | 10.83        | 11.8       
440 | 17 | 0      | 0.8148    | 1.006        | 0.04878     | 0.9232    | 0               | 13.1         | 12.09      
530 | 7  | 0      | 0.6273    | 12.7         | 0.08765     | 0.5281    | 60.21           | 26.91        | 14.21      
530 | 8  | 0      | 0.6991    | -0.6107      | 0.06926     | 1.05      | 34.54           | 12.27        | 12.88      
530 | 9  | 0      | 0.5705    | -1.327       | 0.1015      | 1.118     | 63              | 11.23        | 12.55      
530 | 10 | 0      | 0.8654    | -0.388       | 0.04514     | 1.031     | 31.02           | 12.6         | 12.98      
530 | 11 | 0      | 0.471     | -4.797       | 0.1024      | 1.521     | 63              | 9.209        | 14.01      
530 | 12 | 0      | 0.7106    | 4.448        | 0.0733      | 0.743     | 59.68           | 17.31        | 12.86      
530 | 13 | 0      | 0.5932    | 7.796        | 0.09159     | 0.6072    | 64.56           | 19.85        | 12.05      
530 | 14 | 0      | 0.8725    | 2.387        | 0.05737     | 0.8359    | 0               | 14.55        | 12.16      
530 | 15 | 0      | 0.8921    | 8.753        | 0.04638     | 0.6268    | 30              | 23.45        | 14.7       
530 | 16 | 0      | 0.7946    | 1.166        | 0.0817      | 0.918     | 0               | 14.23        | 13.06      
530 | 17 | 0      | 0.7949    | 9.35         | 0.07135     | 0.576     | 70.24           | 22.05        | 12.7       
```

## Mean by MHz (primary)

```
mhz | n  | pearson_r_mean | max_diff_ohm_mean | pattern_mae_mean | max_ratio_mean | peak_loc_err_px_mean | real_max_ohm_mean | gen_max_ohm_mean
----|----|----------------|-------------------|------------------|----------------|----------------------|-------------------|-----------------
20  | 11 | 0.9782         | 0.4555            | 0.01769          | 0.3235         | 17.18                | 0.6734            | 0.2179          
50  | 11 | 0.9822         | -0.5195           | 0.005168         | 1.301          | 11.45                | 1.732             | 2.251           
140 | 11 | 0.9706         | -0.2972           | 0.01982          | 1.087          | 11.45                | 6.36              | 6.657           
260 | 11 | 0.8713         | 4.606             | 0.06619          | 0.9173         | 32.37                | 20.04             | 15.43           
440 | 11 | 0.8015         | 2.264             | 0.05198          | 0.924          | 49.84                | 13.43             | 11.17           
530 | 11 | 0.7174         | 3.589             | 0.07524          | 0.8686         | 43.29                | 16.69             | 13.11           
```

## Agent copy block

```
sim_compare_metrics: n=66
  20.0MHz K7 s0: r=0.968 Δmax=+0.472Ω pattern_mae=0.017 max_ratio=0.337 real_max=0.712 gen_max=0.24 peak_err=0.0px
  20.0MHz K8 s0: r=0.981 Δmax=+0.476Ω pattern_mae=0.017 max_ratio=0.337 real_max=0.718 gen_max=0.242 peak_err=0.0px
  20.0MHz K9 s0: r=0.974 Δmax=+0.447Ω pattern_mae=0.028 max_ratio=0.323 real_max=0.661 gen_max=0.214 peak_err=0.0px
  20.0MHz K10 s0: r=0.987 Δmax=+0.466Ω pattern_mae=0.015 max_ratio=0.327 real_max=0.693 gen_max=0.227 peak_err=0.0px
  20.0MHz K11 s0: r=0.983 Δmax=+0.453Ω pattern_mae=0.010 max_ratio=0.317 real_max=0.663 gen_max=0.21 peak_err=63.0px
  20.0MHz K12 s0: r=0.976 Δmax=+0.446Ω pattern_mae=0.017 max_ratio=0.340 real_max=0.675 gen_max=0.229 peak_err=0.0px
  20.0MHz K13 s0: r=0.982 Δmax=+0.487Ω pattern_mae=0.018 max_ratio=0.294 real_max=0.69 gen_max=0.203 peak_err=0.0px
  20.0MHz K14 s0: r=0.969 Δmax=+0.435Ω pattern_mae=0.017 max_ratio=0.332 real_max=0.651 gen_max=0.216 peak_err=0.0px
  20.0MHz K15 s0: r=0.982 Δmax=+0.457Ω pattern_mae=0.020 max_ratio=0.314 real_max=0.666 gen_max=0.209 peak_err=63.0px
  20.0MHz K16 s0: r=0.970 Δmax=+0.415Ω pattern_mae=0.020 max_ratio=0.333 real_max=0.623 gen_max=0.207 peak_err=63.0px
  20.0MHz K17 s0: r=0.988 Δmax=+0.456Ω pattern_mae=0.017 max_ratio=0.304 real_max=0.655 gen_max=0.199 peak_err=0.0px
  50.0MHz K7 s0: r=0.980 Δmax=-0.504Ω pattern_mae=0.008 max_ratio=1.261 real_max=1.93 gen_max=2.43 peak_err=0.0px
  50.0MHz K8 s0: r=0.981 Δmax=-0.612Ω pattern_mae=0.006 max_ratio=1.330 real_max=1.85 gen_max=2.47 peak_err=0.0px
  50.0MHz K9 s0: r=0.987 Δmax=-0.556Ω pattern_mae=0.003 max_ratio=1.328 real_max=1.7 gen_max=2.25 peak_err=0.0px
  50.0MHz K10 s0: r=0.996 Δmax=-0.517Ω pattern_mae=0.000 max_ratio=1.290 real_max=1.78 gen_max=2.3 peak_err=0.0px
  50.0MHz K11 s0: r=0.984 Δmax=-0.461Ω pattern_mae=0.001 max_ratio=1.272 real_max=1.7 gen_max=2.16 peak_err=63.0px
  50.0MHz K12 s0: r=0.983 Δmax=-0.612Ω pattern_mae=0.003 max_ratio=1.355 real_max=1.72 gen_max=2.33 peak_err=0.0px
  50.0MHz K13 s0: r=0.987 Δmax=-0.308Ω pattern_mae=0.004 max_ratio=1.176 real_max=1.75 gen_max=2.06 peak_err=0.0px
  50.0MHz K14 s0: r=0.965 Δmax=-0.65Ω pattern_mae=0.012 max_ratio=1.391 real_max=1.66 gen_max=2.31 peak_err=0.0px
  50.0MHz K15 s0: r=0.986 Δmax=-0.501Ω pattern_mae=0.004 max_ratio=1.293 real_max=1.71 gen_max=2.21 peak_err=0.0px
  50.0MHz K16 s0: r=0.966 Δmax=-0.57Ω pattern_mae=0.013 max_ratio=1.363 real_max=1.57 gen_max=2.14 peak_err=63.0px
  50.0MHz K17 s0: r=0.989 Δmax=-0.424Ω pattern_mae=0.002 max_ratio=1.254 real_max=1.67 gen_max=2.09 peak_err=0.0px
  140.0MHz K7 s0: r=0.933 Δmax=+3.32Ω pattern_mae=0.078 max_ratio=0.679 real_max=10.4 gen_max=7.04 peak_err=0.0px
  140.0MHz K8 s0: r=0.970 Δmax=+0.875Ω pattern_mae=0.036 max_ratio=0.898 real_max=8.58 gen_max=7.7 peak_err=0.0px
  140.0MHz K9 s0: r=0.926 Δmax=+0.282Ω pattern_mae=0.037 max_ratio=0.957 real_max=6.53 gen_max=6.25 peak_err=63.0px
  140.0MHz K10 s0: r=0.993 Δmax=-0.504Ω pattern_mae=0.000 max_ratio=1.079 real_max=6.39 gen_max=6.9 peak_err=0.0px
  140.0MHz K11 s0: r=0.982 Δmax=-1.59Ω pattern_mae=0.005 max_ratio=1.284 real_max=5.62 gen_max=7.21 peak_err=0.0px
  140.0MHz K12 s0: r=0.992 Δmax=-1.62Ω pattern_mae=0.003 max_ratio=1.268 real_max=6.05 gen_max=7.67 peak_err=0.0px
  140.0MHz K13 s0: r=0.993 Δmax=-0.33Ω pattern_mae=0.001 max_ratio=1.054 real_max=6.16 gen_max=6.49 peak_err=0.0px
  140.0MHz K14 s0: r=0.959 Δmax=-1.26Ω pattern_mae=0.015 max_ratio=1.241 real_max=5.23 gen_max=6.49 peak_err=0.0px
  140.0MHz K15 s0: r=0.984 Δmax=-0.606Ω pattern_mae=0.007 max_ratio=1.109 real_max=5.58 gen_max=6.18 peak_err=0.0px
  140.0MHz K16 s0: r=0.967 Δmax=-0.956Ω pattern_mae=0.021 max_ratio=1.203 real_max=4.7 gen_max=5.66 peak_err=63.0px
  140.0MHz K17 s0: r=0.977 Δmax=-0.874Ω pattern_mae=0.016 max_ratio=1.184 real_max=4.75 gen_max=5.63 peak_err=0.0px
  260.0MHz K7 s0: r=0.722 Δmax=+8.9Ω pattern_mae=0.121 max_ratio=0.547 real_max=19.6 gen_max=10.7 peak_err=60.0px
  260.0MHz K8 s0: r=0.679 Δmax=+7.37Ω pattern_mae=0.072 max_ratio=0.665 real_max=22 gen_max=14.6 peak_err=0.0px
  260.0MHz K9 s0: r=0.944 Δmax=+5.85Ω pattern_mae=0.031 max_ratio=0.749 real_max=23.3 gen_max=17.5 peak_err=0.0px
  260.0MHz K10 s0: r=0.867 Δmax=+1.86Ω pattern_mae=0.069 max_ratio=0.850 real_max=12.4 gen_max=10.5 peak_err=81.3px
  260.0MHz K11 s0: r=0.746 Δmax=-9.5Ω pattern_mae=0.106 max_ratio=2.332 real_max=7.13 gen_max=16.6 peak_err=62.0px
  260.0MHz K12 s0: r=0.987 Δmax=-0.535Ω pattern_mae=0.020 max_ratio=1.050 real_max=10.7 gen_max=11.2 peak_err=0.0px
  260.0MHz K13 s0: r=0.957 Δmax=+18Ω pattern_mae=0.046 max_ratio=0.459 real_max=33.2 gen_max=15.2 peak_err=0.0px
  260.0MHz K14 s0: r=0.916 Δmax=-1.38Ω pattern_mae=0.029 max_ratio=1.104 real_max=13.2 gen_max=14.6 peak_err=0.0px
  260.0MHz K15 s0: r=0.937 Δmax=+11.9Ω pattern_mae=0.040 max_ratio=0.597 real_max=29.5 gen_max=17.6 peak_err=4.1px
  260.0MHz K16 s0: r=0.901 Δmax=-0.935Ω pattern_mae=0.125 max_ratio=1.047 real_max=19.7 gen_max=20.6 peak_err=63.0px
  260.0MHz K17 s0: r=0.929 Δmax=+9.21Ω pattern_mae=0.071 max_ratio=0.689 real_max=29.7 gen_max=20.5 peak_err=85.6px
  440.0MHz K7 s0: r=0.474 Δmax=+12.1Ω pattern_mae=0.118 max_ratio=0.507 real_max=24.6 gen_max=12.5 peak_err=88.4px
  440.0MHz K8 s0: r=0.670 Δmax=+5.85Ω pattern_mae=0.089 max_ratio=0.623 real_max=15.5 gen_max=9.66 peak_err=64.0px
  440.0MHz K9 s0: r=0.900 Δmax=+3.46Ω pattern_mae=0.032 max_ratio=0.763 real_max=14.6 gen_max=11.1 peak_err=88.4px
  440.0MHz K10 s0: r=0.882 Δmax=+6.86Ω pattern_mae=0.030 max_ratio=0.614 real_max=17.8 gen_max=10.9 peak_err=0.0px
  440.0MHz K11 s0: r=0.931 Δmax=+0.53Ω pattern_mae=0.019 max_ratio=0.957 real_max=12.2 gen_max=11.7 peak_err=62.0px
  440.0MHz K12 s0: r=0.476 Δmax=-5.92Ω pattern_mae=0.116 max_ratio=1.871 real_max=6.8 gen_max=12.7 peak_err=32.0px
  440.0MHz K13 s0: r=0.935 Δmax=+0.289Ω pattern_mae=0.019 max_ratio=0.971 real_max=9.83 gen_max=9.54 peak_err=88.4px
  440.0MHz K14 s0: r=0.861 Δmax=+1.13Ω pattern_mae=0.057 max_ratio=0.895 real_max=10.8 gen_max=9.67 peak_err=63.0px
  440.0MHz K15 s0: r=0.944 Δmax=+0.576Ω pattern_mae=0.024 max_ratio=0.951 real_max=11.7 gen_max=11.1 peak_err=62.0px
  440.0MHz K16 s0: r=0.929 Δmax=-0.97Ω pattern_mae=0.020 max_ratio=1.090 real_max=10.8 gen_max=11.8 peak_err=0.0px
  440.0MHz K17 s0: r=0.815 Δmax=+1.01Ω pattern_mae=0.049 max_ratio=0.923 real_max=13.1 gen_max=12.1 peak_err=0.0px
  530.0MHz K7 s0: r=0.627 Δmax=+12.7Ω pattern_mae=0.088 max_ratio=0.528 real_max=26.9 gen_max=14.2 peak_err=60.2px
  530.0MHz K8 s0: r=0.699 Δmax=-0.611Ω pattern_mae=0.069 max_ratio=1.050 real_max=12.3 gen_max=12.9 peak_err=34.5px
  530.0MHz K9 s0: r=0.571 Δmax=-1.33Ω pattern_mae=0.101 max_ratio=1.118 real_max=11.2 gen_max=12.6 peak_err=63.0px
  530.0MHz K10 s0: r=0.865 Δmax=-0.388Ω pattern_mae=0.045 max_ratio=1.031 real_max=12.6 gen_max=13 peak_err=31.0px
  530.0MHz K11 s0: r=0.471 Δmax=-4.8Ω pattern_mae=0.102 max_ratio=1.521 real_max=9.21 gen_max=14 peak_err=63.0px
  530.0MHz K12 s0: r=0.711 Δmax=+4.45Ω pattern_mae=0.073 max_ratio=0.743 real_max=17.3 gen_max=12.9 peak_err=59.7px
  530.0MHz K13 s0: r=0.593 Δmax=+7.8Ω pattern_mae=0.092 max_ratio=0.607 real_max=19.8 gen_max=12.1 peak_err=64.6px
  530.0MHz K14 s0: r=0.872 Δmax=+2.39Ω pattern_mae=0.057 max_ratio=0.836 real_max=14.5 gen_max=12.2 peak_err=0.0px
  530.0MHz K15 s0: r=0.892 Δmax=+8.75Ω pattern_mae=0.046 max_ratio=0.627 real_max=23.5 gen_max=14.7 peak_err=30.0px
  530.0MHz K16 s0: r=0.795 Δmax=+1.17Ω pattern_mae=0.082 max_ratio=0.918 real_max=14.2 gen_max=13.1 peak_err=0.0px
  530.0MHz K17 s0: r=0.795 Δmax=+9.35Ω pattern_mae=0.071 max_ratio=0.576 real_max=22.1 gen_max=12.7 peak_err=70.2px
```
