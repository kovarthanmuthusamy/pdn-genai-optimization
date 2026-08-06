# Simulated Real vs Generated — Heatmap Metrics

- **Rows**: 48 (one per freq × K × sample)
- **Source**: ECADStar `.map` in `K*/Real/` vs `data_sample_*/heatmap_physical.npy`
- **CSV**: `sim_compare_metrics.csv`
- **JSON**: `sim_compare_metrics.json`

**Primary QC** (tables below): `pearson_r`, `max_diff_ohm` (real−gen; − = gen higher), `pattern_mae`, `max_ratio`.
Use `mae_ohm` / `mape_pct` from CSV with care — often misleading at 10 MHz (FG Ω ≈ 0).

## Per-sample metrics (primary)

```
mhz | k  | sample | pearson_r | max_diff_ohm | pattern_mae | max_ratio | peak_loc_err_px | real_max_ohm | gen_max_ohm
----|----|--------|-----------|--------------|-------------|-----------|-----------------|--------------|------------
30  | 3  | 0      | 0.1924    | 1.333        | 0.2154      | 0.08302   | 88.39           | 1.454        | 0.1207     
30  | 4  | 0      | 0.9269    | 0.9528       | 0.2131      | 0.1679    | 63              | 1.145        | 0.1922     
30  | 5  | 0      | 0.7911    | 0.9595       | 0.2579      | 0.1807    | 0               | 1.171        | 0.2116     
30  | 6  | 0      | 0.9075    | 0.8872       | 0.1117      | 0.1754    | 0               | 1.076        | 0.1887     
30  | 7  | 0      | 0.9407    | 0.9249       | 0.07387     | 0.1685    | 0               | 1.112        | 0.1874     
30  | 8  | 0      | 0.947     | 0.9089       | 0.09171     | 0.1716    | 0               | 1.097        | 0.1883     
30  | 9  | 0      | 0.9631    | 0.8695       | 0.09369     | 0.1662    | 0               | 1.043        | 0.1733     
30  | 10 | 0      | 0.8859    | 0.8321       | 0.07668     | 0.1744    | 0               | 1.008        | 0.1758     
85  | 3  | 0      | 0.6625    | 8.372        | 0.374       | 0.2315    | 56.32           | 10.89        | 2.522      
85  | 4  | 0      | 0.8434    | 1.237        | 0.06877     | 0.7242    | 63              | 4.484        | 3.247      
85  | 5  | 0      | 0.9738    | 0.05776      | 0.01494     | 0.9848    | 0               | 3.797        | 3.739      
85  | 6  | 0      | 0.8687    | 0.08255      | 0.04503     | 0.9744    | 0               | 3.219        | 3.137      
85  | 7  | 0      | 0.9268    | 0.09124      | 0.01953     | 0.9735    | 0               | 3.438        | 3.347      
85  | 8  | 0      | 0.9438    | 0.3564       | 0.0201      | 0.8937    | 0               | 3.352        | 2.996      
85  | 9  | 0      | 0.9254    | 0.2629       | 0.03344     | 0.9122    | 0               | 2.996        | 2.733      
85  | 10 | 0      | 0.9299    | 0.3814       | 0.02853     | 0.8829    | 0               | 3.258        | 2.877      
160 | 3  | 0      | 0.9707    | 1.486        | 0.009647    | 0.7822    | 0               | 6.825        | 5.339      
160 | 4  | 0      | 0.9709    | 0.3963       | 0.02566     | 0.9301    | 0               | 5.672        | 5.276      
160 | 5  | 0      | 0.7039    | -1.841       | 0.07727     | 1.408     | 0               | 4.517        | 6.357      
160 | 6  | 0      | 0.3551    | 3.824        | 0.2132      | 0.5554    | 80.62           | 8.6          | 4.776      
160 | 7  | 0      | 0.1785    | 0.3905       | 0.2039      | 0.9229    | 53.34           | 5.067        | 4.676      
160 | 8  | 0      | 0.6399    | 23.04        | 0.1544      | 0.1753    | 60.01           | 27.93        | 4.897      
160 | 9  | 0      | 0.8888    | 1.875        | 0.04362     | 0.752     | 0               | 7.562        | 5.687      
160 | 10 | 0      | 0.881     | 5.111        | 0.05179     | 0.5802    | 0               | 12.17        | 7.062      
260 | 3  | 0      | 0.9383    | 6.056        | 0.05693     | 0.7237    | 60.13           | 21.92        | 15.86      
260 | 4  | 0      | 0.9408    | 3.812        | 0.04031     | 0.8646    | 1.414           | 28.15        | 24.34      
260 | 5  | 0      | 0.981     | -3.445       | 0.007856    | 1.439     | 0               | 7.84         | 11.29      
260 | 6  | 0      | 0.6234    | 7.173        | 0.1177      | 0.7566    | 0               | 29.47        | 22.3       
260 | 7  | 0      | 0.5567    | 11.11        | 0.1211      | 0.4369    | 62              | 19.73        | 8.618      
260 | 8  | 0      | 0.9432    | 3.756        | 0.03098     | 0.7659    | 0               | 16.05        | 12.29      
260 | 9  | 0      | 0.5429    | -9.196       | 0.1072      | 1.619     | 0               | 14.86        | 24.06      
260 | 10 | 0      | 0.7195    | -3.645       | 0.05511     | 1.253     | 63              | 14.4         | 18.05      
480 | 3  | 0      | 0.9281    | 0.8203       | 0.03149     | 0.9452    | 0               | 14.97        | 14.15      
480 | 4  | 0      | 0.8333    | 7.885        | 0.07282     | 0.5585    | 70.21           | 17.86        | 9.976      
480 | 5  | 0      | 0.9373    | 2.306        | 0.0335      | 0.8316    | 0               | 13.7         | 11.39      
480 | 6  | 0      | 0.9357    | -0.03587     | 0.01871     | 1.003     | 0               | 12.03        | 12.07      
480 | 7  | 0      | 0.8913    | 2.057        | 0.04938     | 0.8343    | 63              | 12.42        | 10.36      
480 | 8  | 0      | 0.9374    | 1.271        | 0.03918     | 0.898     | 0               | 12.46        | 11.19      
480 | 9  | 0      | 0.8729    | -0.1824      | 0.05        | 1.016     | 70.66           | 11.52        | 11.7       
480 | 10 | 0      | 0.9587    | 1.014        | 0.01811     | 0.9032    | 0               | 10.47        | 9.457      
530 | 3  | 0      | 0.6702    | 5.556        | 0.08449     | 0.738     | 29              | 21.21        | 15.65      
530 | 4  | 0      | 0.8152    | 12.03        | 0.09101     | 0.4531    | 60.75           | 21.99        | 9.965      
530 | 5  | 0      | 0.4914    | 12.76        | 0.1271      | 0.4718    | 59.68           | 24.16        | 11.4       
530 | 6  | 0      | 0.6117    | -1.888       | 0.1068      | 1.16      | 63              | 11.82        | 13.71      
530 | 7  | 0      | 0.4869    | 7.733        | 0.1182      | 0.5814    | 63              | 18.47        | 10.74      
530 | 8  | 0      | 0.5136    | -0.7497      | 0.09324     | 1.066     | 63              | 11.31        | 12.06      
530 | 9  | 0      | 0.4474    | 0.9635       | 0.1353      | 0.9031    | 19              | 9.939        | 8.976      
530 | 10 | 0      | 0.3974    | -5.157       | 0.1297      | 1.529     | 33.62           | 9.756        | 14.91      
```

## Mean by MHz (primary)

```
mhz | n | pearson_r_mean | max_diff_ohm_mean | pattern_mae_mean | max_ratio_mean | peak_loc_err_px_mean | real_max_ohm_mean | gen_max_ohm_mean
----|---|----------------|-------------------|------------------|----------------|----------------------|-------------------|-----------------
30  | 8 | 0.8193         | 0.9585            | 0.1418           | 0.161          | 18.92                | 1.138             | 0.1797          
85  | 8 | 0.8843         | 1.355             | 0.07555          | 0.8221         | 14.92                | 4.43              | 3.075           
160 | 8 | 0.6986         | 4.285             | 0.09745          | 0.7632         | 24.25                | 9.794             | 5.509           
260 | 8 | 0.7807         | 1.952             | 0.06715          | 0.9824         | 23.32                | 19.05             | 17.1            
480 | 8 | 0.9118         | 1.892             | 0.03915          | 0.8737         | 25.48                | 13.18             | 11.29           
530 | 8 | 0.5542         | 3.906             | 0.1107           | 0.8627         | 48.88                | 16.08             | 12.18           
```

## Agent copy block

```
sim_compare_metrics: n=48
  30.0MHz K3 s0: r=0.192 Δmax=+1.33Ω pattern_mae=0.215 max_ratio=0.083 real_max=1.45 gen_max=0.121 peak_err=88.4px
  30.0MHz K4 s0: r=0.927 Δmax=+0.953Ω pattern_mae=0.213 max_ratio=0.168 real_max=1.14 gen_max=0.192 peak_err=63.0px
  30.0MHz K5 s0: r=0.791 Δmax=+0.96Ω pattern_mae=0.258 max_ratio=0.181 real_max=1.17 gen_max=0.212 peak_err=0.0px
  30.0MHz K6 s0: r=0.907 Δmax=+0.887Ω pattern_mae=0.112 max_ratio=0.175 real_max=1.08 gen_max=0.189 peak_err=0.0px
  30.0MHz K7 s0: r=0.941 Δmax=+0.925Ω pattern_mae=0.074 max_ratio=0.168 real_max=1.11 gen_max=0.187 peak_err=0.0px
  30.0MHz K8 s0: r=0.947 Δmax=+0.909Ω pattern_mae=0.092 max_ratio=0.172 real_max=1.1 gen_max=0.188 peak_err=0.0px
  30.0MHz K9 s0: r=0.963 Δmax=+0.87Ω pattern_mae=0.094 max_ratio=0.166 real_max=1.04 gen_max=0.173 peak_err=0.0px
  30.0MHz K10 s0: r=0.886 Δmax=+0.832Ω pattern_mae=0.077 max_ratio=0.174 real_max=1.01 gen_max=0.176 peak_err=0.0px
  85.0MHz K3 s0: r=0.662 Δmax=+8.37Ω pattern_mae=0.374 max_ratio=0.232 real_max=10.9 gen_max=2.52 peak_err=56.3px
  85.0MHz K4 s0: r=0.843 Δmax=+1.24Ω pattern_mae=0.069 max_ratio=0.724 real_max=4.48 gen_max=3.25 peak_err=63.0px
  85.0MHz K5 s0: r=0.974 Δmax=+0.0578Ω pattern_mae=0.015 max_ratio=0.985 real_max=3.8 gen_max=3.74 peak_err=0.0px
  85.0MHz K6 s0: r=0.869 Δmax=+0.0825Ω pattern_mae=0.045 max_ratio=0.974 real_max=3.22 gen_max=3.14 peak_err=0.0px
  85.0MHz K7 s0: r=0.927 Δmax=+0.0912Ω pattern_mae=0.020 max_ratio=0.973 real_max=3.44 gen_max=3.35 peak_err=0.0px
  85.0MHz K8 s0: r=0.944 Δmax=+0.356Ω pattern_mae=0.020 max_ratio=0.894 real_max=3.35 gen_max=3 peak_err=0.0px
  85.0MHz K9 s0: r=0.925 Δmax=+0.263Ω pattern_mae=0.033 max_ratio=0.912 real_max=3 gen_max=2.73 peak_err=0.0px
  85.0MHz K10 s0: r=0.930 Δmax=+0.381Ω pattern_mae=0.029 max_ratio=0.883 real_max=3.26 gen_max=2.88 peak_err=0.0px
  160.0MHz K3 s0: r=0.971 Δmax=+1.49Ω pattern_mae=0.010 max_ratio=0.782 real_max=6.83 gen_max=5.34 peak_err=0.0px
  160.0MHz K4 s0: r=0.971 Δmax=+0.396Ω pattern_mae=0.026 max_ratio=0.930 real_max=5.67 gen_max=5.28 peak_err=0.0px
  160.0MHz K5 s0: r=0.704 Δmax=-1.84Ω pattern_mae=0.077 max_ratio=1.408 real_max=4.52 gen_max=6.36 peak_err=0.0px
  160.0MHz K6 s0: r=0.355 Δmax=+3.82Ω pattern_mae=0.213 max_ratio=0.555 real_max=8.6 gen_max=4.78 peak_err=80.6px
  160.0MHz K7 s0: r=0.179 Δmax=+0.39Ω pattern_mae=0.204 max_ratio=0.923 real_max=5.07 gen_max=4.68 peak_err=53.3px
  160.0MHz K8 s0: r=0.640 Δmax=+23Ω pattern_mae=0.154 max_ratio=0.175 real_max=27.9 gen_max=4.9 peak_err=60.0px
  160.0MHz K9 s0: r=0.889 Δmax=+1.88Ω pattern_mae=0.044 max_ratio=0.752 real_max=7.56 gen_max=5.69 peak_err=0.0px
  160.0MHz K10 s0: r=0.881 Δmax=+5.11Ω pattern_mae=0.052 max_ratio=0.580 real_max=12.2 gen_max=7.06 peak_err=0.0px
  260.0MHz K3 s0: r=0.938 Δmax=+6.06Ω pattern_mae=0.057 max_ratio=0.724 real_max=21.9 gen_max=15.9 peak_err=60.1px
  260.0MHz K4 s0: r=0.941 Δmax=+3.81Ω pattern_mae=0.040 max_ratio=0.865 real_max=28.2 gen_max=24.3 peak_err=1.4px
  260.0MHz K5 s0: r=0.981 Δmax=-3.44Ω pattern_mae=0.008 max_ratio=1.439 real_max=7.84 gen_max=11.3 peak_err=0.0px
  260.0MHz K6 s0: r=0.623 Δmax=+7.17Ω pattern_mae=0.118 max_ratio=0.757 real_max=29.5 gen_max=22.3 peak_err=0.0px
  260.0MHz K7 s0: r=0.557 Δmax=+11.1Ω pattern_mae=0.121 max_ratio=0.437 real_max=19.7 gen_max=8.62 peak_err=62.0px
  260.0MHz K8 s0: r=0.943 Δmax=+3.76Ω pattern_mae=0.031 max_ratio=0.766 real_max=16 gen_max=12.3 peak_err=0.0px
  260.0MHz K9 s0: r=0.543 Δmax=-9.2Ω pattern_mae=0.107 max_ratio=1.619 real_max=14.9 gen_max=24.1 peak_err=0.0px
  260.0MHz K10 s0: r=0.719 Δmax=-3.64Ω pattern_mae=0.055 max_ratio=1.253 real_max=14.4 gen_max=18 peak_err=63.0px
  480.0MHz K3 s0: r=0.928 Δmax=+0.82Ω pattern_mae=0.031 max_ratio=0.945 real_max=15 gen_max=14.1 peak_err=0.0px
  480.0MHz K4 s0: r=0.833 Δmax=+7.88Ω pattern_mae=0.073 max_ratio=0.559 real_max=17.9 gen_max=9.98 peak_err=70.2px
  480.0MHz K5 s0: r=0.937 Δmax=+2.31Ω pattern_mae=0.033 max_ratio=0.832 real_max=13.7 gen_max=11.4 peak_err=0.0px
  480.0MHz K6 s0: r=0.936 Δmax=-0.0359Ω pattern_mae=0.019 max_ratio=1.003 real_max=12 gen_max=12.1 peak_err=0.0px
  480.0MHz K7 s0: r=0.891 Δmax=+2.06Ω pattern_mae=0.049 max_ratio=0.834 real_max=12.4 gen_max=10.4 peak_err=63.0px
  480.0MHz K8 s0: r=0.937 Δmax=+1.27Ω pattern_mae=0.039 max_ratio=0.898 real_max=12.5 gen_max=11.2 peak_err=0.0px
  480.0MHz K9 s0: r=0.873 Δmax=-0.182Ω pattern_mae=0.050 max_ratio=1.016 real_max=11.5 gen_max=11.7 peak_err=70.7px
  480.0MHz K10 s0: r=0.959 Δmax=+1.01Ω pattern_mae=0.018 max_ratio=0.903 real_max=10.5 gen_max=9.46 peak_err=0.0px
  530.0MHz K3 s0: r=0.670 Δmax=+5.56Ω pattern_mae=0.084 max_ratio=0.738 real_max=21.2 gen_max=15.7 peak_err=29.0px
  530.0MHz K4 s0: r=0.815 Δmax=+12Ω pattern_mae=0.091 max_ratio=0.453 real_max=22 gen_max=9.97 peak_err=60.7px
  530.0MHz K5 s0: r=0.491 Δmax=+12.8Ω pattern_mae=0.127 max_ratio=0.472 real_max=24.2 gen_max=11.4 peak_err=59.7px
  530.0MHz K6 s0: r=0.612 Δmax=-1.89Ω pattern_mae=0.107 max_ratio=1.160 real_max=11.8 gen_max=13.7 peak_err=63.0px
  530.0MHz K7 s0: r=0.487 Δmax=+7.73Ω pattern_mae=0.118 max_ratio=0.581 real_max=18.5 gen_max=10.7 peak_err=63.0px
  530.0MHz K8 s0: r=0.514 Δmax=-0.75Ω pattern_mae=0.093 max_ratio=1.066 real_max=11.3 gen_max=12.1 peak_err=63.0px
  530.0MHz K9 s0: r=0.447 Δmax=+0.963Ω pattern_mae=0.135 max_ratio=0.903 real_max=9.94 gen_max=8.98 peak_err=19.0px
  530.0MHz K10 s0: r=0.397 Δmax=-5.16Ω pattern_mae=0.130 max_ratio=1.529 real_max=9.76 gen_max=14.9 peak_err=33.6px
```
