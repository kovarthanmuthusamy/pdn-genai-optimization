# Simulated Real vs Generated — Heatmap Metrics

- **Rows**: 55 (one per freq × K × sample)
- **Source**: ECADStar `.map` in `K*/Real/` vs `data_sample_*/heatmap_physical.npy`
- **CSV**: `sim_compare_metrics.csv`
- **JSON**: `sim_compare_metrics.json`

**Primary QC** (tables below): `pearson_r`, `pattern_mae`, `max_ratio`.
Use `mae_ohm` / `mape_pct` from CSV with care — often misleading at 10 MHz (FG Ω ≈ 0).

## Per-sample metrics (primary)

```
mhz | k  | sample | pearson_r | pattern_mae | max_ratio | peak_loc_err_px | real_max_ohm | gen_max_ohm
----|----|--------|-----------|-------------|-----------|-----------------|--------------|------------
10  | 7  | 0      | 0.9042    | 0.07778     | 0.5998    | 0               | 0.3493       | 0.2095     
10  | 8  | 0      | 0.9651    | 0.04336     | 0.6452    | 0               | 0.3523       | 0.2273     
10  | 9  | 0      | 0.9422    | 0.0746      | 0.618     | 63              | 0.3343       | 0.2066     
10  | 10 | 0      | 0.8103    | 0.09917     | 0.7       | 1               | 0.3462       | 0.2423     
10  | 11 | 0      | 0.9695    | 0.06523     | 0.6396    | 0               | 0.3519       | 0.2251     
10  | 12 | 0      | 0.93      | 0.04983     | 0.6204    | 63              | 0.3465       | 0.215      
10  | 13 | 0      | 0.8939    | 0.1085      | 0.6881    | 1               | 0.3621       | 0.2491     
10  | 14 | 0      | 0.9053    | 0.1067      | 0.7278    | 1               | 0.3443       | 0.2506     
10  | 15 | 0      | 0.9286    | 0.04239     | 0.602     | 63              | 0.3522       | 0.212      
10  | 16 | 0      | 0.9438    | 0.04417     | 0.6019    | 0               | 0.3355       | 0.2019     
10  | 17 | 0      | 0.9564    | 0.05219     | 0.622     | 0               | 0.3485       | 0.2168     
70  | 7  | 0      | 0.9408    | 0.04222     | 0.7579    | 63              | 2.849        | 2.159      
70  | 8  | 0      | 0.9847    | 0.002764    | 0.7979    | 0               | 2.716        | 2.167      
70  | 9  | 0      | 0.9724    | 0.02487     | 0.7514    | 0               | 2.62         | 1.969      
70  | 10 | 0      | 0.7901    | 0.08499     | 0.8265    | 2               | 2.569        | 2.123      
70  | 11 | 0      | 0.9845    | 0.01056     | 0.8149    | 0               | 2.642        | 2.153      
70  | 12 | 0      | 0.9919    | 0.008267    | 0.8337    | 0               | 2.636        | 2.197      
70  | 13 | 0      | 0.9565    | 0.04284     | 1.032     | 0               | 2.797        | 2.886      
70  | 14 | 0      | 0.9768    | 0.01164     | 0.8911    | 0               | 2.628        | 2.342      
70  | 15 | 0      | 0.9373    | 0.02728     | 0.805     | 63              | 2.608        | 2.1        
70  | 16 | 0      | 0.9848    | 0.007714    | 0.8171    | 0               | 2.469        | 2.017      
70  | 17 | 0      | 0.9799    | 0.01314     | 0.8536    | 0               | 2.618        | 2.235      
120 | 7  | 0      | 0.7097    | 0.1014      | 0.5953    | 63              | 6.676        | 3.975      
120 | 8  | 0      | 0.8382    | 0.06201     | 0.679     | 63              | 5.92         | 4.02       
120 | 9  | 0      | 0.6671    | 0.1248      | 0.7168    | 63              | 5.725        | 4.104      
120 | 10 | 0      | 0.783     | 0.1185      | 0.704     | 8.602           | 5.036        | 3.545      
120 | 11 | 0      | 0.968     | 0.02662     | 0.78      | 44              | 5.383        | 4.199      
120 | 12 | 0      | 0.9739    | 0.02074     | 0.8175    | 63              | 5.381        | 4.399      
120 | 13 | 0      | 0.9825    | 0.02752     | 1.046     | 0               | 6.539        | 6.839      
120 | 14 | 0      | 0.9472    | 0.02499     | 1.415     | 1               | 5.246        | 7.425      
120 | 15 | 0      | 0.9256    | 0.03423     | 0.894     | 63              | 5.142        | 4.597      
120 | 16 | 0      | 0.9846    | 0.009951    | 1.016     | 0               | 4.617        | 4.691      
120 | 17 | 0      | 0.9707    | 0.009431    | 0.9133    | 0               | 5.271        | 4.814      
270 | 7  | 0      | 0.3906    | 0.1334      | 1.214     | 88.39           | 12.18        | 14.78      
270 | 8  | 0      | 0.7859    | 0.0894      | 0.2796    | 4.123           | 50.81        | 14.2       
270 | 9  | 0      | 0.3707    | 0.1466      | 1.582     | 56.08           | 10.78        | 17.06      
270 | 10 | 0      | 0.3541    | 0.159       | 0.2642    | 64.03           | 44.46        | 11.75      
270 | 11 | 0      | 0.7215    | 0.1072      | 1.006     | 60.21           | 14.13        | 14.22      
270 | 12 | 0      | 0.2577    | 0.1673      | 1.057     | 84.17           | 17.02        | 17.98      
270 | 13 | 0      | 0.1517    | 0.1634      | 0.3892    | 55              | 74.18        | 28.87      
270 | 14 | 0      | 0.2172    | 0.1642      | 4.121     | 62.01           | 9.366        | 38.6       
270 | 15 | 0      | 0.4698    | 0.1077      | 0.7553    | 62              | 32.71        | 24.71      
270 | 16 | 0      | 0.2238    | 0.1534      | 1.31      | 63              | 13.55        | 17.75      
270 | 17 | 0      | 0.3389    | 0.1253      | 1.549     | 63              | 13.27        | 20.56      
400 | 7  | 0      | 0.8461    | 0.03839     | 0.8029    | 76.03           | 16.81        | 13.49      
400 | 8  | 0      | 0.8882    | 0.03608     | 1.282     | 76.03           | 8.487        | 10.88      
400 | 9  | 0      | 0.7716    | 0.05171     | 1.142     | 76.03           | 10.61        | 12.11      
400 | 10 | 0      | 0.9412    | 0.01509     | 1.149     | 44              | 9.973        | 11.46      
400 | 11 | 0      | 0.6535    | 0.05101     | 1.512     | 19              | 8.865        | 13.4       
400 | 12 | 0      | 0.4443    | 0.07602     | 1.206     | 43.1            | 11.83        | 14.28      
400 | 13 | 0      | 0.759     | 0.0745      | 1.773     | 64.85           | 8.431        | 14.95      
400 | 14 | 0      | 0.1007    | 0.1182      | 1.332     | 88.39           | 12.13        | 16.16      
400 | 15 | 0      | 0.6552    | 0.09168     | 0.5704    | 63              | 20.62        | 11.76      
400 | 16 | 0      | 0.8466    | 0.05769     | 0.4602    | 0               | 26           | 11.97      
400 | 17 | 0      | 0.7083    | 0.08594     | 1.177     | 62              | 9.033        | 10.63      
```

## Mean by MHz (primary)

```
mhz | n  | pearson_r_mean | pattern_mae_mean | max_ratio_mean | peak_loc_err_px_mean | real_max_ohm_mean | gen_max_ohm_mean
----|----|----------------|------------------|----------------|----------------------|-------------------|-----------------
10  | 11 | 0.9226         | 0.06945          | 0.6422         | 17.45                | 0.3475            | 0.2233          
70  | 11 | 0.9545         | 0.02512          | 0.8346         | 11.64                | 2.65              | 2.214           
120 | 11 | 0.8864         | 0.05092          | 0.8706         | 33.51                | 5.54              | 4.782           
270 | 11 | 0.3892         | 0.1379           | 1.23           | 60.18                | 26.59             | 20.04           
400 | 11 | 0.6922         | 0.0633           | 1.128          | 55.67                | 12.98             | 12.83           
```

## Agent copy block

```
sim_compare_metrics: n=55
  10.0MHz K7 s0: r=0.904 pattern_mae=0.078 max_ratio=0.600 real_max=0.349 gen_max=0.209 peak_err=0.0px
  10.0MHz K8 s0: r=0.965 pattern_mae=0.043 max_ratio=0.645 real_max=0.352 gen_max=0.227 peak_err=0.0px
  10.0MHz K9 s0: r=0.942 pattern_mae=0.075 max_ratio=0.618 real_max=0.334 gen_max=0.207 peak_err=63.0px
  10.0MHz K10 s0: r=0.810 pattern_mae=0.099 max_ratio=0.700 real_max=0.346 gen_max=0.242 peak_err=1.0px
  10.0MHz K11 s0: r=0.969 pattern_mae=0.065 max_ratio=0.640 real_max=0.352 gen_max=0.225 peak_err=0.0px
  10.0MHz K12 s0: r=0.930 pattern_mae=0.050 max_ratio=0.620 real_max=0.347 gen_max=0.215 peak_err=63.0px
  10.0MHz K13 s0: r=0.894 pattern_mae=0.108 max_ratio=0.688 real_max=0.362 gen_max=0.249 peak_err=1.0px
  10.0MHz K14 s0: r=0.905 pattern_mae=0.107 max_ratio=0.728 real_max=0.344 gen_max=0.251 peak_err=1.0px
  10.0MHz K15 s0: r=0.929 pattern_mae=0.042 max_ratio=0.602 real_max=0.352 gen_max=0.212 peak_err=63.0px
  10.0MHz K16 s0: r=0.944 pattern_mae=0.044 max_ratio=0.602 real_max=0.336 gen_max=0.202 peak_err=0.0px
  10.0MHz K17 s0: r=0.956 pattern_mae=0.052 max_ratio=0.622 real_max=0.349 gen_max=0.217 peak_err=0.0px
  70.0MHz K7 s0: r=0.941 pattern_mae=0.042 max_ratio=0.758 real_max=2.85 gen_max=2.16 peak_err=63.0px
  70.0MHz K8 s0: r=0.985 pattern_mae=0.003 max_ratio=0.798 real_max=2.72 gen_max=2.17 peak_err=0.0px
  70.0MHz K9 s0: r=0.972 pattern_mae=0.025 max_ratio=0.751 real_max=2.62 gen_max=1.97 peak_err=0.0px
  70.0MHz K10 s0: r=0.790 pattern_mae=0.085 max_ratio=0.826 real_max=2.57 gen_max=2.12 peak_err=2.0px
  70.0MHz K11 s0: r=0.984 pattern_mae=0.011 max_ratio=0.815 real_max=2.64 gen_max=2.15 peak_err=0.0px
  70.0MHz K12 s0: r=0.992 pattern_mae=0.008 max_ratio=0.834 real_max=2.64 gen_max=2.2 peak_err=0.0px
  70.0MHz K13 s0: r=0.956 pattern_mae=0.043 max_ratio=1.032 real_max=2.8 gen_max=2.89 peak_err=0.0px
  70.0MHz K14 s0: r=0.977 pattern_mae=0.012 max_ratio=0.891 real_max=2.63 gen_max=2.34 peak_err=0.0px
  70.0MHz K15 s0: r=0.937 pattern_mae=0.027 max_ratio=0.805 real_max=2.61 gen_max=2.1 peak_err=63.0px
  70.0MHz K16 s0: r=0.985 pattern_mae=0.008 max_ratio=0.817 real_max=2.47 gen_max=2.02 peak_err=0.0px
  70.0MHz K17 s0: r=0.980 pattern_mae=0.013 max_ratio=0.854 real_max=2.62 gen_max=2.23 peak_err=0.0px
  120.0MHz K7 s0: r=0.710 pattern_mae=0.101 max_ratio=0.595 real_max=6.68 gen_max=3.97 peak_err=63.0px
  120.0MHz K8 s0: r=0.838 pattern_mae=0.062 max_ratio=0.679 real_max=5.92 gen_max=4.02 peak_err=63.0px
  120.0MHz K9 s0: r=0.667 pattern_mae=0.125 max_ratio=0.717 real_max=5.72 gen_max=4.1 peak_err=63.0px
  120.0MHz K10 s0: r=0.783 pattern_mae=0.118 max_ratio=0.704 real_max=5.04 gen_max=3.55 peak_err=8.6px
  120.0MHz K11 s0: r=0.968 pattern_mae=0.027 max_ratio=0.780 real_max=5.38 gen_max=4.2 peak_err=44.0px
  120.0MHz K12 s0: r=0.974 pattern_mae=0.021 max_ratio=0.817 real_max=5.38 gen_max=4.4 peak_err=63.0px
  120.0MHz K13 s0: r=0.982 pattern_mae=0.028 max_ratio=1.046 real_max=6.54 gen_max=6.84 peak_err=0.0px
  120.0MHz K14 s0: r=0.947 pattern_mae=0.025 max_ratio=1.415 real_max=5.25 gen_max=7.42 peak_err=1.0px
  120.0MHz K15 s0: r=0.926 pattern_mae=0.034 max_ratio=0.894 real_max=5.14 gen_max=4.6 peak_err=63.0px
  120.0MHz K16 s0: r=0.985 pattern_mae=0.010 max_ratio=1.016 real_max=4.62 gen_max=4.69 peak_err=0.0px
  120.0MHz K17 s0: r=0.971 pattern_mae=0.009 max_ratio=0.913 real_max=5.27 gen_max=4.81 peak_err=0.0px
  270.0MHz K7 s0: r=0.391 pattern_mae=0.133 max_ratio=1.214 real_max=12.2 gen_max=14.8 peak_err=88.4px
  270.0MHz K8 s0: r=0.786 pattern_mae=0.089 max_ratio=0.280 real_max=50.8 gen_max=14.2 peak_err=4.1px
  270.0MHz K9 s0: r=0.371 pattern_mae=0.147 max_ratio=1.582 real_max=10.8 gen_max=17.1 peak_err=56.1px
  270.0MHz K10 s0: r=0.354 pattern_mae=0.159 max_ratio=0.264 real_max=44.5 gen_max=11.7 peak_err=64.0px
  270.0MHz K11 s0: r=0.721 pattern_mae=0.107 max_ratio=1.006 real_max=14.1 gen_max=14.2 peak_err=60.2px
  270.0MHz K12 s0: r=0.258 pattern_mae=0.167 max_ratio=1.057 real_max=17 gen_max=18 peak_err=84.2px
  270.0MHz K13 s0: r=0.152 pattern_mae=0.163 max_ratio=0.389 real_max=74.2 gen_max=28.9 peak_err=55.0px
  270.0MHz K14 s0: r=0.217 pattern_mae=0.164 max_ratio=4.121 real_max=9.37 gen_max=38.6 peak_err=62.0px
  270.0MHz K15 s0: r=0.470 pattern_mae=0.108 max_ratio=0.755 real_max=32.7 gen_max=24.7 peak_err=62.0px
  270.0MHz K16 s0: r=0.224 pattern_mae=0.153 max_ratio=1.310 real_max=13.6 gen_max=17.7 peak_err=63.0px
  270.0MHz K17 s0: r=0.339 pattern_mae=0.125 max_ratio=1.549 real_max=13.3 gen_max=20.6 peak_err=63.0px
  400.0MHz K7 s0: r=0.846 pattern_mae=0.038 max_ratio=0.803 real_max=16.8 gen_max=13.5 peak_err=76.0px
  400.0MHz K8 s0: r=0.888 pattern_mae=0.036 max_ratio=1.282 real_max=8.49 gen_max=10.9 peak_err=76.0px
  400.0MHz K9 s0: r=0.772 pattern_mae=0.052 max_ratio=1.142 real_max=10.6 gen_max=12.1 peak_err=76.0px
  400.0MHz K10 s0: r=0.941 pattern_mae=0.015 max_ratio=1.149 real_max=9.97 gen_max=11.5 peak_err=44.0px
  400.0MHz K11 s0: r=0.654 pattern_mae=0.051 max_ratio=1.512 real_max=8.86 gen_max=13.4 peak_err=19.0px
  400.0MHz K12 s0: r=0.444 pattern_mae=0.076 max_ratio=1.206 real_max=11.8 gen_max=14.3 peak_err=43.1px
  400.0MHz K13 s0: r=0.759 pattern_mae=0.075 max_ratio=1.773 real_max=8.43 gen_max=14.9 peak_err=64.8px
  400.0MHz K14 s0: r=0.101 pattern_mae=0.118 max_ratio=1.332 real_max=12.1 gen_max=16.2 peak_err=88.4px
  400.0MHz K15 s0: r=0.655 pattern_mae=0.092 max_ratio=0.570 real_max=20.6 gen_max=11.8 peak_err=63.0px
  400.0MHz K16 s0: r=0.847 pattern_mae=0.058 max_ratio=0.460 real_max=26 gen_max=12 peak_err=0.0px
  400.0MHz K17 s0: r=0.708 pattern_mae=0.086 max_ratio=1.177 real_max=9.03 gen_max=10.6 peak_err=62.0px
```
