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
10  | 3  | 0      | 0.9489    | 0.1039       | 0.02222     | 0.7374    | 63              | 0.3955       | 0.2916     
10  | 4  | 0      | 0.9883    | 0.04782      | 0.02917     | 0.8569    | 63              | 0.3343       | 0.2864     
10  | 5  | 0      | 0.9848    | 0.05202      | 0.02723     | 0.8521    | 63              | 0.3516       | 0.2996     
10  | 6  | 0      | 0.9533    | 0.0369       | 0.02968     | 0.8869    | 63              | 0.3263       | 0.2894     
10  | 7  | 0      | 0.9186    | 0.06257      | 0.02249     | 0.8209    | 0               | 0.3493       | 0.2867     
10  | 8  | 0      | 0.9882    | 0.04741      | 0.006209    | 0.863     | 0               | 0.3462       | 0.2988     
10  | 9  | 0      | 0.9649    | 0.0512       | 0.02095     | 0.8466    | 44              | 0.3337       | 0.2825     
10  | 10 | 0      | 0.97      | 0.05471      | 0.01473     | 0.8387    | 0               | 0.3392       | 0.2845     
80  | 3  | 0      | 0.9305    | 10.82        | 0.05579     | 0.3196    | 19              | 15.91        | 5.084      
80  | 4  | 0      | 0.9377    | 0.4273       | 0.0519      | 0.883     | 0               | 3.652        | 3.225      
80  | 5  | 0      | 0.9759    | 0.101        | 0.0583      | 0.9713    | 0               | 3.52         | 3.419      
80  | 6  | 0      | 0.9246    | 0.2537       | 0.02444     | 0.9175    | 63              | 3.075        | 2.821      
80  | 7  | 0      | 0.9478    | 0.504        | 0.01669     | 0.8516    | 0               | 3.396        | 2.892      
80  | 8  | 0      | 0.9805    | 0.215        | 0.005201    | 0.9299    | 0               | 3.066        | 2.851      
80  | 9  | 0      | 0.9801    | 0.3468       | 0.01496     | 0.8771    | 63              | 2.821        | 2.474      
80  | 10 | 0      | 0.9243    | 0.2361       | 0.01992     | 0.9185    | 0               | 2.895        | 2.659      
150 | 3  | 0      | 0.9914    | 1.389        | 0.02242     | 0.7574    | 0               | 5.726        | 4.337      
150 | 4  | 0      | 0.9624    | 2.22         | 0.04585     | 0.6011    | 0               | 5.565        | 3.345      
150 | 5  | 0      | 0.9237    | 0.8448       | 0.05245     | 0.7856    | 0               | 3.941        | 3.096      
150 | 6  | 0      | 0.4521    | 6.91         | 0.1674      | 0.6353    | 44              | 18.95        | 12.04      
150 | 7  | 0      | 0.7067    | -6.81        | 0.1566      | 2.794     | 63              | 3.795        | 10.6       
150 | 8  | 0      | 0.9474    | -0.1288      | 0.04288     | 1.013     | 0               | 10.21        | 10.34      
150 | 9  | 0      | 0.9352    | 0.4716       | 0.03679     | 0.9322    | 61              | 6.955        | 6.483      
150 | 10 | 0      | 0.9325    | 1.091        | 0.03602     | 0.859     | 63              | 7.735        | 6.644      
270 | 3  | 0      | 0.945     | -0.4731      | 0.03248     | 1.046     | 58.42           | 10.26        | 10.74      
270 | 4  | 0      | 0.9592    | 0.0131       | 0.05116     | 0.999     | 59.3            | 12.85        | 12.84      
270 | 5  | 0      | 0.9434    | 1.236        | 0.02297     | 0.8597    | 62              | 8.809        | 7.573      
270 | 6  | 0      | 0.8772    | -1.112       | 0.06086     | 1.109     | 88.39           | 10.18        | 11.29      
270 | 7  | 0      | 0.1957    | -0.4739      | 0.1557      | 1.054     | 76.03           | 8.769        | 9.242      
270 | 8  | 0      | 0.8419    | 8.233        | 0.08427     | 0.5471    | 62.59           | 18.18        | 9.946      
270 | 9  | 0      | 0.94      | -0.01004     | 0.04853     | 1.001     | 0               | 13.31        | 13.32      
270 | 10 | 0      | 0.5103    | 2.416        | 0.1072      | 0.8233    | 76.03           | 13.67        | 11.26      
450 | 3  | 0      | 0.8386    | 2.774        | 0.07458     | 0.7308    | 0               | 10.3         | 7.53       
450 | 4  | 0      | 0.9538    | 3.26         | 0.04875     | 0.7074    | 63              | 11.14        | 7.88       
450 | 5  | 0      | 0.8633    | 0.1706       | 0.0522      | 0.9772    | 0               | 7.479        | 7.309      
450 | 6  | 0      | 0.9594    | 2.431        | 0.04357     | 0.7574    | 63              | 10.02        | 7.591      
450 | 7  | 0      | 0.4209    | 10.18        | 0.1526      | 0.3979    | 88.39           | 16.91        | 6.727      
450 | 8  | 0      | 0.8415    | 0.008804     | 0.06884     | 0.9989    | 21.63           | 8.181        | 8.173      
450 | 9  | 0      | 0.7552    | 13.13        | 0.07975     | 0.3548    | 76.03           | 20.34        | 7.217      
450 | 10 | 0      | 0.5191    | 13.83        | 0.1068      | 0.3337    | 19              | 20.75        | 6.925      
550 | 3  | 0      | 0.5293    | 5.964        | 0.1178      | 0.6947    | 88.39           | 19.54        | 13.57      
550 | 4  | 0      | 0.2534    | 1.844        | 0.1372      | 0.8352    | 0               | 11.19        | 9.342      
550 | 5  | 0      | 0.6598    | 10.62        | 0.1189      | 0.5162    | 76.03           | 21.95        | 11.33      
550 | 6  | 0      | 0.631     | -0.2482      | 0.09336     | 1.024     | 44              | 10.14        | 10.39      
550 | 7  | 0      | 0.5093    | 4.379        | 0.1047      | 0.6926    | 88.39           | 14.24        | 9.866      
550 | 8  | 0      | 0.6245    | 14.91        | 0.1101      | 0.4493    | 33.62           | 27.07        | 12.17      
550 | 9  | 0      | 0.5846    | 5.412        | 0.1065      | 0.7031    | 34.54           | 18.23        | 12.82      
550 | 10 | 0      | 0.4884    | 8.435        | 0.1156      | 0.5688    | 88.39           | 19.56        | 11.12      
```

## Mean by MHz (primary)

```
mhz | n | pearson_r_mean | max_diff_ohm_mean | pattern_mae_mean | max_ratio_mean | peak_loc_err_px_mean | real_max_ohm_mean | gen_max_ohm_mean
----|---|----------------|-------------------|------------------|----------------|----------------------|-------------------|-----------------
10  | 8 | 0.9646         | 0.05706           | 0.02158          | 0.8378         | 37                   | 0.347             | 0.2899          
80  | 8 | 0.9502         | 1.613             | 0.0309           | 0.8335         | 18.12                | 4.792             | 3.178           
150 | 8 | 0.8564         | 0.7485            | 0.07005          | 1.047          | 28.88                | 7.86              | 7.111           
270 | 8 | 0.7766         | 1.229             | 0.07039          | 0.9299         | 60.34                | 12                | 10.78           
450 | 8 | 0.769          | 5.722             | 0.07839          | 0.6573         | 41.38                | 13.14             | 7.419           
550 | 8 | 0.535          | 6.414             | 0.113            | 0.6855         | 56.67                | 17.74             | 11.33           
```

## Agent copy block

```
sim_compare_metrics: n=48
  10.0MHz K3 s0: r=0.949 Δmax=+0.104Ω pattern_mae=0.022 max_ratio=0.737 real_max=0.395 gen_max=0.292 peak_err=63.0px
  10.0MHz K4 s0: r=0.988 Δmax=+0.0478Ω pattern_mae=0.029 max_ratio=0.857 real_max=0.334 gen_max=0.286 peak_err=63.0px
  10.0MHz K5 s0: r=0.985 Δmax=+0.052Ω pattern_mae=0.027 max_ratio=0.852 real_max=0.352 gen_max=0.3 peak_err=63.0px
  10.0MHz K6 s0: r=0.953 Δmax=+0.0369Ω pattern_mae=0.030 max_ratio=0.887 real_max=0.326 gen_max=0.289 peak_err=63.0px
  10.0MHz K7 s0: r=0.919 Δmax=+0.0626Ω pattern_mae=0.022 max_ratio=0.821 real_max=0.349 gen_max=0.287 peak_err=0.0px
  10.0MHz K8 s0: r=0.988 Δmax=+0.0474Ω pattern_mae=0.006 max_ratio=0.863 real_max=0.346 gen_max=0.299 peak_err=0.0px
  10.0MHz K9 s0: r=0.965 Δmax=+0.0512Ω pattern_mae=0.021 max_ratio=0.847 real_max=0.334 gen_max=0.283 peak_err=44.0px
  10.0MHz K10 s0: r=0.970 Δmax=+0.0547Ω pattern_mae=0.015 max_ratio=0.839 real_max=0.339 gen_max=0.285 peak_err=0.0px
  80.0MHz K3 s0: r=0.930 Δmax=+10.8Ω pattern_mae=0.056 max_ratio=0.320 real_max=15.9 gen_max=5.08 peak_err=19.0px
  80.0MHz K4 s0: r=0.938 Δmax=+0.427Ω pattern_mae=0.052 max_ratio=0.883 real_max=3.65 gen_max=3.22 peak_err=0.0px
  80.0MHz K5 s0: r=0.976 Δmax=+0.101Ω pattern_mae=0.058 max_ratio=0.971 real_max=3.52 gen_max=3.42 peak_err=0.0px
  80.0MHz K6 s0: r=0.925 Δmax=+0.254Ω pattern_mae=0.024 max_ratio=0.917 real_max=3.07 gen_max=2.82 peak_err=63.0px
  80.0MHz K7 s0: r=0.948 Δmax=+0.504Ω pattern_mae=0.017 max_ratio=0.852 real_max=3.4 gen_max=2.89 peak_err=0.0px
  80.0MHz K8 s0: r=0.981 Δmax=+0.215Ω pattern_mae=0.005 max_ratio=0.930 real_max=3.07 gen_max=2.85 peak_err=0.0px
  80.0MHz K9 s0: r=0.980 Δmax=+0.347Ω pattern_mae=0.015 max_ratio=0.877 real_max=2.82 gen_max=2.47 peak_err=63.0px
  80.0MHz K10 s0: r=0.924 Δmax=+0.236Ω pattern_mae=0.020 max_ratio=0.918 real_max=2.89 gen_max=2.66 peak_err=0.0px
  150.0MHz K3 s0: r=0.991 Δmax=+1.39Ω pattern_mae=0.022 max_ratio=0.757 real_max=5.73 gen_max=4.34 peak_err=0.0px
  150.0MHz K4 s0: r=0.962 Δmax=+2.22Ω pattern_mae=0.046 max_ratio=0.601 real_max=5.56 gen_max=3.35 peak_err=0.0px
  150.0MHz K5 s0: r=0.924 Δmax=+0.845Ω pattern_mae=0.052 max_ratio=0.786 real_max=3.94 gen_max=3.1 peak_err=0.0px
  150.0MHz K6 s0: r=0.452 Δmax=+6.91Ω pattern_mae=0.167 max_ratio=0.635 real_max=18.9 gen_max=12 peak_err=44.0px
  150.0MHz K7 s0: r=0.707 Δmax=-6.81Ω pattern_mae=0.157 max_ratio=2.794 real_max=3.79 gen_max=10.6 peak_err=63.0px
  150.0MHz K8 s0: r=0.947 Δmax=-0.129Ω pattern_mae=0.043 max_ratio=1.013 real_max=10.2 gen_max=10.3 peak_err=0.0px
  150.0MHz K9 s0: r=0.935 Δmax=+0.472Ω pattern_mae=0.037 max_ratio=0.932 real_max=6.95 gen_max=6.48 peak_err=61.0px
  150.0MHz K10 s0: r=0.932 Δmax=+1.09Ω pattern_mae=0.036 max_ratio=0.859 real_max=7.74 gen_max=6.64 peak_err=63.0px
  270.0MHz K3 s0: r=0.945 Δmax=-0.473Ω pattern_mae=0.032 max_ratio=1.046 real_max=10.3 gen_max=10.7 peak_err=58.4px
  270.0MHz K4 s0: r=0.959 Δmax=+0.0131Ω pattern_mae=0.051 max_ratio=0.999 real_max=12.9 gen_max=12.8 peak_err=59.3px
  270.0MHz K5 s0: r=0.943 Δmax=+1.24Ω pattern_mae=0.023 max_ratio=0.860 real_max=8.81 gen_max=7.57 peak_err=62.0px
  270.0MHz K6 s0: r=0.877 Δmax=-1.11Ω pattern_mae=0.061 max_ratio=1.109 real_max=10.2 gen_max=11.3 peak_err=88.4px
  270.0MHz K7 s0: r=0.196 Δmax=-0.474Ω pattern_mae=0.156 max_ratio=1.054 real_max=8.77 gen_max=9.24 peak_err=76.0px
  270.0MHz K8 s0: r=0.842 Δmax=+8.23Ω pattern_mae=0.084 max_ratio=0.547 real_max=18.2 gen_max=9.95 peak_err=62.6px
  270.0MHz K9 s0: r=0.940 Δmax=-0.01Ω pattern_mae=0.049 max_ratio=1.001 real_max=13.3 gen_max=13.3 peak_err=0.0px
  270.0MHz K10 s0: r=0.510 Δmax=+2.42Ω pattern_mae=0.107 max_ratio=0.823 real_max=13.7 gen_max=11.3 peak_err=76.0px
  450.0MHz K3 s0: r=0.839 Δmax=+2.77Ω pattern_mae=0.075 max_ratio=0.731 real_max=10.3 gen_max=7.53 peak_err=0.0px
  450.0MHz K4 s0: r=0.954 Δmax=+3.26Ω pattern_mae=0.049 max_ratio=0.707 real_max=11.1 gen_max=7.88 peak_err=63.0px
  450.0MHz K5 s0: r=0.863 Δmax=+0.171Ω pattern_mae=0.052 max_ratio=0.977 real_max=7.48 gen_max=7.31 peak_err=0.0px
  450.0MHz K6 s0: r=0.959 Δmax=+2.43Ω pattern_mae=0.044 max_ratio=0.757 real_max=10 gen_max=7.59 peak_err=63.0px
  450.0MHz K7 s0: r=0.421 Δmax=+10.2Ω pattern_mae=0.153 max_ratio=0.398 real_max=16.9 gen_max=6.73 peak_err=88.4px
  450.0MHz K8 s0: r=0.841 Δmax=+0.0088Ω pattern_mae=0.069 max_ratio=0.999 real_max=8.18 gen_max=8.17 peak_err=21.6px
  450.0MHz K9 s0: r=0.755 Δmax=+13.1Ω pattern_mae=0.080 max_ratio=0.355 real_max=20.3 gen_max=7.22 peak_err=76.0px
  450.0MHz K10 s0: r=0.519 Δmax=+13.8Ω pattern_mae=0.107 max_ratio=0.334 real_max=20.8 gen_max=6.92 peak_err=19.0px
  550.0MHz K3 s0: r=0.529 Δmax=+5.96Ω pattern_mae=0.118 max_ratio=0.695 real_max=19.5 gen_max=13.6 peak_err=88.4px
  550.0MHz K4 s0: r=0.253 Δmax=+1.84Ω pattern_mae=0.137 max_ratio=0.835 real_max=11.2 gen_max=9.34 peak_err=0.0px
  550.0MHz K5 s0: r=0.660 Δmax=+10.6Ω pattern_mae=0.119 max_ratio=0.516 real_max=21.9 gen_max=11.3 peak_err=76.0px
  550.0MHz K6 s0: r=0.631 Δmax=-0.248Ω pattern_mae=0.093 max_ratio=1.024 real_max=10.1 gen_max=10.4 peak_err=44.0px
  550.0MHz K7 s0: r=0.509 Δmax=+4.38Ω pattern_mae=0.105 max_ratio=0.693 real_max=14.2 gen_max=9.87 peak_err=88.4px
  550.0MHz K8 s0: r=0.624 Δmax=+14.9Ω pattern_mae=0.110 max_ratio=0.449 real_max=27.1 gen_max=12.2 peak_err=33.6px
  550.0MHz K9 s0: r=0.585 Δmax=+5.41Ω pattern_mae=0.107 max_ratio=0.703 real_max=18.2 gen_max=12.8 peak_err=34.5px
  550.0MHz K10 s0: r=0.488 Δmax=+8.43Ω pattern_mae=0.116 max_ratio=0.569 real_max=19.6 gen_max=11.1 peak_err=88.4px
```
