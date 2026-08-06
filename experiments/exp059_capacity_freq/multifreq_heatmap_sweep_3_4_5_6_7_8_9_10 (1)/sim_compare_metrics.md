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
10  | 3  | 0      | 0.9311    | 0.09096      | 0.06587     | 0.7851    | 63              | 0.4232       | 0.3323     
10  | 4  | 0      | 0.9575    | 0.04579      | 0.02109     | 0.8697    | 0               | 0.3512       | 0.3055     
10  | 5  | 0      | 0.976     | 0.01393      | 0.0116      | 0.9605    | 0               | 0.353        | 0.339      
10  | 6  | 0      | 0.9335    | 0.02678      | 0.01682     | 0.9212    | 0               | 0.3399       | 0.3131     
10  | 7  | 0      | 0.9547    | 0.04251      | 0.009977    | 0.8785    | 0               | 0.3499       | 0.3074     
10  | 8  | 0      | 0.9636    | 0.03236      | 0.008439    | 0.9067    | 0               | 0.3469       | 0.3146     
10  | 9  | 0      | 0.9675    | 0.03008      | 0.009367    | 0.9095    | 0               | 0.3324       | 0.3024     
10  | 10 | 0      | 0.9057    | 0.01604      | 0.02584     | 0.9487    | 0               | 0.3125       | 0.2965     
80  | 3  | 0      | 0.9052    | 12.51        | 0.1008      | 0.2133    | 0               | 15.91        | 3.392      
80  | 4  | 0      | 0.854     | 1.071        | 0.05839     | 0.7279    | 63              | 3.934        | 2.864      
80  | 5  | 0      | 0.9726    | 0.07767      | 0.02021     | 0.9776    | 0               | 3.473        | 3.395      
80  | 6  | 0      | 0.883     | 0.188        | 0.03182     | 0.9373    | 0               | 2.998        | 2.81       
80  | 7  | 0      | 0.9551    | 0.4259       | 0.01025     | 0.8664    | 0               | 3.187        | 2.761      
80  | 8  | 0      | 0.9453    | 0.3997       | 0.01718     | 0.8716    | 0               | 3.114        | 2.714      
80  | 9  | 0      | 0.9132    | 0.2992       | 0.02359     | 0.8932    | 63              | 2.801        | 2.502      
80  | 10 | 0      | 0.9362    | 0.3513       | 0.0182      | 0.8842    | 0               | 3.033        | 2.682      
150 | 3  | 0      | 0.9678    | 0.8903       | 0.0125      | 0.8473    | 0               | 5.831        | 4.941      
150 | 4  | 0      | 0.8583    | 0.6662       | 0.04654     | 0.8608    | 0               | 4.785        | 4.119      
150 | 5  | 0      | -0.2807   | -0.2551      | 0.2676      | 1.065     | 54.45           | 3.953        | 4.209      
150 | 6  | 0      | 0.7291    | 7.364        | 0.2073      | 0.457     | 19              | 13.56        | 6.197      
150 | 7  | 0      | 0.9668    | 13.83        | 0.09761     | 0.3364    | 2               | 20.84        | 7.01       
150 | 8  | 0      | 0.9259    | 7.783        | 0.04499     | 0.4901    | 0               | 15.26        | 7.481      
150 | 9  | 0      | 0.8801    | 0.9785       | 0.04003     | 0.8442    | 0               | 6.28         | 5.301      
150 | 10 | 0      | 0.8741    | 2.856        | 0.04349     | 0.6756    | 0               | 8.805        | 5.949      
270 | 3  | 0      | 0.9118    | -7.884       | 0.06911     | 1.939     | 58.01           | 8.394        | 16.28      
270 | 4  | 0      | 0.8941    | -14.98       | 0.07184     | 2.578     | 2.236           | 9.491        | 24.47      
270 | 5  | 0      | 0.9776    | -4.468       | 0.02833     | 1.508     | 0               | 8.793        | 13.26      
270 | 6  | 0      | 0.7311    | 9.023        | 0.1136      | 0.7082    | 5               | 30.93        | 21.9       
270 | 7  | 0      | 0.1703    | 11.08        | 0.1667      | 0.4869    | 61.13           | 21.59        | 10.51      
270 | 8  | 0      | 0.9622    | 18.4         | 0.0293      | 0.5062    | 0               | 37.26        | 18.86      
270 | 9  | 0      | 0.8925    | -8.897       | 0.05566     | 1.422     | 0               | 21.07        | 29.96      
270 | 10 | 0      | 0.4093    | 12.87        | 0.1143      | 0.6535    | 63              | 37.14        | 24.27      
450 | 3  | 0      | 0.9539    | 0.8429       | 0.01251     | 0.9223    | 0               | 10.85        | 10         
450 | 4  | 0      | 0.9508    | -0.1857      | 0.01912     | 1.019     | 63              | 9.845        | 10.03      
450 | 5  | 0      | 0.8398    | 0.6739       | 0.04971     | 0.9168    | 44              | 8.095        | 7.421      
450 | 6  | 0      | 0.9639    | 0.3781       | 0.01569     | 0.9575    | 63              | 8.906        | 8.527      
450 | 7  | 0      | 0.5191    | 15.5         | 0.1588      | 0.2929    | 34              | 21.92        | 6.421      
450 | 8  | 0      | 0.765     | 0.5876       | 0.06399     | 0.9214    | 30              | 7.476        | 6.889      
450 | 9  | 0      | 0.6823    | 1.615        | 0.1179      | 0.8034    | 41.05           | 8.216        | 6.6        
450 | 10 | 0      | 0.4516    | -8.106       | 0.1625      | 2.385     | 38              | 5.851        | 13.96      
550 | 3  | 0      | 0.5573    | 4.91         | 0.1055      | 0.7487    | 88.39           | 19.54        | 14.63      
550 | 4  | 0      | 0.4764    | 1.956        | 0.1134      | 0.8367    | 34.93           | 11.97        | 10.02      
550 | 5  | 0      | 0.7625    | 6.309        | 0.07773     | 0.6768    | 31.78           | 19.52        | 13.21      
550 | 6  | 0      | 0.6137    | 8.125        | 0.09828     | 0.5439    | 33.62           | 17.81        | 9.69       
550 | 7  | 0      | 0.3589    | 0.9498       | 0.1244      | 0.9345    | 71.59           | 14.49        | 13.54      
550 | 8  | 0      | 0.716     | 2.219        | 0.08405     | 0.8633    | 33.62           | 16.23        | 14.01      
550 | 9  | 0      | 0.6019    | 2.116        | 0.1126      | 0.8478    | 33.62           | 13.9         | 11.78      
550 | 10 | 0      | 0.6039    | 0.5109       | 0.103       | 0.9645    | 33.62           | 14.38        | 13.87      
```

## Mean by MHz (primary)

```
mhz | n | pearson_r_mean | max_diff_ohm_mean | pattern_mae_mean | max_ratio_mean | peak_loc_err_px_mean | real_max_ohm_mean | gen_max_ohm_mean
----|---|----------------|-------------------|------------------|----------------|----------------------|-------------------|-----------------
10  | 8 | 0.9487         | 0.03731           | 0.02113          | 0.8975         | 7.875                | 0.3511            | 0.3138          
80  | 8 | 0.9206         | 1.916             | 0.03505          | 0.7964         | 15.75                | 4.806             | 2.89            
150 | 8 | 0.7402         | 4.264             | 0.095            | 0.697          | 9.431                | 9.915             | 5.651           
270 | 8 | 0.7436         | 1.893             | 0.0811           | 1.225          | 23.67                | 21.83             | 19.94           
450 | 8 | 0.7658         | 1.413             | 0.07503          | 1.027          | 39.13                | 10.14             | 8.731           
550 | 8 | 0.5863         | 3.387             | 0.1024           | 0.802          | 45.14                | 15.98             | 12.59           
```

## Agent copy block

```
sim_compare_metrics: n=48
  10.0MHz K3 s0: r=0.931 Δmax=+0.091Ω pattern_mae=0.066 max_ratio=0.785 real_max=0.423 gen_max=0.332 peak_err=63.0px
  10.0MHz K4 s0: r=0.957 Δmax=+0.0458Ω pattern_mae=0.021 max_ratio=0.870 real_max=0.351 gen_max=0.305 peak_err=0.0px
  10.0MHz K5 s0: r=0.976 Δmax=+0.0139Ω pattern_mae=0.012 max_ratio=0.961 real_max=0.353 gen_max=0.339 peak_err=0.0px
  10.0MHz K6 s0: r=0.934 Δmax=+0.0268Ω pattern_mae=0.017 max_ratio=0.921 real_max=0.34 gen_max=0.313 peak_err=0.0px
  10.0MHz K7 s0: r=0.955 Δmax=+0.0425Ω pattern_mae=0.010 max_ratio=0.879 real_max=0.35 gen_max=0.307 peak_err=0.0px
  10.0MHz K8 s0: r=0.964 Δmax=+0.0324Ω pattern_mae=0.008 max_ratio=0.907 real_max=0.347 gen_max=0.315 peak_err=0.0px
  10.0MHz K9 s0: r=0.967 Δmax=+0.0301Ω pattern_mae=0.009 max_ratio=0.910 real_max=0.332 gen_max=0.302 peak_err=0.0px
  10.0MHz K10 s0: r=0.906 Δmax=+0.016Ω pattern_mae=0.026 max_ratio=0.949 real_max=0.313 gen_max=0.296 peak_err=0.0px
  80.0MHz K3 s0: r=0.905 Δmax=+12.5Ω pattern_mae=0.101 max_ratio=0.213 real_max=15.9 gen_max=3.39 peak_err=0.0px
  80.0MHz K4 s0: r=0.854 Δmax=+1.07Ω pattern_mae=0.058 max_ratio=0.728 real_max=3.93 gen_max=2.86 peak_err=63.0px
  80.0MHz K5 s0: r=0.973 Δmax=+0.0777Ω pattern_mae=0.020 max_ratio=0.978 real_max=3.47 gen_max=3.4 peak_err=0.0px
  80.0MHz K6 s0: r=0.883 Δmax=+0.188Ω pattern_mae=0.032 max_ratio=0.937 real_max=3 gen_max=2.81 peak_err=0.0px
  80.0MHz K7 s0: r=0.955 Δmax=+0.426Ω pattern_mae=0.010 max_ratio=0.866 real_max=3.19 gen_max=2.76 peak_err=0.0px
  80.0MHz K8 s0: r=0.945 Δmax=+0.4Ω pattern_mae=0.017 max_ratio=0.872 real_max=3.11 gen_max=2.71 peak_err=0.0px
  80.0MHz K9 s0: r=0.913 Δmax=+0.299Ω pattern_mae=0.024 max_ratio=0.893 real_max=2.8 gen_max=2.5 peak_err=63.0px
  80.0MHz K10 s0: r=0.936 Δmax=+0.351Ω pattern_mae=0.018 max_ratio=0.884 real_max=3.03 gen_max=2.68 peak_err=0.0px
  150.0MHz K3 s0: r=0.968 Δmax=+0.89Ω pattern_mae=0.013 max_ratio=0.847 real_max=5.83 gen_max=4.94 peak_err=0.0px
  150.0MHz K4 s0: r=0.858 Δmax=+0.666Ω pattern_mae=0.047 max_ratio=0.861 real_max=4.79 gen_max=4.12 peak_err=0.0px
  150.0MHz K5 s0: r=-0.281 Δmax=-0.255Ω pattern_mae=0.268 max_ratio=1.065 real_max=3.95 gen_max=4.21 peak_err=54.5px
  150.0MHz K6 s0: r=0.729 Δmax=+7.36Ω pattern_mae=0.207 max_ratio=0.457 real_max=13.6 gen_max=6.2 peak_err=19.0px
  150.0MHz K7 s0: r=0.967 Δmax=+13.8Ω pattern_mae=0.098 max_ratio=0.336 real_max=20.8 gen_max=7.01 peak_err=2.0px
  150.0MHz K8 s0: r=0.926 Δmax=+7.78Ω pattern_mae=0.045 max_ratio=0.490 real_max=15.3 gen_max=7.48 peak_err=0.0px
  150.0MHz K9 s0: r=0.880 Δmax=+0.979Ω pattern_mae=0.040 max_ratio=0.844 real_max=6.28 gen_max=5.3 peak_err=0.0px
  150.0MHz K10 s0: r=0.874 Δmax=+2.86Ω pattern_mae=0.043 max_ratio=0.676 real_max=8.8 gen_max=5.95 peak_err=0.0px
  270.0MHz K3 s0: r=0.912 Δmax=-7.88Ω pattern_mae=0.069 max_ratio=1.939 real_max=8.39 gen_max=16.3 peak_err=58.0px
  270.0MHz K4 s0: r=0.894 Δmax=-15Ω pattern_mae=0.072 max_ratio=2.578 real_max=9.49 gen_max=24.5 peak_err=2.2px
  270.0MHz K5 s0: r=0.978 Δmax=-4.47Ω pattern_mae=0.028 max_ratio=1.508 real_max=8.79 gen_max=13.3 peak_err=0.0px
  270.0MHz K6 s0: r=0.731 Δmax=+9.02Ω pattern_mae=0.114 max_ratio=0.708 real_max=30.9 gen_max=21.9 peak_err=5.0px
  270.0MHz K7 s0: r=0.170 Δmax=+11.1Ω pattern_mae=0.167 max_ratio=0.487 real_max=21.6 gen_max=10.5 peak_err=61.1px
  270.0MHz K8 s0: r=0.962 Δmax=+18.4Ω pattern_mae=0.029 max_ratio=0.506 real_max=37.3 gen_max=18.9 peak_err=0.0px
  270.0MHz K9 s0: r=0.892 Δmax=-8.9Ω pattern_mae=0.056 max_ratio=1.422 real_max=21.1 gen_max=30 peak_err=0.0px
  270.0MHz K10 s0: r=0.409 Δmax=+12.9Ω pattern_mae=0.114 max_ratio=0.654 real_max=37.1 gen_max=24.3 peak_err=63.0px
  450.0MHz K3 s0: r=0.954 Δmax=+0.843Ω pattern_mae=0.013 max_ratio=0.922 real_max=10.8 gen_max=10 peak_err=0.0px
  450.0MHz K4 s0: r=0.951 Δmax=-0.186Ω pattern_mae=0.019 max_ratio=1.019 real_max=9.85 gen_max=10 peak_err=63.0px
  450.0MHz K5 s0: r=0.840 Δmax=+0.674Ω pattern_mae=0.050 max_ratio=0.917 real_max=8.1 gen_max=7.42 peak_err=44.0px
  450.0MHz K6 s0: r=0.964 Δmax=+0.378Ω pattern_mae=0.016 max_ratio=0.958 real_max=8.91 gen_max=8.53 peak_err=63.0px
  450.0MHz K7 s0: r=0.519 Δmax=+15.5Ω pattern_mae=0.159 max_ratio=0.293 real_max=21.9 gen_max=6.42 peak_err=34.0px
  450.0MHz K8 s0: r=0.765 Δmax=+0.588Ω pattern_mae=0.064 max_ratio=0.921 real_max=7.48 gen_max=6.89 peak_err=30.0px
  450.0MHz K9 s0: r=0.682 Δmax=+1.62Ω pattern_mae=0.118 max_ratio=0.803 real_max=8.22 gen_max=6.6 peak_err=41.0px
  450.0MHz K10 s0: r=0.452 Δmax=-8.11Ω pattern_mae=0.163 max_ratio=2.385 real_max=5.85 gen_max=14 peak_err=38.0px
  550.0MHz K3 s0: r=0.557 Δmax=+4.91Ω pattern_mae=0.105 max_ratio=0.749 real_max=19.5 gen_max=14.6 peak_err=88.4px
  550.0MHz K4 s0: r=0.476 Δmax=+1.96Ω pattern_mae=0.113 max_ratio=0.837 real_max=12 gen_max=10 peak_err=34.9px
  550.0MHz K5 s0: r=0.763 Δmax=+6.31Ω pattern_mae=0.078 max_ratio=0.677 real_max=19.5 gen_max=13.2 peak_err=31.8px
  550.0MHz K6 s0: r=0.614 Δmax=+8.12Ω pattern_mae=0.098 max_ratio=0.544 real_max=17.8 gen_max=9.69 peak_err=33.6px
  550.0MHz K7 s0: r=0.359 Δmax=+0.95Ω pattern_mae=0.124 max_ratio=0.934 real_max=14.5 gen_max=13.5 peak_err=71.6px
  550.0MHz K8 s0: r=0.716 Δmax=+2.22Ω pattern_mae=0.084 max_ratio=0.863 real_max=16.2 gen_max=14 peak_err=33.6px
  550.0MHz K9 s0: r=0.602 Δmax=+2.12Ω pattern_mae=0.113 max_ratio=0.848 real_max=13.9 gen_max=11.8 peak_err=33.6px
  550.0MHz K10 s0: r=0.604 Δmax=+0.511Ω pattern_mae=0.103 max_ratio=0.964 real_max=14.4 gen_max=13.9 peak_err=33.6px
```
