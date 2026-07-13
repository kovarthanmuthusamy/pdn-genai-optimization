# Simulated Real vs Generated — Heatmap Metrics

- **Rows**: 95 (one per freq × K × sample)
- **Source**: ECADStar `.map` in `K*/Real/` vs `data_sample_*/heatmap_physical.npy`
- **CSV**: `sim_compare_metrics.csv`
- **JSON**: `sim_compare_metrics.json`

**Primary QC** (tables below): `pearson_r`, `pattern_mae`, `max_ratio`.
Use `mae_ohm` / `mape_pct` from CSV with care — often misleading at 10 MHz (FG Ω ≈ 0).

## Per-sample metrics (primary)

```
mhz | k  | sample | pearson_r | pattern_mae | max_ratio | peak_loc_err_px | real_max_ohm | gen_max_ohm
----|----|--------|-----------|-------------|-----------|-----------------|--------------|------------
10  | 2  | 0      | 0.9123    | 0.1804      | 0.7483    | 0               | 0.3286       | 0.2459     
10  | 3  | 0      | 0.9362    | 0.08258     | 0.679     | 63              | 0.3693       | 0.2508     
10  | 4  | 0      | 0.9567    | 0.1007      | 0.7011    | 0               | 0.3275       | 0.2296     
10  | 5  | 0      | 0.9676    | 0.1066      | 0.6315    | 63              | 0.3463       | 0.2187     
10  | 6  | 0      | 0.8535    | 0.07247     | 0.6041    | 0               | 0.3684       | 0.2226     
10  | 7  | 0      | 0.9671    | 0.04063     | 0.6592    | 63              | 0.3489       | 0.23       
10  | 8  | 0      | 0.9205    | 0.04536     | 0.5933    | 63              | 0.3748       | 0.2224     
10  | 9  | 0      | 0.9258    | 0.03765     | 0.6304    | 0               | 0.3453       | 0.2177     
10  | 10 | 0      | 0.9443    | 0.03162     | 0.6608    | 0               | 0.3592       | 0.2374     
10  | 11 | 0      | 0.905     | 0.05628     | 0.6586    | 0               | 0.3243       | 0.2136     
10  | 12 | 0      | 0.9383    | 0.04037     | 0.6628    | 0               | 0.3226       | 0.2138     
10  | 13 | 0      | 0.8693    | 0.0551      | 0.5195    | 63              | 0.3891       | 0.2021     
10  | 14 | 0      | 0.973     | 0.04379     | 0.601     | 63              | 0.3571       | 0.2146     
10  | 15 | 0      | 0.9165    | 0.04064     | 0.6096    | 0               | 0.3314       | 0.202      
10  | 16 | 0      | 0.9261    | 0.05813     | 0.5909    | 0               | 0.3396       | 0.2007     
10  | 17 | 0      | 0.9498    | 0.03357     | 0.6294    | 0               | 0.3331       | 0.2097     
10  | 18 | 0      | 0.9067    | 0.04107     | 0.6015    | 0               | 0.3325       | 0.2        
10  | 19 | 0      | 0.9657    | 0.03193     | 0.6963    | 0               | 0.3372       | 0.2348     
10  | 20 | 0      | 0.9021    | 0.03494     | 0.6586    | 0               | 0.271        | 0.1785     
70  | 2  | 0      | 0.9665    | 0.05146     | 0.839     | 0               | 3.473        | 2.914      
70  | 3  | 0      | 0.9079    | 0.08536     | 0.615     | 0               | 4.578        | 2.815      
70  | 4  | 0      | 0.932     | 0.06309     | 0.673     | 0               | 2.948        | 1.984      
70  | 5  | 0      | 0.9921    | 0.01658     | 0.7086    | 0               | 2.848        | 2.018      
70  | 6  | 0      | 0.9642    | 0.0256      | 0.6437    | 0               | 3.097        | 1.993      
70  | 7  | 0      | 0.9745    | 0.01575     | 0.6235    | 63              | 2.945        | 1.836      
70  | 8  | 0      | 0.8939    | 0.04813     | 0.6147    | 63              | 3.002        | 1.845      
70  | 9  | 0      | 0.9769    | 0.01412     | 0.7442    | 0               | 2.626        | 1.954      
70  | 10 | 0      | 0.9649    | 0.01587     | 0.7773    | 0               | 2.662        | 2.07       
70  | 11 | 0      | 0.9429    | 0.02042     | 0.8017    | 63              | 2.422        | 1.942      
70  | 12 | 0      | 0.9733    | 0.01438     | 0.7976    | 0               | 2.408        | 1.921      
70  | 13 | 0      | 0.8763    | 0.05257     | 0.5405    | 0               | 3.114        | 1.683      
70  | 14 | 0      | 0.9697    | 0.01336     | 0.6779    | 63              | 2.697        | 1.828      
70  | 15 | 0      | 0.9734    | 0.02217     | 0.6886    | 0               | 2.45         | 1.687      
70  | 16 | 0      | 0.9773    | 0.01636     | 0.7055    | 0               | 2.476        | 1.747      
70  | 17 | 0      | 0.9792    | 0.005237    | 0.7455    | 0               | 2.461        | 1.835      
70  | 18 | 0      | 0.9747    | 0.01418     | 0.7011    | 0               | 2.418        | 1.695      
70  | 19 | 0      | 0.9687    | 0.01659     | 0.7325    | 0               | 2.625        | 1.923      
70  | 20 | 0      | 0.8723    | 0.03611     | 0.6056    | 0               | 2.624        | 1.589      
120 | 2  | 0      | 0.5409    | 0.3192      | 1.307     | 63.01           | 3.36         | 4.392      
120 | 3  | 0      | 0.311     | 0.2002      | 1.36      | 0               | 2.98         | 4.052      
120 | 4  | 0      | 0.2127    | 0.2351      | 0.2057    | 88.39           | 16.5         | 3.394      
120 | 5  | 0      | 0.7148    | 0.1354      | 0.2707    | 63              | 11.33        | 3.066      
120 | 6  | 0      | -0.2236   | 0.4012      | 0.4595    | 38.08           | 6.218        | 2.857      
120 | 7  | 0      | -0.02137  | 0.2316      | 0.2106    | 63              | 12.58        | 2.648      
120 | 8  | 0      | -0.1796   | 0.3571      | 0.09177   | 61.01           | 28.57        | 2.622      
120 | 9  | 0      | 0.8161    | 0.07178     | 0.37      | 63              | 6.008        | 2.223      
120 | 10 | 0      | 0.8664    | 0.1027      | 0.3818    | 63              | 5.474        | 2.09       
120 | 11 | 0      | 0.7237    | 0.07288     | 0.857     | 63              | 4.551        | 3.9        
120 | 12 | 0      | 0.8251    | 0.0634      | 0.8258    | 0               | 4.373        | 3.611      
120 | 13 | 0      | 0.7052    | 0.2246      | 0.1721    | 58.22           | 17.51        | 3.014      
120 | 14 | 0      | 0.9106    | 0.0357      | 0.5358    | 63              | 5.82         | 3.118      
120 | 15 | 0      | 0.9077    | 0.05058     | 0.5706    | 0               | 4.674        | 2.667      
120 | 16 | 0      | 0.9479    | 0.05113     | 0.5791    | 0               | 4.636        | 2.685      
120 | 17 | 0      | 0.9501    | 0.021       | 0.642     | 0               | 4.568        | 2.933      
120 | 18 | 0      | 0.9756    | 0.01168     | 0.5675    | 0               | 4.573        | 2.595      
120 | 19 | 0      | 0.9728    | 0.01751     | 0.6335    | 0               | 5.254        | 3.328      
120 | 20 | 0      | 0.9266    | 0.0253      | 0.4865    | 0               | 4.48         | 2.179      
270 | 2  | 0      | 0.8928    | 0.09081     | 1.396     | 1               | 27.74        | 38.71      
270 | 3  | 0      | 0.3628    | 0.17        | 2.959     | 63              | 12.02        | 35.57      
270 | 4  | 0      | 0.1191    | 0.1425      | 0.9286    | 63.01           | 27.6         | 25.63      
270 | 5  | 0      | 0.4581    | 0.1585      | 0.4036    | 85.59           | 49.96        | 20.16      
270 | 6  | 0      | 0.5257    | 0.1253      | 1.821     | 56.08           | 9.456        | 17.22      
270 | 7  | 0      | 0.186     | 0.1527      | 0.4649    | 86.28           | 42.06        | 19.55      
270 | 8  | 0      | 0.4741    | 0.1437      | 1.096     | 62              | 13.89        | 15.22      
270 | 9  | 0      | 0.48      | 0.1022      | 0.9674    | 88.39           | 12.75        | 12.33      
270 | 10 | 0      | 0.7337    | 0.08583     | 0.487     | 0               | 21.55        | 10.5       
270 | 11 | 0      | 0.7904    | 0.09451     | 2.138     | 60              | 11.65        | 24.91      
270 | 12 | 0      | 0.7352    | 0.1088      | 1.663     | 1               | 15.96        | 26.53      
270 | 13 | 0      | 0.6951    | 0.1302      | 1.425     | 83.45           | 11.1         | 15.81      
270 | 14 | 0      | 0.1426    | 0.1882      | 2.934     | 88.39           | 7.535        | 22.1       
270 | 15 | 0      | 0.8435    | 0.06056     | 1.305     | 0               | 12.25        | 15.98      
270 | 16 | 0      | 0.4048    | 0.1329      | 0.3609    | 63              | 39.92        | 14.41      
270 | 17 | 0      | -0.02514  | 0.1602      | 0.8341    | 88.39           | 18.65        | 15.56      
270 | 18 | 0      | 0.8249    | 0.1085      | 0.6345    | 4.123           | 25.63        | 16.26      
270 | 19 | 0      | 0.08326   | 0.1723      | 2.355     | 63.01           | 11           | 25.9       
270 | 20 | 0      | 0.417     | 0.1224      | 1.149     | 88.39           | 10           | 11.49      
400 | 2  | 0      | 0.6685    | 0.1193      | 0.4926    | 62              | 18.57        | 9.146      
400 | 3  | 0      | 0.9142    | 0.03143     | 1.015     | 62              | 10.29        | 10.44      
400 | 4  | 0      | 0.7303    | 0.09347     | 0.4365    | 0               | 21.23        | 9.266      
400 | 5  | 0      | 0.8666    | 0.04165     | 0.6733    | 63              | 15.26        | 10.28      
400 | 6  | 0      | 0.9       | 0.03881     | 0.8355    | 0               | 12.43        | 10.38      
400 | 7  | 0      | 0.8659    | 0.0592      | 0.6288    | 63              | 13.22        | 8.313      
400 | 8  | 0      | 0.7021    | 0.09527     | 0.504     | 0               | 19.15        | 9.652      
400 | 9  | 0      | 0.8927    | 0.04075     | 0.769     | 0               | 11.63        | 8.94       
400 | 10 | 0      | 0.8747    | 0.06312     | 0.9148    | 0               | 9.06         | 8.288      
400 | 11 | 0      | 0.8572    | 0.063       | 0.913     | 0               | 8.384        | 7.654      
400 | 12 | 0      | 0.7949    | 0.07008     | 0.892     | 0               | 8.934        | 7.969      
400 | 13 | 0      | 0.3812    | 0.1442      | 0.4261    | 31              | 17.67        | 7.53       
400 | 14 | 0      | 0.6894    | 0.1587      | 0.2693    | 76.03           | 27.11        | 7.299      
400 | 15 | 0      | 0.8845    | 0.0389      | 0.7968    | 0               | 8.978        | 7.154      
400 | 16 | 0      | 0.8912    | 0.06705     | 0.7535    | 63              | 9.139        | 6.886      
400 | 17 | 0      | -0.04827  | 0.1785      | 0.8944    | 51.42           | 7.443        | 6.657      
400 | 18 | 0      | 0.7131    | 0.09838     | 1.239     | 11              | 9.444        | 11.7       
400 | 19 | 0      | 0.0713    | 0.1768      | 0.6057    | 57              | 17.39        | 10.53      
400 | 20 | 0      | 0.727     | 0.08279     | 0.608     | 0               | 15.92        | 9.681      
```

## Mean by MHz (primary)

```
mhz | n  | pearson_r_mean | pattern_mae_mean | max_ratio_mean | peak_loc_err_px_mean | real_max_ohm_mean | gen_max_ohm_mean
----|----|----------------|------------------|----------------|----------------------|-------------------|-----------------
10  | 19 | 0.9282         | 0.05967          | 0.6387         | 19.89                | 0.3424            | 0.2181          
70  | 19 | 0.9516         | 0.02881          | 0.6966         | 13.26                | 2.836             | 1.962           
120 | 19 | 0.6254         | 0.1383           | 0.554          | 36.14                | 8.077             | 3.02            
270 | 19 | 0.4813         | 0.1289           | 1.333          | 55.01                | 20.04             | 20.2            
400 | 19 | 0.704          | 0.08744          | 0.7193         | 28.39                | 13.75             | 8.83            
```

## Agent copy block

```
sim_compare_metrics: n=95
  10.0MHz K2 s0: r=0.912 pattern_mae=0.180 max_ratio=0.748 real_max=0.329 gen_max=0.246 peak_err=0.0px
  10.0MHz K3 s0: r=0.936 pattern_mae=0.083 max_ratio=0.679 real_max=0.369 gen_max=0.251 peak_err=63.0px
  10.0MHz K4 s0: r=0.957 pattern_mae=0.101 max_ratio=0.701 real_max=0.327 gen_max=0.23 peak_err=0.0px
  10.0MHz K5 s0: r=0.968 pattern_mae=0.107 max_ratio=0.632 real_max=0.346 gen_max=0.219 peak_err=63.0px
  10.0MHz K6 s0: r=0.854 pattern_mae=0.072 max_ratio=0.604 real_max=0.368 gen_max=0.223 peak_err=0.0px
  10.0MHz K7 s0: r=0.967 pattern_mae=0.041 max_ratio=0.659 real_max=0.349 gen_max=0.23 peak_err=63.0px
  10.0MHz K8 s0: r=0.920 pattern_mae=0.045 max_ratio=0.593 real_max=0.375 gen_max=0.222 peak_err=63.0px
  10.0MHz K9 s0: r=0.926 pattern_mae=0.038 max_ratio=0.630 real_max=0.345 gen_max=0.218 peak_err=0.0px
  10.0MHz K10 s0: r=0.944 pattern_mae=0.032 max_ratio=0.661 real_max=0.359 gen_max=0.237 peak_err=0.0px
  10.0MHz K11 s0: r=0.905 pattern_mae=0.056 max_ratio=0.659 real_max=0.324 gen_max=0.214 peak_err=0.0px
  10.0MHz K12 s0: r=0.938 pattern_mae=0.040 max_ratio=0.663 real_max=0.323 gen_max=0.214 peak_err=0.0px
  10.0MHz K13 s0: r=0.869 pattern_mae=0.055 max_ratio=0.519 real_max=0.389 gen_max=0.202 peak_err=63.0px
  10.0MHz K14 s0: r=0.973 pattern_mae=0.044 max_ratio=0.601 real_max=0.357 gen_max=0.215 peak_err=63.0px
  10.0MHz K15 s0: r=0.917 pattern_mae=0.041 max_ratio=0.610 real_max=0.331 gen_max=0.202 peak_err=0.0px
  10.0MHz K16 s0: r=0.926 pattern_mae=0.058 max_ratio=0.591 real_max=0.34 gen_max=0.201 peak_err=0.0px
  10.0MHz K17 s0: r=0.950 pattern_mae=0.034 max_ratio=0.629 real_max=0.333 gen_max=0.21 peak_err=0.0px
  10.0MHz K18 s0: r=0.907 pattern_mae=0.041 max_ratio=0.602 real_max=0.332 gen_max=0.2 peak_err=0.0px
  10.0MHz K19 s0: r=0.966 pattern_mae=0.032 max_ratio=0.696 real_max=0.337 gen_max=0.235 peak_err=0.0px
  10.0MHz K20 s0: r=0.902 pattern_mae=0.035 max_ratio=0.659 real_max=0.271 gen_max=0.178 peak_err=0.0px
  70.0MHz K2 s0: r=0.967 pattern_mae=0.051 max_ratio=0.839 real_max=3.47 gen_max=2.91 peak_err=0.0px
  70.0MHz K3 s0: r=0.908 pattern_mae=0.085 max_ratio=0.615 real_max=4.58 gen_max=2.82 peak_err=0.0px
  70.0MHz K4 s0: r=0.932 pattern_mae=0.063 max_ratio=0.673 real_max=2.95 gen_max=1.98 peak_err=0.0px
  70.0MHz K5 s0: r=0.992 pattern_mae=0.017 max_ratio=0.709 real_max=2.85 gen_max=2.02 peak_err=0.0px
  70.0MHz K6 s0: r=0.964 pattern_mae=0.026 max_ratio=0.644 real_max=3.1 gen_max=1.99 peak_err=0.0px
  70.0MHz K7 s0: r=0.974 pattern_mae=0.016 max_ratio=0.624 real_max=2.95 gen_max=1.84 peak_err=63.0px
  70.0MHz K8 s0: r=0.894 pattern_mae=0.048 max_ratio=0.615 real_max=3 gen_max=1.85 peak_err=63.0px
  70.0MHz K9 s0: r=0.977 pattern_mae=0.014 max_ratio=0.744 real_max=2.63 gen_max=1.95 peak_err=0.0px
  70.0MHz K10 s0: r=0.965 pattern_mae=0.016 max_ratio=0.777 real_max=2.66 gen_max=2.07 peak_err=0.0px
  70.0MHz K11 s0: r=0.943 pattern_mae=0.020 max_ratio=0.802 real_max=2.42 gen_max=1.94 peak_err=63.0px
  70.0MHz K12 s0: r=0.973 pattern_mae=0.014 max_ratio=0.798 real_max=2.41 gen_max=1.92 peak_err=0.0px
  70.0MHz K13 s0: r=0.876 pattern_mae=0.053 max_ratio=0.541 real_max=3.11 gen_max=1.68 peak_err=0.0px
  70.0MHz K14 s0: r=0.970 pattern_mae=0.013 max_ratio=0.678 real_max=2.7 gen_max=1.83 peak_err=63.0px
  70.0MHz K15 s0: r=0.973 pattern_mae=0.022 max_ratio=0.689 real_max=2.45 gen_max=1.69 peak_err=0.0px
  70.0MHz K16 s0: r=0.977 pattern_mae=0.016 max_ratio=0.705 real_max=2.48 gen_max=1.75 peak_err=0.0px
  70.0MHz K17 s0: r=0.979 pattern_mae=0.005 max_ratio=0.745 real_max=2.46 gen_max=1.83 peak_err=0.0px
  70.0MHz K18 s0: r=0.975 pattern_mae=0.014 max_ratio=0.701 real_max=2.42 gen_max=1.7 peak_err=0.0px
  70.0MHz K19 s0: r=0.969 pattern_mae=0.017 max_ratio=0.733 real_max=2.63 gen_max=1.92 peak_err=0.0px
  70.0MHz K20 s0: r=0.872 pattern_mae=0.036 max_ratio=0.606 real_max=2.62 gen_max=1.59 peak_err=0.0px
  120.0MHz K2 s0: r=0.541 pattern_mae=0.319 max_ratio=1.307 real_max=3.36 gen_max=4.39 peak_err=63.0px
  120.0MHz K3 s0: r=0.311 pattern_mae=0.200 max_ratio=1.360 real_max=2.98 gen_max=4.05 peak_err=0.0px
  120.0MHz K4 s0: r=0.213 pattern_mae=0.235 max_ratio=0.206 real_max=16.5 gen_max=3.39 peak_err=88.4px
  120.0MHz K5 s0: r=0.715 pattern_mae=0.135 max_ratio=0.271 real_max=11.3 gen_max=3.07 peak_err=63.0px
  120.0MHz K6 s0: r=-0.224 pattern_mae=0.401 max_ratio=0.459 real_max=6.22 gen_max=2.86 peak_err=38.1px
  120.0MHz K7 s0: r=-0.021 pattern_mae=0.232 max_ratio=0.211 real_max=12.6 gen_max=2.65 peak_err=63.0px
  120.0MHz K8 s0: r=-0.180 pattern_mae=0.357 max_ratio=0.092 real_max=28.6 gen_max=2.62 peak_err=61.0px
  120.0MHz K9 s0: r=0.816 pattern_mae=0.072 max_ratio=0.370 real_max=6.01 gen_max=2.22 peak_err=63.0px
  120.0MHz K10 s0: r=0.866 pattern_mae=0.103 max_ratio=0.382 real_max=5.47 gen_max=2.09 peak_err=63.0px
  120.0MHz K11 s0: r=0.724 pattern_mae=0.073 max_ratio=0.857 real_max=4.55 gen_max=3.9 peak_err=63.0px
  120.0MHz K12 s0: r=0.825 pattern_mae=0.063 max_ratio=0.826 real_max=4.37 gen_max=3.61 peak_err=0.0px
  120.0MHz K13 s0: r=0.705 pattern_mae=0.225 max_ratio=0.172 real_max=17.5 gen_max=3.01 peak_err=58.2px
  120.0MHz K14 s0: r=0.911 pattern_mae=0.036 max_ratio=0.536 real_max=5.82 gen_max=3.12 peak_err=63.0px
  120.0MHz K15 s0: r=0.908 pattern_mae=0.051 max_ratio=0.571 real_max=4.67 gen_max=2.67 peak_err=0.0px
  120.0MHz K16 s0: r=0.948 pattern_mae=0.051 max_ratio=0.579 real_max=4.64 gen_max=2.68 peak_err=0.0px
  120.0MHz K17 s0: r=0.950 pattern_mae=0.021 max_ratio=0.642 real_max=4.57 gen_max=2.93 peak_err=0.0px
  120.0MHz K18 s0: r=0.976 pattern_mae=0.012 max_ratio=0.567 real_max=4.57 gen_max=2.6 peak_err=0.0px
  120.0MHz K19 s0: r=0.973 pattern_mae=0.018 max_ratio=0.634 real_max=5.25 gen_max=3.33 peak_err=0.0px
  120.0MHz K20 s0: r=0.927 pattern_mae=0.025 max_ratio=0.486 real_max=4.48 gen_max=2.18 peak_err=0.0px
  270.0MHz K2 s0: r=0.893 pattern_mae=0.091 max_ratio=1.396 real_max=27.7 gen_max=38.7 peak_err=1.0px
  270.0MHz K3 s0: r=0.363 pattern_mae=0.170 max_ratio=2.959 real_max=12 gen_max=35.6 peak_err=63.0px
  270.0MHz K4 s0: r=0.119 pattern_mae=0.143 max_ratio=0.929 real_max=27.6 gen_max=25.6 peak_err=63.0px
  270.0MHz K5 s0: r=0.458 pattern_mae=0.158 max_ratio=0.404 real_max=50 gen_max=20.2 peak_err=85.6px
  270.0MHz K6 s0: r=0.526 pattern_mae=0.125 max_ratio=1.821 real_max=9.46 gen_max=17.2 peak_err=56.1px
  270.0MHz K7 s0: r=0.186 pattern_mae=0.153 max_ratio=0.465 real_max=42.1 gen_max=19.6 peak_err=86.3px
  270.0MHz K8 s0: r=0.474 pattern_mae=0.144 max_ratio=1.096 real_max=13.9 gen_max=15.2 peak_err=62.0px
  270.0MHz K9 s0: r=0.480 pattern_mae=0.102 max_ratio=0.967 real_max=12.7 gen_max=12.3 peak_err=88.4px
  270.0MHz K10 s0: r=0.734 pattern_mae=0.086 max_ratio=0.487 real_max=21.6 gen_max=10.5 peak_err=0.0px
  270.0MHz K11 s0: r=0.790 pattern_mae=0.095 max_ratio=2.138 real_max=11.6 gen_max=24.9 peak_err=60.0px
  270.0MHz K12 s0: r=0.735 pattern_mae=0.109 max_ratio=1.663 real_max=16 gen_max=26.5 peak_err=1.0px
  270.0MHz K13 s0: r=0.695 pattern_mae=0.130 max_ratio=1.425 real_max=11.1 gen_max=15.8 peak_err=83.5px
  270.0MHz K14 s0: r=0.143 pattern_mae=0.188 max_ratio=2.934 real_max=7.53 gen_max=22.1 peak_err=88.4px
  270.0MHz K15 s0: r=0.844 pattern_mae=0.061 max_ratio=1.305 real_max=12.2 gen_max=16 peak_err=0.0px
  270.0MHz K16 s0: r=0.405 pattern_mae=0.133 max_ratio=0.361 real_max=39.9 gen_max=14.4 peak_err=63.0px
  270.0MHz K17 s0: r=-0.025 pattern_mae=0.160 max_ratio=0.834 real_max=18.7 gen_max=15.6 peak_err=88.4px
  270.0MHz K18 s0: r=0.825 pattern_mae=0.108 max_ratio=0.635 real_max=25.6 gen_max=16.3 peak_err=4.1px
  270.0MHz K19 s0: r=0.083 pattern_mae=0.172 max_ratio=2.355 real_max=11 gen_max=25.9 peak_err=63.0px
  270.0MHz K20 s0: r=0.417 pattern_mae=0.122 max_ratio=1.149 real_max=10 gen_max=11.5 peak_err=88.4px
  400.0MHz K2 s0: r=0.668 pattern_mae=0.119 max_ratio=0.493 real_max=18.6 gen_max=9.15 peak_err=62.0px
  400.0MHz K3 s0: r=0.914 pattern_mae=0.031 max_ratio=1.015 real_max=10.3 gen_max=10.4 peak_err=62.0px
  400.0MHz K4 s0: r=0.730 pattern_mae=0.093 max_ratio=0.437 real_max=21.2 gen_max=9.27 peak_err=0.0px
  400.0MHz K5 s0: r=0.867 pattern_mae=0.042 max_ratio=0.673 real_max=15.3 gen_max=10.3 peak_err=63.0px
  400.0MHz K6 s0: r=0.900 pattern_mae=0.039 max_ratio=0.836 real_max=12.4 gen_max=10.4 peak_err=0.0px
  400.0MHz K7 s0: r=0.866 pattern_mae=0.059 max_ratio=0.629 real_max=13.2 gen_max=8.31 peak_err=63.0px
  400.0MHz K8 s0: r=0.702 pattern_mae=0.095 max_ratio=0.504 real_max=19.2 gen_max=9.65 peak_err=0.0px
  400.0MHz K9 s0: r=0.893 pattern_mae=0.041 max_ratio=0.769 real_max=11.6 gen_max=8.94 peak_err=0.0px
  400.0MHz K10 s0: r=0.875 pattern_mae=0.063 max_ratio=0.915 real_max=9.06 gen_max=8.29 peak_err=0.0px
  400.0MHz K11 s0: r=0.857 pattern_mae=0.063 max_ratio=0.913 real_max=8.38 gen_max=7.65 peak_err=0.0px
  400.0MHz K12 s0: r=0.795 pattern_mae=0.070 max_ratio=0.892 real_max=8.93 gen_max=7.97 peak_err=0.0px
  400.0MHz K13 s0: r=0.381 pattern_mae=0.144 max_ratio=0.426 real_max=17.7 gen_max=7.53 peak_err=31.0px
  400.0MHz K14 s0: r=0.689 pattern_mae=0.159 max_ratio=0.269 real_max=27.1 gen_max=7.3 peak_err=76.0px
  400.0MHz K15 s0: r=0.885 pattern_mae=0.039 max_ratio=0.797 real_max=8.98 gen_max=7.15 peak_err=0.0px
  400.0MHz K16 s0: r=0.891 pattern_mae=0.067 max_ratio=0.753 real_max=9.14 gen_max=6.89 peak_err=63.0px
  400.0MHz K17 s0: r=-0.048 pattern_mae=0.178 max_ratio=0.894 real_max=7.44 gen_max=6.66 peak_err=51.4px
  400.0MHz K18 s0: r=0.713 pattern_mae=0.098 max_ratio=1.239 real_max=9.44 gen_max=11.7 peak_err=11.0px
  400.0MHz K19 s0: r=0.071 pattern_mae=0.177 max_ratio=0.606 real_max=17.4 gen_max=10.5 peak_err=57.0px
  400.0MHz K20 s0: r=0.727 pattern_mae=0.083 max_ratio=0.608 real_max=15.9 gen_max=9.68 peak_err=0.0px
```
