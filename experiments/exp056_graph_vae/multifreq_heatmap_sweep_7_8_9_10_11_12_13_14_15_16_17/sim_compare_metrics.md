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
10  | 7  | 0      | 0.8992    | 0.07213      | 0.02684     | 0.7983    | 0               | 0.3577       | 0.2855     
10  | 8  | 0      | 0.9762    | 0.05926      | 0.008682    | 0.8319    | 0               | 0.3526       | 0.2933     
10  | 9  | 0      | 0.986     | 0.02447      | 0.01496     | 0.92      | 0               | 0.3059       | 0.2814     
10  | 10 | 0      | 0.972     | 0.04726      | 0.009091    | 0.8594    | 0               | 0.3361       | 0.2888     
10  | 11 | 0      | 0.9524    | 0.05214      | 0.01517     | 0.8384    | 0               | 0.3225       | 0.2704     
10  | 12 | 0      | 0.9722    | 0.05065      | 0.01378     | 0.8423    | 0               | 0.3212       | 0.2706     
10  | 13 | 0      | 0.9614    | 0.06363      | 0.02194     | 0.8114    | 0               | 0.3374       | 0.2738     
10  | 14 | 0      | 0.9389    | 0.04854      | 0.01509     | 0.8481    | 0               | 0.3196       | 0.271      
10  | 15 | 0      | 0.9267    | 0.05682      | 0.02652     | 0.8253    | 63              | 0.3252       | 0.2684     
10  | 16 | 0      | 0.9461    | 0.03119      | 0.01875     | 0.8921    | 63              | 0.2892       | 0.258      
10  | 17 | 0      | 0.9721    | 0.03257      | 0.01104     | 0.8937    | 0               | 0.3065       | 0.2739     
63  | 7  | 0      | 0.9696    | 0.4385       | 0.008553    | 0.8227    | 0               | 2.474        | 2.035      
63  | 8  | 0      | 0.99      | 0.3368       | 0.002504    | 0.8595    | 0               | 2.397        | 2.061      
63  | 9  | 0      | 0.9927    | 0.1909       | 0.002155    | 0.9117    | 0               | 2.161        | 1.97       
63  | 10 | 0      | 0.9677    | 0.2942       | 0.01576     | 0.8708    | 63              | 2.277        | 1.982      
63  | 11 | 0      | 0.9825    | 0.2176       | 0.003002    | 0.8994    | 63              | 2.164        | 1.947      
63  | 12 | 0      | 0.9768    | 0.1759       | 0.006061    | 0.9188    | 0               | 2.167        | 1.991      
63  | 13 | 0      | 0.9663    | 0.2572       | 0.009893    | 0.8822    | 0               | 2.183        | 1.926      
63  | 14 | 0      | 0.9272    | 0.1377       | 0.02318     | 0.9337    | 0               | 2.075        | 1.938      
63  | 15 | 0      | 0.9798    | 0.3577       | 0.005392    | 0.8375    | 0               | 2.201        | 1.843      
63  | 16 | 0      | 0.9597    | 0.1879       | 0.01339     | 0.9065    | 63              | 2.01         | 1.822      
63  | 17 | 0      | 0.9863    | 0.2369       | 0.004352    | 0.8866    | 0               | 2.09         | 1.853      
150 | 7  | 0      | 0.9244    | -2.083       | 0.08912     | 1.371     | 11.66           | 5.611        | 7.695      
150 | 8  | 0      | 0.9762    | 7.563        | 0.02677     | 0.4891    | 0               | 14.81        | 7.242      
150 | 9  | 0      | 0.9616    | -0.598       | 0.03043     | 1.092     | 62              | 6.476        | 7.074      
150 | 10 | 0      | 0.9449    | 0.5064       | 0.02777     | 0.9286    | 63              | 7.092        | 6.585      
150 | 11 | 0      | 0.9775    | 0.4318       | 0.005375    | 0.9325    | 0               | 6.4          | 5.968      
150 | 12 | 0      | 0.9348    | -0.696       | 0.0365      | 1.106     | 0               | 6.539        | 7.235      
150 | 13 | 0      | 0.947     | -0.2715      | 0.02111     | 1.043     | 0               | 6.307        | 6.578      
150 | 14 | 0      | 0.9782    | 0.1826       | 0.01426     | 0.9709    | 0               | 6.277        | 6.095      
150 | 15 | 0      | 0.9792    | -0.1949      | 0.008666    | 1.031     | 0               | 6.194        | 6.389      
150 | 16 | 0      | 0.9167    | 0.4911       | 0.03027     | 0.9104    | 63              | 5.482        | 4.991      
150 | 17 | 0      | 0.986     | 0.2742       | 0.003512    | 0.9512    | 0               | 5.614        | 5.34       
270 | 7  | 0      | 0.4384    | 1.19         | 0.0966      | 0.8869    | 58.08           | 10.52        | 9.33       
270 | 8  | 0      | 0.6731    | 5.798        | 0.0579      | 0.6815    | 62              | 18.2         | 12.41      
270 | 9  | 0      | 0.9447    | 26.68        | 0.03883     | 0.4442    | 0               | 48.01        | 21.33      
270 | 10 | 0      | 0.8325    | 2.401        | 0.05368     | 0.8372    | 0               | 14.74        | 12.34      
270 | 11 | 0      | 0.8914    | 0.2104       | 0.05403     | 0.9763    | 56.08           | 8.866        | 8.656      
270 | 12 | 0      | 0.731     | 11.29        | 0.07623     | 0.5965    | 63              | 27.99        | 16.69      
270 | 13 | 0      | 0.6955    | -8.281       | 0.1539      | 2.348     | 62              | 6.144        | 14.42      
270 | 14 | 0      | 0.1876    | 3.212        | 0.1171      | 0.8078    | 58.03           | 16.71        | 13.5       
270 | 15 | 0      | 0.8424    | 0.2824       | 0.06742     | 0.9663    | 0               | 8.371        | 8.089      
270 | 16 | 0      | 0.6513    | -2.656       | 0.06575     | 1.19      | 0               | 14.01        | 16.67      
270 | 17 | 0      | 0.6674    | 38.75        | 0.09175     | 0.2636    | 62              | 52.62        | 13.87      
450 | 7  | 0      | 0.5714    | -0.6469      | 0.06834     | 1.073     | 0               | 8.908        | 9.555      
450 | 8  | 0      | 0.7343    | 10.05        | 0.1019      | 0.4873    | 58.52           | 19.6         | 9.55       
450 | 9  | 0      | 0.959     | 0.5261       | 0.02046     | 0.9344    | 3               | 8.017        | 7.491      
450 | 10 | 0      | 0.8946    | 6.883        | 0.04621     | 0.5412    | 0               | 15           | 8.119      
450 | 11 | 0      | 0.9631    | 4.814        | 0.0189      | 0.6923    | 0               | 15.64        | 10.83      
450 | 12 | 0      | 0.5052    | 0.3031       | 0.116       | 0.9605    | 19              | 7.68         | 7.377      
450 | 13 | 0      | 0.8882    | 4.609        | 0.02607     | 0.6727    | 76.61           | 14.08        | 9.474      
450 | 14 | 0      | 0.8285    | 7.441        | 0.05542     | 0.5743    | 63              | 17.48        | 10.04      
450 | 15 | 0      | 0.8333    | 1.718        | 0.04396     | 0.8472    | 62              | 11.24        | 9.527      
450 | 16 | 0      | 0.8405    | 6.376        | 0.04006     | 0.5932    | 0               | 15.67        | 9.298      
450 | 17 | 0      | 0.694     | 6.456        | 0.1165      | 0.6077    | 62              | 16.46        | 10         
500 | 7  | 0      | 0.7991    | 4.784        | 0.07255     | 0.7349    | 0               | 18.05        | 13.26      
500 | 8  | 0      | 0.7766    | 5.534        | 0.06465     | 0.6605    | 0               | 16.3         | 10.77      
500 | 9  | 0      | 0.882     | 12.33        | 0.0605      | 0.478     | 1               | 23.63        | 11.29      
500 | 10 | 0      | 0.9358    | 3.444        | 0.03014     | 0.7746    | 70.21           | 15.28        | 11.84      
500 | 11 | 0      | 0.8885    | 1.789        | 0.03882     | 0.871     | 88.39           | 13.86        | 12.08      
500 | 12 | 0      | 0.9526    | 1.523        | 0.03471     | 0.8886    | 32              | 13.67        | 12.15      
500 | 13 | 0      | 0.7678    | 2.446        | 0.05897     | 0.8134    | 0               | 13.11        | 10.66      
500 | 14 | 0      | 0.7766    | -0.718       | 0.0547      | 1.063     | 0               | 11.34        | 12.05      
500 | 15 | 0      | 0.7194    | 8.044        | 0.07474     | 0.6013    | 88.39           | 20.18        | 12.13      
500 | 16 | 0      | 0.8229    | 0.3714       | 0.05422     | 0.9654    | 88.39           | 10.75        | 10.37      
500 | 17 | 0      | 0.8693    | 2.751        | 0.0428      | 0.7902    | 70.66           | 13.11        | 10.36      
```

## Mean by MHz (primary)

```
mhz | n  | pearson_r_mean | max_diff_ohm_mean | pattern_mae_mean | max_ratio_mean | peak_loc_err_px_mean | real_max_ohm_mean | gen_max_ohm_mean
----|----|----------------|-------------------|------------------|----------------|----------------------|-------------------|-----------------
10  | 11 | 0.9548         | 0.04897           | 0.01653          | 0.851          | 11.45                | 0.3249            | 0.2759          
63  | 11 | 0.9726         | 0.2574            | 0.008568         | 0.8845         | 17.18                | 2.2               | 1.943           
150 | 11 | 0.957          | 0.5096            | 0.02671          | 0.9843         | 18.15                | 6.982             | 6.472           
270 | 11 | 0.6868         | 7.171             | 0.07939          | 0.9089         | 38.29                | 20.56             | 13.39           
450 | 11 | 0.792          | 4.411             | 0.05943          | 0.7258         | 31.28                | 13.62             | 9.205           
500 | 11 | 0.8355         | 3.846             | 0.05335          | 0.7856         | 39.91                | 15.39             | 11.54           
```

## Agent copy block

```
sim_compare_metrics: n=66
  10.0MHz K7 s0: r=0.899 Δmax=+0.0721Ω pattern_mae=0.027 max_ratio=0.798 real_max=0.358 gen_max=0.286 peak_err=0.0px
  10.0MHz K8 s0: r=0.976 Δmax=+0.0593Ω pattern_mae=0.009 max_ratio=0.832 real_max=0.353 gen_max=0.293 peak_err=0.0px
  10.0MHz K9 s0: r=0.986 Δmax=+0.0245Ω pattern_mae=0.015 max_ratio=0.920 real_max=0.306 gen_max=0.281 peak_err=0.0px
  10.0MHz K10 s0: r=0.972 Δmax=+0.0473Ω pattern_mae=0.009 max_ratio=0.859 real_max=0.336 gen_max=0.289 peak_err=0.0px
  10.0MHz K11 s0: r=0.952 Δmax=+0.0521Ω pattern_mae=0.015 max_ratio=0.838 real_max=0.323 gen_max=0.27 peak_err=0.0px
  10.0MHz K12 s0: r=0.972 Δmax=+0.0506Ω pattern_mae=0.014 max_ratio=0.842 real_max=0.321 gen_max=0.271 peak_err=0.0px
  10.0MHz K13 s0: r=0.961 Δmax=+0.0636Ω pattern_mae=0.022 max_ratio=0.811 real_max=0.337 gen_max=0.274 peak_err=0.0px
  10.0MHz K14 s0: r=0.939 Δmax=+0.0485Ω pattern_mae=0.015 max_ratio=0.848 real_max=0.32 gen_max=0.271 peak_err=0.0px
  10.0MHz K15 s0: r=0.927 Δmax=+0.0568Ω pattern_mae=0.027 max_ratio=0.825 real_max=0.325 gen_max=0.268 peak_err=63.0px
  10.0MHz K16 s0: r=0.946 Δmax=+0.0312Ω pattern_mae=0.019 max_ratio=0.892 real_max=0.289 gen_max=0.258 peak_err=63.0px
  10.0MHz K17 s0: r=0.972 Δmax=+0.0326Ω pattern_mae=0.011 max_ratio=0.894 real_max=0.306 gen_max=0.274 peak_err=0.0px
  63.0MHz K7 s0: r=0.970 Δmax=+0.438Ω pattern_mae=0.009 max_ratio=0.823 real_max=2.47 gen_max=2.04 peak_err=0.0px
  63.0MHz K8 s0: r=0.990 Δmax=+0.337Ω pattern_mae=0.003 max_ratio=0.860 real_max=2.4 gen_max=2.06 peak_err=0.0px
  63.0MHz K9 s0: r=0.993 Δmax=+0.191Ω pattern_mae=0.002 max_ratio=0.912 real_max=2.16 gen_max=1.97 peak_err=0.0px
  63.0MHz K10 s0: r=0.968 Δmax=+0.294Ω pattern_mae=0.016 max_ratio=0.871 real_max=2.28 gen_max=1.98 peak_err=63.0px
  63.0MHz K11 s0: r=0.982 Δmax=+0.218Ω pattern_mae=0.003 max_ratio=0.899 real_max=2.16 gen_max=1.95 peak_err=63.0px
  63.0MHz K12 s0: r=0.977 Δmax=+0.176Ω pattern_mae=0.006 max_ratio=0.919 real_max=2.17 gen_max=1.99 peak_err=0.0px
  63.0MHz K13 s0: r=0.966 Δmax=+0.257Ω pattern_mae=0.010 max_ratio=0.882 real_max=2.18 gen_max=1.93 peak_err=0.0px
  63.0MHz K14 s0: r=0.927 Δmax=+0.138Ω pattern_mae=0.023 max_ratio=0.934 real_max=2.08 gen_max=1.94 peak_err=0.0px
  63.0MHz K15 s0: r=0.980 Δmax=+0.358Ω pattern_mae=0.005 max_ratio=0.838 real_max=2.2 gen_max=1.84 peak_err=0.0px
  63.0MHz K16 s0: r=0.960 Δmax=+0.188Ω pattern_mae=0.013 max_ratio=0.907 real_max=2.01 gen_max=1.82 peak_err=63.0px
  63.0MHz K17 s0: r=0.986 Δmax=+0.237Ω pattern_mae=0.004 max_ratio=0.887 real_max=2.09 gen_max=1.85 peak_err=0.0px
  150.0MHz K7 s0: r=0.924 Δmax=-2.08Ω pattern_mae=0.089 max_ratio=1.371 real_max=5.61 gen_max=7.69 peak_err=11.7px
  150.0MHz K8 s0: r=0.976 Δmax=+7.56Ω pattern_mae=0.027 max_ratio=0.489 real_max=14.8 gen_max=7.24 peak_err=0.0px
  150.0MHz K9 s0: r=0.962 Δmax=-0.598Ω pattern_mae=0.030 max_ratio=1.092 real_max=6.48 gen_max=7.07 peak_err=62.0px
  150.0MHz K10 s0: r=0.945 Δmax=+0.506Ω pattern_mae=0.028 max_ratio=0.929 real_max=7.09 gen_max=6.59 peak_err=63.0px
  150.0MHz K11 s0: r=0.978 Δmax=+0.432Ω pattern_mae=0.005 max_ratio=0.933 real_max=6.4 gen_max=5.97 peak_err=0.0px
  150.0MHz K12 s0: r=0.935 Δmax=-0.696Ω pattern_mae=0.037 max_ratio=1.106 real_max=6.54 gen_max=7.24 peak_err=0.0px
  150.0MHz K13 s0: r=0.947 Δmax=-0.271Ω pattern_mae=0.021 max_ratio=1.043 real_max=6.31 gen_max=6.58 peak_err=0.0px
  150.0MHz K14 s0: r=0.978 Δmax=+0.183Ω pattern_mae=0.014 max_ratio=0.971 real_max=6.28 gen_max=6.09 peak_err=0.0px
  150.0MHz K15 s0: r=0.979 Δmax=-0.195Ω pattern_mae=0.009 max_ratio=1.031 real_max=6.19 gen_max=6.39 peak_err=0.0px
  150.0MHz K16 s0: r=0.917 Δmax=+0.491Ω pattern_mae=0.030 max_ratio=0.910 real_max=5.48 gen_max=4.99 peak_err=63.0px
  150.0MHz K17 s0: r=0.986 Δmax=+0.274Ω pattern_mae=0.004 max_ratio=0.951 real_max=5.61 gen_max=5.34 peak_err=0.0px
  270.0MHz K7 s0: r=0.438 Δmax=+1.19Ω pattern_mae=0.097 max_ratio=0.887 real_max=10.5 gen_max=9.33 peak_err=58.1px
  270.0MHz K8 s0: r=0.673 Δmax=+5.8Ω pattern_mae=0.058 max_ratio=0.681 real_max=18.2 gen_max=12.4 peak_err=62.0px
  270.0MHz K9 s0: r=0.945 Δmax=+26.7Ω pattern_mae=0.039 max_ratio=0.444 real_max=48 gen_max=21.3 peak_err=0.0px
  270.0MHz K10 s0: r=0.832 Δmax=+2.4Ω pattern_mae=0.054 max_ratio=0.837 real_max=14.7 gen_max=12.3 peak_err=0.0px
  270.0MHz K11 s0: r=0.891 Δmax=+0.21Ω pattern_mae=0.054 max_ratio=0.976 real_max=8.87 gen_max=8.66 peak_err=56.1px
  270.0MHz K12 s0: r=0.731 Δmax=+11.3Ω pattern_mae=0.076 max_ratio=0.596 real_max=28 gen_max=16.7 peak_err=63.0px
  270.0MHz K13 s0: r=0.695 Δmax=-8.28Ω pattern_mae=0.154 max_ratio=2.348 real_max=6.14 gen_max=14.4 peak_err=62.0px
  270.0MHz K14 s0: r=0.188 Δmax=+3.21Ω pattern_mae=0.117 max_ratio=0.808 real_max=16.7 gen_max=13.5 peak_err=58.0px
  270.0MHz K15 s0: r=0.842 Δmax=+0.282Ω pattern_mae=0.067 max_ratio=0.966 real_max=8.37 gen_max=8.09 peak_err=0.0px
  270.0MHz K16 s0: r=0.651 Δmax=-2.66Ω pattern_mae=0.066 max_ratio=1.190 real_max=14 gen_max=16.7 peak_err=0.0px
  270.0MHz K17 s0: r=0.667 Δmax=+38.8Ω pattern_mae=0.092 max_ratio=0.264 real_max=52.6 gen_max=13.9 peak_err=62.0px
  450.0MHz K7 s0: r=0.571 Δmax=-0.647Ω pattern_mae=0.068 max_ratio=1.073 real_max=8.91 gen_max=9.56 peak_err=0.0px
  450.0MHz K8 s0: r=0.734 Δmax=+10Ω pattern_mae=0.102 max_ratio=0.487 real_max=19.6 gen_max=9.55 peak_err=58.5px
  450.0MHz K9 s0: r=0.959 Δmax=+0.526Ω pattern_mae=0.020 max_ratio=0.934 real_max=8.02 gen_max=7.49 peak_err=3.0px
  450.0MHz K10 s0: r=0.895 Δmax=+6.88Ω pattern_mae=0.046 max_ratio=0.541 real_max=15 gen_max=8.12 peak_err=0.0px
  450.0MHz K11 s0: r=0.963 Δmax=+4.81Ω pattern_mae=0.019 max_ratio=0.692 real_max=15.6 gen_max=10.8 peak_err=0.0px
  450.0MHz K12 s0: r=0.505 Δmax=+0.303Ω pattern_mae=0.116 max_ratio=0.961 real_max=7.68 gen_max=7.38 peak_err=19.0px
  450.0MHz K13 s0: r=0.888 Δmax=+4.61Ω pattern_mae=0.026 max_ratio=0.673 real_max=14.1 gen_max=9.47 peak_err=76.6px
  450.0MHz K14 s0: r=0.828 Δmax=+7.44Ω pattern_mae=0.055 max_ratio=0.574 real_max=17.5 gen_max=10 peak_err=63.0px
  450.0MHz K15 s0: r=0.833 Δmax=+1.72Ω pattern_mae=0.044 max_ratio=0.847 real_max=11.2 gen_max=9.53 peak_err=62.0px
  450.0MHz K16 s0: r=0.841 Δmax=+6.38Ω pattern_mae=0.040 max_ratio=0.593 real_max=15.7 gen_max=9.3 peak_err=0.0px
  450.0MHz K17 s0: r=0.694 Δmax=+6.46Ω pattern_mae=0.116 max_ratio=0.608 real_max=16.5 gen_max=10 peak_err=62.0px
  500.0MHz K7 s0: r=0.799 Δmax=+4.78Ω pattern_mae=0.073 max_ratio=0.735 real_max=18 gen_max=13.3 peak_err=0.0px
  500.0MHz K8 s0: r=0.777 Δmax=+5.53Ω pattern_mae=0.065 max_ratio=0.661 real_max=16.3 gen_max=10.8 peak_err=0.0px
  500.0MHz K9 s0: r=0.882 Δmax=+12.3Ω pattern_mae=0.061 max_ratio=0.478 real_max=23.6 gen_max=11.3 peak_err=1.0px
  500.0MHz K10 s0: r=0.936 Δmax=+3.44Ω pattern_mae=0.030 max_ratio=0.775 real_max=15.3 gen_max=11.8 peak_err=70.2px
  500.0MHz K11 s0: r=0.889 Δmax=+1.79Ω pattern_mae=0.039 max_ratio=0.871 real_max=13.9 gen_max=12.1 peak_err=88.4px
  500.0MHz K12 s0: r=0.953 Δmax=+1.52Ω pattern_mae=0.035 max_ratio=0.889 real_max=13.7 gen_max=12.1 peak_err=32.0px
  500.0MHz K13 s0: r=0.768 Δmax=+2.45Ω pattern_mae=0.059 max_ratio=0.813 real_max=13.1 gen_max=10.7 peak_err=0.0px
  500.0MHz K14 s0: r=0.777 Δmax=-0.718Ω pattern_mae=0.055 max_ratio=1.063 real_max=11.3 gen_max=12.1 peak_err=0.0px
  500.0MHz K15 s0: r=0.719 Δmax=+8.04Ω pattern_mae=0.075 max_ratio=0.601 real_max=20.2 gen_max=12.1 peak_err=88.4px
  500.0MHz K16 s0: r=0.823 Δmax=+0.371Ω pattern_mae=0.054 max_ratio=0.965 real_max=10.7 gen_max=10.4 peak_err=88.4px
  500.0MHz K17 s0: r=0.869 Δmax=+2.75Ω pattern_mae=0.043 max_ratio=0.790 real_max=13.1 gen_max=10.4 peak_err=70.7px
```
