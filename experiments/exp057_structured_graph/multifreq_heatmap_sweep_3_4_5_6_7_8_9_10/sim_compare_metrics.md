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
10  | 3  | 0      | 0.9865    | 0.009872     | 0.004101    | 0.9722    | 0               | 0.3556       | 0.3458     
10  | 4  | 0      | 0.9886    | 0.01312      | 0.01063     | 0.9609    | 0               | 0.3352       | 0.3221     
10  | 5  | 0      | 0.9934    | 0.0105       | 0.002672    | 0.9704    | 0               | 0.3554       | 0.3449     
10  | 6  | 0      | 0.9949    | 0.0175       | 0.004074    | 0.9478    | 0               | 0.3351       | 0.3176     
10  | 7  | 0      | 0.9693    | 0.00407      | 0.01052     | 0.9881    | 0               | 0.3424       | 0.3383     
10  | 8  | 0      | 0.9877    | 0.008045     | 0.00143     | 0.9769    | 0               | 0.3489       | 0.3409     
10  | 9  | 0      | 0.988     | 0.004623     | 0.01243     | 0.9854    | 63              | 0.3163       | 0.3117     
10  | 10 | 0      | 0.9438    | -0.000536    | 0.01226     | 1.002     | 0               | 0.3339       | 0.3345     
63  | 3  | 0      | 0.9945    | 0.01848      | 4.547e-05   | 0.9942    | 0               | 3.211        | 3.192      
63  | 4  | 0      | 0.9912    | 0.04082      | 0.01136     | 0.9831    | 0               | 2.418        | 2.378      
63  | 5  | 0      | 0.9961    | 0.07634      | 0.0009306   | 0.9713    | 0               | 2.663        | 2.587      
63  | 6  | 0      | 0.9956    | 0.1335       | 0.000105    | 0.9426    | 0               | 2.326        | 2.193      
63  | 7  | 0      | 0.9954    | 0.04775      | 1.785e-05   | 0.9799    | 0               | 2.371        | 2.324      
63  | 8  | 0      | 0.9861    | 0.2076       | 0.01123     | 0.9126    | 0               | 2.376        | 2.169      
63  | 9  | 0      | 0.9983    | 0.04625      | 0           | 0.9786    | 0               | 2.163        | 2.117      
63  | 10 | 0      | 0.9982    | 0.1101       | 0           | 0.951     | 0               | 2.248        | 2.138      
150 | 3  | 0      | 0.8276    | -3.577       | 0.1064      | 1.7       | 1               | 5.11         | 8.687      
150 | 4  | 0      | 0.7625    | -2.894       | 0.1185      | 1.634     | 0               | 4.563        | 7.457      
150 | 5  | 0      | 0.6531    | -1.804       | 0.09426     | 1.417     | 0               | 4.329        | 6.133      
150 | 6  | 0      | 0.6678    | 12.65        | 0.1917      | 0.2891    | 0               | 17.79        | 5.144      
150 | 7  | 0      | -0.158    | 18.81        | 0.2199      | 0.1873    | 63              | 23.14        | 4.333      
150 | 8  | 0      | 0.3477    | 8.113        | 0.1393      | 0.371     | 63              | 12.9         | 4.785      
150 | 9  | 0      | 0.6697    | 2.008        | 0.09604     | 0.6941    | 88.39           | 6.563        | 4.556      
150 | 10 | 0      | 0.9887    | 0.606        | 0.01022     | 0.9144    | 0               | 7.081        | 6.475      
270 | 3  | 0      | 0.306     | 4.345        | 0.1663      | 0.8931    | 86.98           | 40.66        | 36.32      
270 | 4  | 0      | 0.1366    | -32.8        | 0.16        | 2.528     | 58.01           | 21.47        | 54.27      
270 | 5  | 0      | 0.2558    | -13.4        | 0.1655      | 2.531     | 62              | 8.748        | 22.14      
270 | 6  | 0      | 0.5062    | -4.813       | 0.135       | 1.346     | 58.08           | 13.89        | 18.7       
270 | 7  | 0      | 0.5017    | -9.066       | 0.1176      | 2.074     | 88.39           | 8.439        | 17.5       
270 | 8  | 0      | 0.7501    | -3.448       | 0.08392     | 1.233     | 62              | 14.79        | 18.24      
270 | 9  | 0      | 0.3168    | 25.46        | 0.1335      | 0.4992    | 61              | 50.84        | 25.38      
270 | 10 | 0      | -0.09017  | -6.398       | 0.1746      | 1.408     | 60.3            | 15.68        | 22.07      
450 | 3  | 0      | 0.9511    | 0.2769       | 0.02908     | 0.9746    | 0               | 10.92        | 10.65      
450 | 4  | 0      | 0.9627    | -2.649       | 0.01542     | 1.243     | 0               | 10.92        | 13.57      
450 | 5  | 0      | 0.9002    | -2.261       | 0.03529     | 1.278     | 76.03           | 8.147        | 10.41      
450 | 6  | 0      | 0.9757    | -0.9693      | 0.01311     | 1.097     | 0               | 9.95         | 10.92      
450 | 7  | 0      | 0.6067    | -4.393       | 0.09227     | 1.618     | 29              | 7.105        | 11.5       
450 | 8  | 0      | 0.7585    | -3.223       | 0.06862     | 1.449     | 29              | 7.173        | 10.4       
450 | 9  | 0      | 0.5144    | -2.133       | 0.1324      | 1.318     | 34              | 6.705        | 8.838      
450 | 10 | 0      | 0.8357    | 8.515        | 0.05065     | 0.5591    | 0               | 19.31        | 10.8       
500 | 3  | 0      | 0.9       | -0.3321      | 0.05388     | 1.02      | 70.21           | 16.38        | 16.71      
500 | 4  | 0      | 0.326     | -9.019       | 0.1483      | 1.86      | 63              | 10.49        | 19.51      
500 | 5  | 0      | 0.9307    | 2.511        | 0.04252     | 0.8647    | 31              | 18.56        | 16.05      
500 | 6  | 0      | 0.9673    | 1.722        | 0.03596     | 0.9       | 0               | 17.23        | 15.51      
500 | 7  | 0      | 0.8982    | 2.796        | 0.06469     | 0.8396    | 0               | 17.43        | 14.64      
500 | 8  | 0      | 0.9344    | 1.626        | 0.04631     | 0.8948    | 0               | 15.46        | 13.83      
500 | 9  | 0      | 0.8102    | 9.768        | 0.07629     | 0.5681    | 71.12           | 22.62        | 12.85      
500 | 10 | 0      | 0.9857    | 0.2476       | 0.004105    | 0.981     | 0               | 13           | 12.75      
```

## Mean by MHz (primary)

```
mhz | n | pearson_r_mean | max_diff_ohm_mean | pattern_mae_mean | max_ratio_mean | peak_loc_err_px_mean | real_max_ohm_mean | gen_max_ohm_mean
----|---|----------------|-------------------|------------------|----------------|----------------------|-------------------|-----------------
10  | 8 | 0.9815         | 0.008399          | 0.007265         | 0.9754         | 7.875                | 0.3404            | 0.332           
63  | 8 | 0.9944         | 0.08511           | 0.002961         | 0.9642         | 0                    | 2.472             | 2.387           
150 | 8 | 0.5949         | 4.239             | 0.122            | 0.9009         | 26.92                | 10.19             | 5.946           
270 | 8 | 0.3354         | -5.014            | 0.1421           | 1.564          | 67.09                | 21.81             | 26.83           
450 | 8 | 0.8131         | -0.8546           | 0.0546           | 1.192          | 21                   | 10.03             | 10.88           
500 | 8 | 0.8441         | 1.165             | 0.059            | 0.991          | 29.42                | 16.39             | 15.23           
```

## Agent copy block

```
sim_compare_metrics: n=48
  10.0MHz K3 s0: r=0.987 Δmax=+0.00987Ω pattern_mae=0.004 max_ratio=0.972 real_max=0.356 gen_max=0.346 peak_err=0.0px
  10.0MHz K4 s0: r=0.989 Δmax=+0.0131Ω pattern_mae=0.011 max_ratio=0.961 real_max=0.335 gen_max=0.322 peak_err=0.0px
  10.0MHz K5 s0: r=0.993 Δmax=+0.0105Ω pattern_mae=0.003 max_ratio=0.970 real_max=0.355 gen_max=0.345 peak_err=0.0px
  10.0MHz K6 s0: r=0.995 Δmax=+0.0175Ω pattern_mae=0.004 max_ratio=0.948 real_max=0.335 gen_max=0.318 peak_err=0.0px
  10.0MHz K7 s0: r=0.969 Δmax=+0.00407Ω pattern_mae=0.011 max_ratio=0.988 real_max=0.342 gen_max=0.338 peak_err=0.0px
  10.0MHz K8 s0: r=0.988 Δmax=+0.00804Ω pattern_mae=0.001 max_ratio=0.977 real_max=0.349 gen_max=0.341 peak_err=0.0px
  10.0MHz K9 s0: r=0.988 Δmax=+0.00462Ω pattern_mae=0.012 max_ratio=0.985 real_max=0.316 gen_max=0.312 peak_err=63.0px
  10.0MHz K10 s0: r=0.944 Δmax=-0.000536Ω pattern_mae=0.012 max_ratio=1.002 real_max=0.334 gen_max=0.334 peak_err=0.0px
  63.0MHz K3 s0: r=0.994 Δmax=+0.0185Ω pattern_mae=0.000 max_ratio=0.994 real_max=3.21 gen_max=3.19 peak_err=0.0px
  63.0MHz K4 s0: r=0.991 Δmax=+0.0408Ω pattern_mae=0.011 max_ratio=0.983 real_max=2.42 gen_max=2.38 peak_err=0.0px
  63.0MHz K5 s0: r=0.996 Δmax=+0.0763Ω pattern_mae=0.001 max_ratio=0.971 real_max=2.66 gen_max=2.59 peak_err=0.0px
  63.0MHz K6 s0: r=0.996 Δmax=+0.133Ω pattern_mae=0.000 max_ratio=0.943 real_max=2.33 gen_max=2.19 peak_err=0.0px
  63.0MHz K7 s0: r=0.995 Δmax=+0.0477Ω pattern_mae=0.000 max_ratio=0.980 real_max=2.37 gen_max=2.32 peak_err=0.0px
  63.0MHz K8 s0: r=0.986 Δmax=+0.208Ω pattern_mae=0.011 max_ratio=0.913 real_max=2.38 gen_max=2.17 peak_err=0.0px
  63.0MHz K9 s0: r=0.998 Δmax=+0.0462Ω pattern_mae=0.000 max_ratio=0.979 real_max=2.16 gen_max=2.12 peak_err=0.0px
  63.0MHz K10 s0: r=0.998 Δmax=+0.11Ω pattern_mae=0.000 max_ratio=0.951 real_max=2.25 gen_max=2.14 peak_err=0.0px
  150.0MHz K3 s0: r=0.828 Δmax=-3.58Ω pattern_mae=0.106 max_ratio=1.700 real_max=5.11 gen_max=8.69 peak_err=1.0px
  150.0MHz K4 s0: r=0.762 Δmax=-2.89Ω pattern_mae=0.118 max_ratio=1.634 real_max=4.56 gen_max=7.46 peak_err=0.0px
  150.0MHz K5 s0: r=0.653 Δmax=-1.8Ω pattern_mae=0.094 max_ratio=1.417 real_max=4.33 gen_max=6.13 peak_err=0.0px
  150.0MHz K6 s0: r=0.668 Δmax=+12.7Ω pattern_mae=0.192 max_ratio=0.289 real_max=17.8 gen_max=5.14 peak_err=0.0px
  150.0MHz K7 s0: r=-0.158 Δmax=+18.8Ω pattern_mae=0.220 max_ratio=0.187 real_max=23.1 gen_max=4.33 peak_err=63.0px
  150.0MHz K8 s0: r=0.348 Δmax=+8.11Ω pattern_mae=0.139 max_ratio=0.371 real_max=12.9 gen_max=4.79 peak_err=63.0px
  150.0MHz K9 s0: r=0.670 Δmax=+2.01Ω pattern_mae=0.096 max_ratio=0.694 real_max=6.56 gen_max=4.56 peak_err=88.4px
  150.0MHz K10 s0: r=0.989 Δmax=+0.606Ω pattern_mae=0.010 max_ratio=0.914 real_max=7.08 gen_max=6.47 peak_err=0.0px
  270.0MHz K3 s0: r=0.306 Δmax=+4.35Ω pattern_mae=0.166 max_ratio=0.893 real_max=40.7 gen_max=36.3 peak_err=87.0px
  270.0MHz K4 s0: r=0.137 Δmax=-32.8Ω pattern_mae=0.160 max_ratio=2.528 real_max=21.5 gen_max=54.3 peak_err=58.0px
  270.0MHz K5 s0: r=0.256 Δmax=-13.4Ω pattern_mae=0.166 max_ratio=2.531 real_max=8.75 gen_max=22.1 peak_err=62.0px
  270.0MHz K6 s0: r=0.506 Δmax=-4.81Ω pattern_mae=0.135 max_ratio=1.346 real_max=13.9 gen_max=18.7 peak_err=58.1px
  270.0MHz K7 s0: r=0.502 Δmax=-9.07Ω pattern_mae=0.118 max_ratio=2.074 real_max=8.44 gen_max=17.5 peak_err=88.4px
  270.0MHz K8 s0: r=0.750 Δmax=-3.45Ω pattern_mae=0.084 max_ratio=1.233 real_max=14.8 gen_max=18.2 peak_err=62.0px
  270.0MHz K9 s0: r=0.317 Δmax=+25.5Ω pattern_mae=0.133 max_ratio=0.499 real_max=50.8 gen_max=25.4 peak_err=61.0px
  270.0MHz K10 s0: r=-0.090 Δmax=-6.4Ω pattern_mae=0.175 max_ratio=1.408 real_max=15.7 gen_max=22.1 peak_err=60.3px
  450.0MHz K3 s0: r=0.951 Δmax=+0.277Ω pattern_mae=0.029 max_ratio=0.975 real_max=10.9 gen_max=10.6 peak_err=0.0px
  450.0MHz K4 s0: r=0.963 Δmax=-2.65Ω pattern_mae=0.015 max_ratio=1.243 real_max=10.9 gen_max=13.6 peak_err=0.0px
  450.0MHz K5 s0: r=0.900 Δmax=-2.26Ω pattern_mae=0.035 max_ratio=1.278 real_max=8.15 gen_max=10.4 peak_err=76.0px
  450.0MHz K6 s0: r=0.976 Δmax=-0.969Ω pattern_mae=0.013 max_ratio=1.097 real_max=9.95 gen_max=10.9 peak_err=0.0px
  450.0MHz K7 s0: r=0.607 Δmax=-4.39Ω pattern_mae=0.092 max_ratio=1.618 real_max=7.1 gen_max=11.5 peak_err=29.0px
  450.0MHz K8 s0: r=0.759 Δmax=-3.22Ω pattern_mae=0.069 max_ratio=1.449 real_max=7.17 gen_max=10.4 peak_err=29.0px
  450.0MHz K9 s0: r=0.514 Δmax=-2.13Ω pattern_mae=0.132 max_ratio=1.318 real_max=6.71 gen_max=8.84 peak_err=34.0px
  450.0MHz K10 s0: r=0.836 Δmax=+8.52Ω pattern_mae=0.051 max_ratio=0.559 real_max=19.3 gen_max=10.8 peak_err=0.0px
  500.0MHz K3 s0: r=0.900 Δmax=-0.332Ω pattern_mae=0.054 max_ratio=1.020 real_max=16.4 gen_max=16.7 peak_err=70.2px
  500.0MHz K4 s0: r=0.326 Δmax=-9.02Ω pattern_mae=0.148 max_ratio=1.860 real_max=10.5 gen_max=19.5 peak_err=63.0px
  500.0MHz K5 s0: r=0.931 Δmax=+2.51Ω pattern_mae=0.043 max_ratio=0.865 real_max=18.6 gen_max=16 peak_err=31.0px
  500.0MHz K6 s0: r=0.967 Δmax=+1.72Ω pattern_mae=0.036 max_ratio=0.900 real_max=17.2 gen_max=15.5 peak_err=0.0px
  500.0MHz K7 s0: r=0.898 Δmax=+2.8Ω pattern_mae=0.065 max_ratio=0.840 real_max=17.4 gen_max=14.6 peak_err=0.0px
  500.0MHz K8 s0: r=0.934 Δmax=+1.63Ω pattern_mae=0.046 max_ratio=0.895 real_max=15.5 gen_max=13.8 peak_err=0.0px
  500.0MHz K9 s0: r=0.810 Δmax=+9.77Ω pattern_mae=0.076 max_ratio=0.568 real_max=22.6 gen_max=12.8 peak_err=71.1px
  500.0MHz K10 s0: r=0.986 Δmax=+0.248Ω pattern_mae=0.004 max_ratio=0.981 real_max=13 gen_max=12.8 peak_err=0.0px
```
