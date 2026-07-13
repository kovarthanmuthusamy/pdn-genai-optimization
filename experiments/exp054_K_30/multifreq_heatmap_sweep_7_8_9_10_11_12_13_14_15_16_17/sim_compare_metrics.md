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
10  | 7  | 0      | 0.9717    | 0.014       | 0.8575    | 0               | 0.3424       | 0.2936     
10  | 8  | 0      | 0.9775    | 0.008095    | 0.8782    | 0               | 0.3489       | 0.3064     
10  | 9  | 0      | 0.9809    | 0.0129      | 0.8963    | 63              | 0.3211       | 0.2878     
10  | 10 | 0      | 0.9741    | 0.007701    | 0.8885    | 0               | 0.3356       | 0.2982     
10  | 11 | 0      | 0.9451    | 0.01927     | 0.9034    | 0               | 0.3154       | 0.2849     
10  | 12 | 0      | 0.9759    | 0.01032     | 0.909     | 0               | 0.3179       | 0.289      
10  | 13 | 0      | 0.9736    | 0.02059     | 0.852     | 0               | 0.328        | 0.2795     
10  | 14 | 0      | 0.9063    | 0.02666     | 0.9623    | 0               | 0.3126       | 0.3008     
10  | 15 | 0      | 0.9801    | 0.005612    | 0.9335    | 63              | 0.3258       | 0.3041     
10  | 16 | 0      | 0.9359    | 0.01643     | 0.8098    | 63              | 0.3302       | 0.2674     
10  | 17 | 0      | 0.9866    | 0.005163    | 0.9267    | 0               | 0.3166       | 0.2934     
70  | 7  | 0      | 0.9899    | 0.006934    | 0.8073    | 0               | 2.669        | 2.154      
70  | 8  | 0      | 0.9971    | 0.0005594   | 0.8318    | 0               | 2.674        | 2.224      
70  | 9  | 0      | 0.989     | 0.01242     | 0.8107    | 0               | 2.425        | 1.966      
70  | 10 | 0      | 0.9839    | 0.008312    | 0.7998    | 0               | 2.509        | 2.007      
70  | 11 | 0      | 0.9884    | 0.007791    | 0.8215    | 63              | 2.415        | 1.984      
70  | 12 | 0      | 0.9666    | 0.01987     | 0.8653    | 0               | 2.418        | 2.093      
70  | 13 | 0      | 0.9569    | 0.02139     | 0.8331    | 0               | 2.471        | 2.058      
70  | 14 | 0      | 0.9703    | 0.005713    | 0.822     | 0               | 2.375        | 1.952      
70  | 15 | 0      | 0.9914    | 0.001991    | 0.8377    | 63              | 2.398        | 2.009      
70  | 16 | 0      | 0.9651    | 0.009976    | 0.7371    | 0               | 2.426        | 1.789      
70  | 17 | 0      | 0.9877    | 0.01105     | 0.7658    | 0               | 2.369        | 1.814      
120 | 7  | 0      | 0.9895    | 0.0052      | 0.8891    | 0               | 5.768        | 5.128      
120 | 8  | 0      | 0.9978    | 0.0001198   | 0.9864    | 0               | 5.578        | 5.502      
120 | 9  | 0      | 0.9944    | 0.00117     | 1.036     | 0               | 4.521        | 4.685      
120 | 10 | 0      | 0.9471    | 0.02092     | 1.152     | 0               | 4.77         | 5.496      
120 | 11 | 0      | 0.9863    | 0.003543    | 1.068     | 0               | 4.48         | 4.784      
120 | 12 | 0      | 0.9299    | 0.02916     | 1.292     | 0               | 4.498        | 5.81       
120 | 13 | 0      | 0.9478    | 0.02099     | 1.268     | 0               | 4.659        | 5.906      
120 | 14 | 0      | 0.9498    | 0.01378     | 1.084     | 0               | 4.337        | 4.701      
120 | 15 | 0      | 0.9701    | 0.01396     | 1.114     | 63              | 4.402        | 4.904      
120 | 16 | 0      | 0.9721    | 0.008275    | 0.9394    | 0               | 4.539        | 4.264      
120 | 17 | 0      | 0.9871    | 0.003335    | 0.9845    | 0               | 4.306        | 4.239      
270 | 7  | 0      | 0.6956    | 0.09886     | 2.516     | 88.39           | 8.439        | 21.23      
270 | 8  | 0      | 0.8533    | 0.06768     | 1.584     | 62              | 14.79        | 23.42      
270 | 9  | 0      | 0.6889    | 0.1555      | 0.5099    | 4.123           | 19.34        | 9.859      
270 | 10 | 0      | 0.2057    | 0.1252      | 1.912     | 88.39           | 15.11        | 28.89      
270 | 11 | 0      | 0.5293    | 0.1207      | 2.783     | 88.39           | 7.785        | 21.66      
270 | 12 | 0      | 0.9663    | 0.0773      | 1.572     | 0               | 12.05        | 18.94      
270 | 13 | 0      | 0.6029    | 0.1159      | 2.643     | 57.14           | 7.77         | 20.54      
270 | 14 | 0      | 0.9302    | 0.08935     | 1.389     | 63              | 17.6         | 24.46      
270 | 15 | 0      | 0.6716    | 0.1004      | 1.757     | 58.08           | 14.38        | 25.27      
270 | 16 | 0      | 0.8109    | 0.08115     | 1.451     | 58.03           | 16           | 23.22      
270 | 17 | 0      | 0.8367    | 0.07755     | 2.011     | 5.831           | 12.89        | 25.93      
400 | 7  | 0      | 0.9671    | 0.01067     | 1.127     | 0               | 9.513        | 10.72      
400 | 8  | 0      | 0.9701    | 0.01789     | 0.9906    | 63              | 9.844        | 9.751      
400 | 9  | 0      | 0.963     | 0.007705    | 1.291     | 0               | 9.715        | 12.55      
400 | 10 | 0      | 0.9676    | 0.008038    | 1.294     | 0               | 9.433        | 12.21      
400 | 11 | 0      | 0.9663    | 0.007649    | 1.266     | 0               | 8.575        | 10.86      
400 | 12 | 0      | 0.9512    | 0.0363      | 1.263     | 0               | 9.668        | 12.21      
400 | 13 | 0      | 0.5427    | 0.1056      | 0.6241    | 44              | 14.94        | 9.327      
400 | 14 | 0      | 0.7127    | 0.07861     | 1.222     | 63              | 7.396        | 9.04       
400 | 15 | 0      | 0.5102    | 0.1511      | 0.3536    | 76.03           | 25.66        | 9.074      
400 | 16 | 0      | 0.8053    | 0.0684      | 0.5429    | 0               | 21.65        | 11.75      
400 | 17 | 0      | 0.9123    | 0.01711     | 1.232     | 0               | 8.355        | 10.3       
```

## Mean by MHz (primary)

```
mhz | n  | pearson_r_mean | pattern_mae_mean | max_ratio_mean | peak_loc_err_px_mean | real_max_ohm_mean | gen_max_ohm_mean
----|----|----------------|------------------|----------------|----------------------|-------------------|-----------------
10  | 11 | 0.9643         | 0.01334          | 0.8925         | 17.18                | 0.3268            | 0.2914          
70  | 11 | 0.9806         | 0.009636         | 0.812          | 11.45                | 2.468             | 2.005           
120 | 11 | 0.9702         | 0.01095          | 1.074          | 5.727                | 4.714             | 5.038           
270 | 11 | 0.7083         | 0.1009           | 1.83           | 52.13                | 13.29             | 22.13           
400 | 11 | 0.8426         | 0.04628          | 1.019          | 22.37                | 12.25             | 10.71           
```

## Agent copy block

```
sim_compare_metrics: n=55
  10.0MHz K7 s0: r=0.972 pattern_mae=0.014 max_ratio=0.858 real_max=0.342 gen_max=0.294 peak_err=0.0px
  10.0MHz K8 s0: r=0.978 pattern_mae=0.008 max_ratio=0.878 real_max=0.349 gen_max=0.306 peak_err=0.0px
  10.0MHz K9 s0: r=0.981 pattern_mae=0.013 max_ratio=0.896 real_max=0.321 gen_max=0.288 peak_err=63.0px
  10.0MHz K10 s0: r=0.974 pattern_mae=0.008 max_ratio=0.889 real_max=0.336 gen_max=0.298 peak_err=0.0px
  10.0MHz K11 s0: r=0.945 pattern_mae=0.019 max_ratio=0.903 real_max=0.315 gen_max=0.285 peak_err=0.0px
  10.0MHz K12 s0: r=0.976 pattern_mae=0.010 max_ratio=0.909 real_max=0.318 gen_max=0.289 peak_err=0.0px
  10.0MHz K13 s0: r=0.974 pattern_mae=0.021 max_ratio=0.852 real_max=0.328 gen_max=0.279 peak_err=0.0px
  10.0MHz K14 s0: r=0.906 pattern_mae=0.027 max_ratio=0.962 real_max=0.313 gen_max=0.301 peak_err=0.0px
  10.0MHz K15 s0: r=0.980 pattern_mae=0.006 max_ratio=0.934 real_max=0.326 gen_max=0.304 peak_err=63.0px
  10.0MHz K16 s0: r=0.936 pattern_mae=0.016 max_ratio=0.810 real_max=0.33 gen_max=0.267 peak_err=63.0px
  10.0MHz K17 s0: r=0.987 pattern_mae=0.005 max_ratio=0.927 real_max=0.317 gen_max=0.293 peak_err=0.0px
  70.0MHz K7 s0: r=0.990 pattern_mae=0.007 max_ratio=0.807 real_max=2.67 gen_max=2.15 peak_err=0.0px
  70.0MHz K8 s0: r=0.997 pattern_mae=0.001 max_ratio=0.832 real_max=2.67 gen_max=2.22 peak_err=0.0px
  70.0MHz K9 s0: r=0.989 pattern_mae=0.012 max_ratio=0.811 real_max=2.42 gen_max=1.97 peak_err=0.0px
  70.0MHz K10 s0: r=0.984 pattern_mae=0.008 max_ratio=0.800 real_max=2.51 gen_max=2.01 peak_err=0.0px
  70.0MHz K11 s0: r=0.988 pattern_mae=0.008 max_ratio=0.821 real_max=2.42 gen_max=1.98 peak_err=63.0px
  70.0MHz K12 s0: r=0.967 pattern_mae=0.020 max_ratio=0.865 real_max=2.42 gen_max=2.09 peak_err=0.0px
  70.0MHz K13 s0: r=0.957 pattern_mae=0.021 max_ratio=0.833 real_max=2.47 gen_max=2.06 peak_err=0.0px
  70.0MHz K14 s0: r=0.970 pattern_mae=0.006 max_ratio=0.822 real_max=2.38 gen_max=1.95 peak_err=0.0px
  70.0MHz K15 s0: r=0.991 pattern_mae=0.002 max_ratio=0.838 real_max=2.4 gen_max=2.01 peak_err=63.0px
  70.0MHz K16 s0: r=0.965 pattern_mae=0.010 max_ratio=0.737 real_max=2.43 gen_max=1.79 peak_err=0.0px
  70.0MHz K17 s0: r=0.988 pattern_mae=0.011 max_ratio=0.766 real_max=2.37 gen_max=1.81 peak_err=0.0px
  120.0MHz K7 s0: r=0.989 pattern_mae=0.005 max_ratio=0.889 real_max=5.77 gen_max=5.13 peak_err=0.0px
  120.0MHz K8 s0: r=0.998 pattern_mae=0.000 max_ratio=0.986 real_max=5.58 gen_max=5.5 peak_err=0.0px
  120.0MHz K9 s0: r=0.994 pattern_mae=0.001 max_ratio=1.036 real_max=4.52 gen_max=4.68 peak_err=0.0px
  120.0MHz K10 s0: r=0.947 pattern_mae=0.021 max_ratio=1.152 real_max=4.77 gen_max=5.5 peak_err=0.0px
  120.0MHz K11 s0: r=0.986 pattern_mae=0.004 max_ratio=1.068 real_max=4.48 gen_max=4.78 peak_err=0.0px
  120.0MHz K12 s0: r=0.930 pattern_mae=0.029 max_ratio=1.292 real_max=4.5 gen_max=5.81 peak_err=0.0px
  120.0MHz K13 s0: r=0.948 pattern_mae=0.021 max_ratio=1.268 real_max=4.66 gen_max=5.91 peak_err=0.0px
  120.0MHz K14 s0: r=0.950 pattern_mae=0.014 max_ratio=1.084 real_max=4.34 gen_max=4.7 peak_err=0.0px
  120.0MHz K15 s0: r=0.970 pattern_mae=0.014 max_ratio=1.114 real_max=4.4 gen_max=4.9 peak_err=63.0px
  120.0MHz K16 s0: r=0.972 pattern_mae=0.008 max_ratio=0.939 real_max=4.54 gen_max=4.26 peak_err=0.0px
  120.0MHz K17 s0: r=0.987 pattern_mae=0.003 max_ratio=0.984 real_max=4.31 gen_max=4.24 peak_err=0.0px
  270.0MHz K7 s0: r=0.696 pattern_mae=0.099 max_ratio=2.516 real_max=8.44 gen_max=21.2 peak_err=88.4px
  270.0MHz K8 s0: r=0.853 pattern_mae=0.068 max_ratio=1.584 real_max=14.8 gen_max=23.4 peak_err=62.0px
  270.0MHz K9 s0: r=0.689 pattern_mae=0.156 max_ratio=0.510 real_max=19.3 gen_max=9.86 peak_err=4.1px
  270.0MHz K10 s0: r=0.206 pattern_mae=0.125 max_ratio=1.912 real_max=15.1 gen_max=28.9 peak_err=88.4px
  270.0MHz K11 s0: r=0.529 pattern_mae=0.121 max_ratio=2.783 real_max=7.79 gen_max=21.7 peak_err=88.4px
  270.0MHz K12 s0: r=0.966 pattern_mae=0.077 max_ratio=1.572 real_max=12 gen_max=18.9 peak_err=0.0px
  270.0MHz K13 s0: r=0.603 pattern_mae=0.116 max_ratio=2.643 real_max=7.77 gen_max=20.5 peak_err=57.1px
  270.0MHz K14 s0: r=0.930 pattern_mae=0.089 max_ratio=1.389 real_max=17.6 gen_max=24.5 peak_err=63.0px
  270.0MHz K15 s0: r=0.672 pattern_mae=0.100 max_ratio=1.757 real_max=14.4 gen_max=25.3 peak_err=58.1px
  270.0MHz K16 s0: r=0.811 pattern_mae=0.081 max_ratio=1.451 real_max=16 gen_max=23.2 peak_err=58.0px
  270.0MHz K17 s0: r=0.837 pattern_mae=0.078 max_ratio=2.011 real_max=12.9 gen_max=25.9 peak_err=5.8px
  400.0MHz K7 s0: r=0.967 pattern_mae=0.011 max_ratio=1.127 real_max=9.51 gen_max=10.7 peak_err=0.0px
  400.0MHz K8 s0: r=0.970 pattern_mae=0.018 max_ratio=0.991 real_max=9.84 gen_max=9.75 peak_err=63.0px
  400.0MHz K9 s0: r=0.963 pattern_mae=0.008 max_ratio=1.291 real_max=9.72 gen_max=12.5 peak_err=0.0px
  400.0MHz K10 s0: r=0.968 pattern_mae=0.008 max_ratio=1.294 real_max=9.43 gen_max=12.2 peak_err=0.0px
  400.0MHz K11 s0: r=0.966 pattern_mae=0.008 max_ratio=1.266 real_max=8.57 gen_max=10.9 peak_err=0.0px
  400.0MHz K12 s0: r=0.951 pattern_mae=0.036 max_ratio=1.263 real_max=9.67 gen_max=12.2 peak_err=0.0px
  400.0MHz K13 s0: r=0.543 pattern_mae=0.106 max_ratio=0.624 real_max=14.9 gen_max=9.33 peak_err=44.0px
  400.0MHz K14 s0: r=0.713 pattern_mae=0.079 max_ratio=1.222 real_max=7.4 gen_max=9.04 peak_err=63.0px
  400.0MHz K15 s0: r=0.510 pattern_mae=0.151 max_ratio=0.354 real_max=25.7 gen_max=9.07 peak_err=76.0px
  400.0MHz K16 s0: r=0.805 pattern_mae=0.068 max_ratio=0.543 real_max=21.6 gen_max=11.8 peak_err=0.0px
  400.0MHz K17 s0: r=0.912 pattern_mae=0.017 max_ratio=1.232 real_max=8.36 gen_max=10.3 peak_err=0.0px
```
