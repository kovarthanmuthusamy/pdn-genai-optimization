# Simulated Real vs Generated — Heatmap Metrics

- **Rows**: 114 (one per freq × K × sample)
- **Source**: ECADStar `.map` in `K*/Real/` vs `data_sample_*/heatmap_physical.npy`
- **CSV**: `sim_compare_metrics.csv`
- **JSON**: `sim_compare_metrics.json`

**Primary QC** (tables below): `pearson_r`, `max_diff_ohm` (real−gen; − = gen higher), `pattern_mae`, `max_ratio`.
Use `mae_ohm` / `mape_pct` from CSV with care — often misleading at 10 MHz (FG Ω ≈ 0).

## Per-sample metrics (primary)

```
mhz | k  | sample | pearson_r | max_diff_ohm | pattern_mae | max_ratio | peak_loc_err_px | real_max_ohm | gen_max_ohm
----|----|--------|-----------|--------------|-------------|-----------|-----------------|--------------|------------
10  | 2  | 0      | 0.9849    | 0.01508      | 0.002479    | 0.9556    | 0               | 0.3396       | 0.3245     
10  | 3  | 0      | 0.9781    | 0.02843      | 0.01505     | 0.9201    | 0               | 0.3556       | 0.3272     
10  | 4  | 0      | 0.9851    | 0.01692      | 0.00409     | 0.9495    | 0               | 0.3352       | 0.3183     
10  | 5  | 0      | 0.9929    | 0.02989      | 0.00229     | 0.9159    | 0               | 0.3554       | 0.3255     
10  | 6  | 0      | 0.9952    | 0.02032      | 0.0007879   | 0.9393    | 0               | 0.3351       | 0.3147     
10  | 7  | 0      | 0.9759    | 0.02271      | 0.007533    | 0.9337    | 0               | 0.3424       | 0.3197     
10  | 8  | 0      | 0.9877    | 0.02184      | 0.001807    | 0.9374    | 0               | 0.3489       | 0.3271     
10  | 9  | 0      | 0.9852    | 0.01225      | 0.008751    | 0.9614    | 63              | 0.3171       | 0.3048     
10  | 10 | 0      | 0.9863    | 0.007537     | 0.001684    | 0.9774    | 0               | 0.3329       | 0.3254     
10  | 11 | 0      | 0.9688    | 0.005197     | 0.005118    | 0.9834    | 0               | 0.3133       | 0.3081     
10  | 12 | 0      | 0.9813    | 0.0219       | 0.009695    | 0.935     | 0               | 0.3369       | 0.315      
10  | 13 | 0      | 0.988     | 0.02413      | 0.001427    | 0.9272    | 0               | 0.3314       | 0.3073     
10  | 14 | 0      | 0.9651    | 0.006386     | 0.01228     | 0.98      | 0               | 0.3186       | 0.3122     
10  | 15 | 0      | 0.972     | 0.01107      | 0.01628     | 0.966     | 63              | 0.3252       | 0.3142     
10  | 16 | 0      | 0.9862    | 0.01024      | 0.001247    | 0.9664    | 0               | 0.3049       | 0.2946     
10  | 17 | 0      | 0.9935    | 0.003661     | 0.0005127   | 0.9883    | 0               | 0.3135       | 0.3098     
10  | 18 | 0      | 0.9774    | -0.001261    | 0.002225    | 1.004     | 0               | 0.3077       | 0.3089     
10  | 19 | 0      | 0.9608    | -0.000905    | 0.006233    | 1.003     | 0               | 0.3123       | 0.3132     
10  | 20 | 0      | 0.9543    | 0.002281     | 0.009902    | 0.9922    | 63              | 0.2911       | 0.2889     
63  | 2  | 0      | 0.9672    | -0.3113      | 0.104       | 1.095     | 0               | 3.294        | 3.605      
63  | 3  | 0      | 0.9894    | 0.08486      | 0.04854     | 0.9736    | 0               | 3.211        | 3.126      
63  | 4  | 0      | 0.9909    | 0.07014      | 0.003119    | 0.971     | 0               | 2.418        | 2.348      
63  | 5  | 0      | 0.9938    | 0.2289       | 0.001777    | 0.914     | 0               | 2.663        | 2.434      
63  | 6  | 0      | 0.9954    | 0.08479      | 0.0007454   | 0.9635    | 0               | 2.325        | 2.241      
63  | 7  | 0      | 0.9927    | 0.1688       | 0.0001071   | 0.9288    | 0               | 2.371        | 2.203      
63  | 8  | 0      | 0.9909    | 0.1433       | 0.004165    | 0.9398    | 0               | 2.379        | 2.235      
63  | 9  | 0      | 0.9966    | 0.03035      | 0           | 0.986     | 0               | 2.163        | 2.133      
63  | 10 | 0      | 0.9982    | 0.09197      | 3.113e-05   | 0.9591    | 0               | 2.248        | 2.156      
63  | 11 | 0      | 0.9959    | 0.09613      | 0.0004398   | 0.9555    | 0               | 2.159        | 2.063      
63  | 12 | 0      | 0.9896    | 0.1199       | 0.001876    | 0.9461    | 0               | 2.226        | 2.106      
63  | 13 | 0      | 0.9916    | 0.1527       | 0.001317    | 0.9316    | 0               | 2.233        | 2.081      
63  | 14 | 0      | 0.9781    | 0.1029       | 0.004007    | 0.9527    | 0               | 2.176        | 2.073      
63  | 15 | 0      | 0.9944    | 0.04825      | 0.0007732   | 0.9777    | 0               | 2.161        | 2.113      
63  | 16 | 0      | 0.9883    | 0.1622       | 0.002266    | 0.9205    | 63              | 2.04         | 1.878      
63  | 17 | 0      | 0.9978    | 0.1406       | 0           | 0.9334    | 0               | 2.112        | 1.972      
63  | 18 | 0      | 0.9971    | 0.09643      | 1.489e-05   | 0.9538    | 0               | 2.087        | 1.99       
63  | 19 | 0      | 0.9925    | 0.04673      | 0.0002631   | 0.9777    | 0               | 2.097        | 2.05       
63  | 20 | 0      | 0.9915    | 0.03101      | 0.0005931   | 0.9838    | 63              | 1.91         | 1.879      
150 | 2  | 0      | 0.7637    | -4.704       | 0.1219      | 1.897     | 1               | 5.242        | 9.946      
150 | 3  | 0      | 0.8534    | -2.021       | 0.09485     | 1.396     | 0               | 5.102        | 7.123      
150 | 4  | 0      | 0.7186    | -3.413       | 0.1364      | 1.748     | 1               | 4.563        | 7.976      
150 | 5  | 0      | 0.615     | -2.397       | 0.1011      | 1.554     | 0               | 4.329        | 6.726      
150 | 6  | 0      | 0.6075    | 12.5         | 0.213       | 0.2943    | 63              | 17.71        | 5.213      
150 | 7  | 0      | 0.2125    | 18.44        | 0.1959      | 0.2033    | 63              | 23.14        | 4.705      
150 | 8  | 0      | 0.1743    | 7.806        | 0.1608      | 0.3796    | 63              | 12.58        | 4.776      
150 | 9  | 0      | 0.8624    | 1.975        | 0.06356     | 0.7004    | 88.39           | 6.593        | 4.618      
150 | 10 | 0      | 0.9452    | 2.351        | 0.0372      | 0.6679    | 63              | 7.081        | 4.729      
150 | 11 | 0      | 0.9944    | 0.04289      | 0.001967    | 0.9932    | 0               | 6.315        | 6.273      
150 | 12 | 0      | 0.9507    | 0.399        | 0.03193     | 0.9449    | 0               | 7.241        | 6.842      
150 | 13 | 0      | 0.9744    | -0.9028      | 0.01913     | 1.132     | 0               | 6.84         | 7.743      
150 | 14 | 0      | 0.9545    | -0.8988      | 0.017       | 1.145     | 0               | 6.197        | 7.096      
150 | 15 | 0      | 0.9913    | -0.307       | 0.001376    | 1.05      | 63              | 6.097        | 6.404      
150 | 16 | 0      | 0.9855    | -0.04913     | 0.00493     | 1.009     | 0               | 5.396        | 5.445      
150 | 17 | 0      | 0.9943    | -0.1494      | 0.0009158   | 1.026     | 0               | 5.75         | 5.9        
150 | 18 | 0      | 0.9934    | 0.01779      | 0.001546    | 0.9968    | 0               | 5.574        | 5.556      
150 | 19 | 0      | 0.9858    | -0.5871      | 0.009908    | 1.104     | 0               | 5.637        | 6.224      
150 | 20 | 0      | 0.9864    | -0.279       | 0.008359    | 1.058     | 63              | 4.787        | 5.066      
270 | 2  | 0      | 0.4418    | -13.6        | 0.1583      | 2.178     | 56.08           | 11.54        | 25.15      
270 | 3  | 0      | 0.599     | 16.14        | 0.133       | 0.603     | 62.03           | 40.66        | 24.52      
270 | 4  | 0      | 0.8998    | 5.965        | 0.05939     | 0.7222    | 5.385           | 21.47        | 15.51      
270 | 5  | 0      | 0.776     | -6.876       | 0.1109      | 1.782     | 62              | 8.789        | 15.66      
270 | 6  | 0      | 0.7045    | -5.923       | 0.1041      | 1.426     | 58.08           | 13.89        | 19.81      
270 | 7  | 0      | 0.6852    | -3.608       | 0.08898     | 1.428     | 88.39           | 8.439        | 12.05      
270 | 8  | 0      | 0.8585    | -2.255       | 0.07922     | 1.152     | 62              | 14.79        | 17.04      
270 | 9  | 0      | 0.7107    | 29.94        | 0.08631     | 0.3225    | 2               | 44.19        | 14.25      
270 | 10 | 0      | 0.8195    | 5.216        | 0.05032     | 0.6666    | 0               | 15.64        | 10.43      
270 | 11 | 0      | 0.8655    | 2.008        | 0.06629     | 0.8179    | 12.37           | 11.02        | 9.015      
270 | 12 | 0      | 0.9527    | 1.658        | 0.03074     | 0.8644    | 0               | 12.23        | 10.57      
270 | 13 | 0      | 0.9763    | 29.3         | 0.03806     | 0.3006    | 18              | 41.89        | 12.59      
270 | 14 | 0      | 0.6752    | 9.529        | 0.1004      | 0.6557    | 9.22            | 27.68        | 18.15      
270 | 15 | 0      | 0.8374    | -5.006       | 0.1469      | 1.615     | 58.14           | 8.135        | 13.14      
270 | 16 | 0      | 0.7798    | 3.628        | 0.04949     | 0.7895    | 63              | 17.24        | 13.61      
270 | 17 | 0      | 0.8198    | 7.779        | 0.07643     | 0.6492    | 85.56           | 22.17        | 14.4       
270 | 18 | 0      | 0.5125    | -5.495       | 0.1347      | 1.453     | 63              | 12.13        | 17.62      
270 | 19 | 0      | 0.9597    | 8.498        | 0.07887     | 0.7145    | 0               | 29.77        | 21.27      
270 | 20 | 0      | 0.8738    | -5.562       | 0.05538     | 1.488     | 88.39           | 11.39        | 16.95      
450 | 2  | 0      | 0.9832    | -1.951       | 0.004043    | 1.178     | 0               | 10.96        | 12.91      
450 | 3  | 0      | 0.968     | -1.735       | 0.01155     | 1.157     | 0               | 11.06        | 12.8       
450 | 4  | 0      | 0.9661    | -2.889       | 0.01428     | 1.264     | 0               | 10.92        | 13.81      
450 | 5  | 0      | 0.9313    | -2.058       | 0.02794     | 1.253     | 19              | 8.147        | 10.2       
450 | 6  | 0      | 0.9916    | -0.7637      | 0.001912    | 1.077     | 0               | 9.95         | 10.71      
450 | 7  | 0      | 0.6448    | -3.58        | 0.09669     | 1.504     | 29              | 7.105        | 10.69      
450 | 8  | 0      | 0.8912    | -2.008       | 0.03632     | 1.268     | 0               | 7.495        | 9.503      
450 | 9  | 0      | 0.6327    | -2.285       | 0.09312     | 1.329     | 70.66           | 6.937        | 9.221      
450 | 10 | 0      | 0.2235    | 10.63        | 0.1457      | 0.4497    | 0               | 19.31        | 8.685      
450 | 11 | 0      | 0.8488    | 7.168        | 0.02988     | 0.5854    | 0               | 17.29        | 10.12      
450 | 12 | 0      | 0.9312    | -0.3534      | 0.0267      | 1.049     | 1               | 7.265        | 7.619      
450 | 13 | 0      | 0.9596    | 0.09666      | 0.01746     | 0.991     | 0               | 10.78        | 10.68      
450 | 14 | 0      | 0.9449    | 1.448        | 0.03011     | 0.8837    | 63              | 12.45        | 11         
450 | 15 | 0      | 0.9533    | 4.258        | 0.01956     | 0.725     | 0               | 15.48        | 11.22      
450 | 16 | 0      | 0.8898    | -0.659       | 0.02735     | 1.052     | 0               | 12.55        | 13.21      
450 | 17 | 0      | 0.922     | 0.8688       | 0.02891     | 0.9303    | 62              | 12.47        | 11.6       
450 | 18 | 0      | 0.9034    | 5.391        | 0.03941     | 0.6633    | 33.62           | 16.01        | 10.62      
450 | 19 | 0      | 0.8408    | -0.6033      | 0.03574     | 1.059     | 0               | 10.24        | 10.85      
450 | 20 | 0      | 0.8172    | 8.505        | 0.04901     | 0.5788    | 63              | 20.19        | 11.69      
500 | 2  | 0      | 0.5595    | -5.472       | 0.1056      | 1.414     | 59.68           | 13.21        | 18.69      
500 | 3  | 0      | 0.8457    | -1.231       | 0.08367     | 1.075     | 70.21           | 16.38        | 17.61      
500 | 4  | 0      | 0.437     | -7.121       | 0.1245      | 1.679     | 63              | 10.49        | 17.61      
500 | 5  | 0      | 0.912     | 4.328        | 0.05649     | 0.7668    | 88.39           | 18.56        | 14.23      
500 | 6  | 0      | 0.9419    | 1.511        | 0.05434     | 0.9114    | 29              | 17.05        | 15.54      
500 | 7  | 0      | 0.9382    | 0.5494       | 0.06481     | 0.9613    | 0               | 14.21        | 13.66      
500 | 8  | 0      | 0.9167    | 1.861        | 0.05379     | 0.8796    | 0               | 15.46        | 13.6       
500 | 9  | 0      | 0.7921    | 13.63        | 0.07372     | 0.464     | 71.12           | 25.42        | 11.8       
500 | 10 | 0      | 0.9326    | 0.9021       | 0.04425     | 0.9306    | 70.21           | 13           | 12.1       
500 | 11 | 0      | 0.6688    | 1.796        | 0.07994     | 0.875     | 32              | 14.36        | 12.57      
500 | 12 | 0      | 0.8867    | 6.434        | 0.03979     | 0.6094    | 2               | 16.47        | 10.04      
500 | 13 | 0      | 0.5651    | -4.245       | 0.1022      | 1.352     | 63              | 12.07        | 16.31      
500 | 14 | 0      | 0.7027    | -0.6612      | 0.05966     | 1.056     | 0               | 11.88        | 12.54      
500 | 15 | 0      | 0.8119    | -10.33       | 0.07269     | 1.805     | 87.69           | 12.83        | 23.16      
500 | 16 | 0      | 0.7839    | 10.99        | 0.0812      | 0.523     | 1.414           | 23.03        | 12.05      
500 | 17 | 0      | 0.6845    | -1.231       | 0.08385     | 1.1       | 70.66           | 12.29        | 13.52      
500 | 18 | 0      | 0.9499    | -0.3151      | 0.02128     | 1.03      | 0               | 10.34        | 10.65      
500 | 19 | 0      | 0.739     | -1.817       | 0.0492      | 1.174     | 0               | 10.42        | 12.24      
500 | 20 | 0      | 0.8592    | -0.1288      | 0.04357     | 1.014     | 0               | 9.48         | 9.609      
```

## Mean by MHz (primary)

```
mhz | n  | pearson_r_mean | max_diff_ohm_mean | pattern_mae_mean | max_ratio_mean | peak_loc_err_px_mean | real_max_ohm_mean | gen_max_ohm_mean
----|----|----------------|-------------------|------------------|----------------|----------------------|-------------------|-----------------
10  | 19 | 0.9799         | 0.01356           | 0.005757         | 0.9598         | 9.947                | 0.3272            | 0.3137          
63  | 19 | 0.9911         | 0.08362           | 0.009161         | 0.9612         | 6.632                | 2.33              | 2.247           
150 | 19 | 0.8191         | 1.464             | 0.0643           | 1.016          | 24.65                | 7.694             | 6.229           
270 | 19 | 0.7762         | 3.754             | 0.08673          | 1.033          | 41.77                | 19.64             | 15.88           
450 | 19 | 0.8549         | 1.025             | 0.03872          | 0.9998         | 17.96                | 11.93             | 10.9            
500 | 19 | 0.7857         | 0.4971            | 0.06813          | 1.033          | 37.28                | 14.58             | 14.08           
```

## Agent copy block

```
sim_compare_metrics: n=114
  10.0MHz K2 s0: r=0.985 Δmax=+0.0151Ω pattern_mae=0.002 max_ratio=0.956 real_max=0.34 gen_max=0.325 peak_err=0.0px
  10.0MHz K3 s0: r=0.978 Δmax=+0.0284Ω pattern_mae=0.015 max_ratio=0.920 real_max=0.356 gen_max=0.327 peak_err=0.0px
  10.0MHz K4 s0: r=0.985 Δmax=+0.0169Ω pattern_mae=0.004 max_ratio=0.950 real_max=0.335 gen_max=0.318 peak_err=0.0px
  10.0MHz K5 s0: r=0.993 Δmax=+0.0299Ω pattern_mae=0.002 max_ratio=0.916 real_max=0.355 gen_max=0.325 peak_err=0.0px
  10.0MHz K6 s0: r=0.995 Δmax=+0.0203Ω pattern_mae=0.001 max_ratio=0.939 real_max=0.335 gen_max=0.315 peak_err=0.0px
  10.0MHz K7 s0: r=0.976 Δmax=+0.0227Ω pattern_mae=0.008 max_ratio=0.934 real_max=0.342 gen_max=0.32 peak_err=0.0px
  10.0MHz K8 s0: r=0.988 Δmax=+0.0218Ω pattern_mae=0.002 max_ratio=0.937 real_max=0.349 gen_max=0.327 peak_err=0.0px
  10.0MHz K9 s0: r=0.985 Δmax=+0.0123Ω pattern_mae=0.009 max_ratio=0.961 real_max=0.317 gen_max=0.305 peak_err=63.0px
  10.0MHz K10 s0: r=0.986 Δmax=+0.00754Ω pattern_mae=0.002 max_ratio=0.977 real_max=0.333 gen_max=0.325 peak_err=0.0px
  10.0MHz K11 s0: r=0.969 Δmax=+0.0052Ω pattern_mae=0.005 max_ratio=0.983 real_max=0.313 gen_max=0.308 peak_err=0.0px
  10.0MHz K12 s0: r=0.981 Δmax=+0.0219Ω pattern_mae=0.010 max_ratio=0.935 real_max=0.337 gen_max=0.315 peak_err=0.0px
  10.0MHz K13 s0: r=0.988 Δmax=+0.0241Ω pattern_mae=0.001 max_ratio=0.927 real_max=0.331 gen_max=0.307 peak_err=0.0px
  10.0MHz K14 s0: r=0.965 Δmax=+0.00639Ω pattern_mae=0.012 max_ratio=0.980 real_max=0.319 gen_max=0.312 peak_err=0.0px
  10.0MHz K15 s0: r=0.972 Δmax=+0.0111Ω pattern_mae=0.016 max_ratio=0.966 real_max=0.325 gen_max=0.314 peak_err=63.0px
  10.0MHz K16 s0: r=0.986 Δmax=+0.0102Ω pattern_mae=0.001 max_ratio=0.966 real_max=0.305 gen_max=0.295 peak_err=0.0px
  10.0MHz K17 s0: r=0.994 Δmax=+0.00366Ω pattern_mae=0.001 max_ratio=0.988 real_max=0.313 gen_max=0.31 peak_err=0.0px
  10.0MHz K18 s0: r=0.977 Δmax=-0.00126Ω pattern_mae=0.002 max_ratio=1.004 real_max=0.308 gen_max=0.309 peak_err=0.0px
  10.0MHz K19 s0: r=0.961 Δmax=-0.000905Ω pattern_mae=0.006 max_ratio=1.003 real_max=0.312 gen_max=0.313 peak_err=0.0px
  10.0MHz K20 s0: r=0.954 Δmax=+0.00228Ω pattern_mae=0.010 max_ratio=0.992 real_max=0.291 gen_max=0.289 peak_err=63.0px
  63.0MHz K2 s0: r=0.967 Δmax=-0.311Ω pattern_mae=0.104 max_ratio=1.095 real_max=3.29 gen_max=3.6 peak_err=0.0px
  63.0MHz K3 s0: r=0.989 Δmax=+0.0849Ω pattern_mae=0.049 max_ratio=0.974 real_max=3.21 gen_max=3.13 peak_err=0.0px
  63.0MHz K4 s0: r=0.991 Δmax=+0.0701Ω pattern_mae=0.003 max_ratio=0.971 real_max=2.42 gen_max=2.35 peak_err=0.0px
  63.0MHz K5 s0: r=0.994 Δmax=+0.229Ω pattern_mae=0.002 max_ratio=0.914 real_max=2.66 gen_max=2.43 peak_err=0.0px
  63.0MHz K6 s0: r=0.995 Δmax=+0.0848Ω pattern_mae=0.001 max_ratio=0.964 real_max=2.33 gen_max=2.24 peak_err=0.0px
  63.0MHz K7 s0: r=0.993 Δmax=+0.169Ω pattern_mae=0.000 max_ratio=0.929 real_max=2.37 gen_max=2.2 peak_err=0.0px
  63.0MHz K8 s0: r=0.991 Δmax=+0.143Ω pattern_mae=0.004 max_ratio=0.940 real_max=2.38 gen_max=2.24 peak_err=0.0px
  63.0MHz K9 s0: r=0.997 Δmax=+0.0304Ω pattern_mae=0.000 max_ratio=0.986 real_max=2.16 gen_max=2.13 peak_err=0.0px
  63.0MHz K10 s0: r=0.998 Δmax=+0.092Ω pattern_mae=0.000 max_ratio=0.959 real_max=2.25 gen_max=2.16 peak_err=0.0px
  63.0MHz K11 s0: r=0.996 Δmax=+0.0961Ω pattern_mae=0.000 max_ratio=0.955 real_max=2.16 gen_max=2.06 peak_err=0.0px
  63.0MHz K12 s0: r=0.990 Δmax=+0.12Ω pattern_mae=0.002 max_ratio=0.946 real_max=2.23 gen_max=2.11 peak_err=0.0px
  63.0MHz K13 s0: r=0.992 Δmax=+0.153Ω pattern_mae=0.001 max_ratio=0.932 real_max=2.23 gen_max=2.08 peak_err=0.0px
  63.0MHz K14 s0: r=0.978 Δmax=+0.103Ω pattern_mae=0.004 max_ratio=0.953 real_max=2.18 gen_max=2.07 peak_err=0.0px
  63.0MHz K15 s0: r=0.994 Δmax=+0.0483Ω pattern_mae=0.001 max_ratio=0.978 real_max=2.16 gen_max=2.11 peak_err=0.0px
  63.0MHz K16 s0: r=0.988 Δmax=+0.162Ω pattern_mae=0.002 max_ratio=0.920 real_max=2.04 gen_max=1.88 peak_err=63.0px
  63.0MHz K17 s0: r=0.998 Δmax=+0.141Ω pattern_mae=0.000 max_ratio=0.933 real_max=2.11 gen_max=1.97 peak_err=0.0px
  63.0MHz K18 s0: r=0.997 Δmax=+0.0964Ω pattern_mae=0.000 max_ratio=0.954 real_max=2.09 gen_max=1.99 peak_err=0.0px
  63.0MHz K19 s0: r=0.992 Δmax=+0.0467Ω pattern_mae=0.000 max_ratio=0.978 real_max=2.1 gen_max=2.05 peak_err=0.0px
  63.0MHz K20 s0: r=0.991 Δmax=+0.031Ω pattern_mae=0.001 max_ratio=0.984 real_max=1.91 gen_max=1.88 peak_err=63.0px
  150.0MHz K2 s0: r=0.764 Δmax=-4.7Ω pattern_mae=0.122 max_ratio=1.897 real_max=5.24 gen_max=9.95 peak_err=1.0px
  150.0MHz K3 s0: r=0.853 Δmax=-2.02Ω pattern_mae=0.095 max_ratio=1.396 real_max=5.1 gen_max=7.12 peak_err=0.0px
  150.0MHz K4 s0: r=0.719 Δmax=-3.41Ω pattern_mae=0.136 max_ratio=1.748 real_max=4.56 gen_max=7.98 peak_err=1.0px
  150.0MHz K5 s0: r=0.615 Δmax=-2.4Ω pattern_mae=0.101 max_ratio=1.554 real_max=4.33 gen_max=6.73 peak_err=0.0px
  150.0MHz K6 s0: r=0.608 Δmax=+12.5Ω pattern_mae=0.213 max_ratio=0.294 real_max=17.7 gen_max=5.21 peak_err=63.0px
  150.0MHz K7 s0: r=0.213 Δmax=+18.4Ω pattern_mae=0.196 max_ratio=0.203 real_max=23.1 gen_max=4.7 peak_err=63.0px
  150.0MHz K8 s0: r=0.174 Δmax=+7.81Ω pattern_mae=0.161 max_ratio=0.380 real_max=12.6 gen_max=4.78 peak_err=63.0px
  150.0MHz K9 s0: r=0.862 Δmax=+1.98Ω pattern_mae=0.064 max_ratio=0.700 real_max=6.59 gen_max=4.62 peak_err=88.4px
  150.0MHz K10 s0: r=0.945 Δmax=+2.35Ω pattern_mae=0.037 max_ratio=0.668 real_max=7.08 gen_max=4.73 peak_err=63.0px
  150.0MHz K11 s0: r=0.994 Δmax=+0.0429Ω pattern_mae=0.002 max_ratio=0.993 real_max=6.32 gen_max=6.27 peak_err=0.0px
  150.0MHz K12 s0: r=0.951 Δmax=+0.399Ω pattern_mae=0.032 max_ratio=0.945 real_max=7.24 gen_max=6.84 peak_err=0.0px
  150.0MHz K13 s0: r=0.974 Δmax=-0.903Ω pattern_mae=0.019 max_ratio=1.132 real_max=6.84 gen_max=7.74 peak_err=0.0px
  150.0MHz K14 s0: r=0.955 Δmax=-0.899Ω pattern_mae=0.017 max_ratio=1.145 real_max=6.2 gen_max=7.1 peak_err=0.0px
  150.0MHz K15 s0: r=0.991 Δmax=-0.307Ω pattern_mae=0.001 max_ratio=1.050 real_max=6.1 gen_max=6.4 peak_err=63.0px
  150.0MHz K16 s0: r=0.985 Δmax=-0.0491Ω pattern_mae=0.005 max_ratio=1.009 real_max=5.4 gen_max=5.45 peak_err=0.0px
  150.0MHz K17 s0: r=0.994 Δmax=-0.149Ω pattern_mae=0.001 max_ratio=1.026 real_max=5.75 gen_max=5.9 peak_err=0.0px
  150.0MHz K18 s0: r=0.993 Δmax=+0.0178Ω pattern_mae=0.002 max_ratio=0.997 real_max=5.57 gen_max=5.56 peak_err=0.0px
  150.0MHz K19 s0: r=0.986 Δmax=-0.587Ω pattern_mae=0.010 max_ratio=1.104 real_max=5.64 gen_max=6.22 peak_err=0.0px
  150.0MHz K20 s0: r=0.986 Δmax=-0.279Ω pattern_mae=0.008 max_ratio=1.058 real_max=4.79 gen_max=5.07 peak_err=63.0px
  270.0MHz K2 s0: r=0.442 Δmax=-13.6Ω pattern_mae=0.158 max_ratio=2.178 real_max=11.5 gen_max=25.1 peak_err=56.1px
  270.0MHz K3 s0: r=0.599 Δmax=+16.1Ω pattern_mae=0.133 max_ratio=0.603 real_max=40.7 gen_max=24.5 peak_err=62.0px
  270.0MHz K4 s0: r=0.900 Δmax=+5.97Ω pattern_mae=0.059 max_ratio=0.722 real_max=21.5 gen_max=15.5 peak_err=5.4px
  270.0MHz K5 s0: r=0.776 Δmax=-6.88Ω pattern_mae=0.111 max_ratio=1.782 real_max=8.79 gen_max=15.7 peak_err=62.0px
  270.0MHz K6 s0: r=0.705 Δmax=-5.92Ω pattern_mae=0.104 max_ratio=1.426 real_max=13.9 gen_max=19.8 peak_err=58.1px
  270.0MHz K7 s0: r=0.685 Δmax=-3.61Ω pattern_mae=0.089 max_ratio=1.428 real_max=8.44 gen_max=12 peak_err=88.4px
  270.0MHz K8 s0: r=0.859 Δmax=-2.25Ω pattern_mae=0.079 max_ratio=1.152 real_max=14.8 gen_max=17 peak_err=62.0px
  270.0MHz K9 s0: r=0.711 Δmax=+29.9Ω pattern_mae=0.086 max_ratio=0.322 real_max=44.2 gen_max=14.2 peak_err=2.0px
  270.0MHz K10 s0: r=0.820 Δmax=+5.22Ω pattern_mae=0.050 max_ratio=0.667 real_max=15.6 gen_max=10.4 peak_err=0.0px
  270.0MHz K11 s0: r=0.866 Δmax=+2.01Ω pattern_mae=0.066 max_ratio=0.818 real_max=11 gen_max=9.01 peak_err=12.4px
  270.0MHz K12 s0: r=0.953 Δmax=+1.66Ω pattern_mae=0.031 max_ratio=0.864 real_max=12.2 gen_max=10.6 peak_err=0.0px
  270.0MHz K13 s0: r=0.976 Δmax=+29.3Ω pattern_mae=0.038 max_ratio=0.301 real_max=41.9 gen_max=12.6 peak_err=18.0px
  270.0MHz K14 s0: r=0.675 Δmax=+9.53Ω pattern_mae=0.100 max_ratio=0.656 real_max=27.7 gen_max=18.2 peak_err=9.2px
  270.0MHz K15 s0: r=0.837 Δmax=-5.01Ω pattern_mae=0.147 max_ratio=1.615 real_max=8.13 gen_max=13.1 peak_err=58.1px
  270.0MHz K16 s0: r=0.780 Δmax=+3.63Ω pattern_mae=0.049 max_ratio=0.790 real_max=17.2 gen_max=13.6 peak_err=63.0px
  270.0MHz K17 s0: r=0.820 Δmax=+7.78Ω pattern_mae=0.076 max_ratio=0.649 real_max=22.2 gen_max=14.4 peak_err=85.6px
  270.0MHz K18 s0: r=0.513 Δmax=-5.49Ω pattern_mae=0.135 max_ratio=1.453 real_max=12.1 gen_max=17.6 peak_err=63.0px
  270.0MHz K19 s0: r=0.960 Δmax=+8.5Ω pattern_mae=0.079 max_ratio=0.714 real_max=29.8 gen_max=21.3 peak_err=0.0px
  270.0MHz K20 s0: r=0.874 Δmax=-5.56Ω pattern_mae=0.055 max_ratio=1.488 real_max=11.4 gen_max=17 peak_err=88.4px
  450.0MHz K2 s0: r=0.983 Δmax=-1.95Ω pattern_mae=0.004 max_ratio=1.178 real_max=11 gen_max=12.9 peak_err=0.0px
  450.0MHz K3 s0: r=0.968 Δmax=-1.73Ω pattern_mae=0.012 max_ratio=1.157 real_max=11.1 gen_max=12.8 peak_err=0.0px
  450.0MHz K4 s0: r=0.966 Δmax=-2.89Ω pattern_mae=0.014 max_ratio=1.264 real_max=10.9 gen_max=13.8 peak_err=0.0px
  450.0MHz K5 s0: r=0.931 Δmax=-2.06Ω pattern_mae=0.028 max_ratio=1.253 real_max=8.15 gen_max=10.2 peak_err=19.0px
  450.0MHz K6 s0: r=0.992 Δmax=-0.764Ω pattern_mae=0.002 max_ratio=1.077 real_max=9.95 gen_max=10.7 peak_err=0.0px
  450.0MHz K7 s0: r=0.645 Δmax=-3.58Ω pattern_mae=0.097 max_ratio=1.504 real_max=7.1 gen_max=10.7 peak_err=29.0px
  450.0MHz K8 s0: r=0.891 Δmax=-2.01Ω pattern_mae=0.036 max_ratio=1.268 real_max=7.5 gen_max=9.5 peak_err=0.0px
  450.0MHz K9 s0: r=0.633 Δmax=-2.28Ω pattern_mae=0.093 max_ratio=1.329 real_max=6.94 gen_max=9.22 peak_err=70.7px
  450.0MHz K10 s0: r=0.224 Δmax=+10.6Ω pattern_mae=0.146 max_ratio=0.450 real_max=19.3 gen_max=8.69 peak_err=0.0px
  450.0MHz K11 s0: r=0.849 Δmax=+7.17Ω pattern_mae=0.030 max_ratio=0.585 real_max=17.3 gen_max=10.1 peak_err=0.0px
  450.0MHz K12 s0: r=0.931 Δmax=-0.353Ω pattern_mae=0.027 max_ratio=1.049 real_max=7.27 gen_max=7.62 peak_err=1.0px
  450.0MHz K13 s0: r=0.960 Δmax=+0.0967Ω pattern_mae=0.017 max_ratio=0.991 real_max=10.8 gen_max=10.7 peak_err=0.0px
  450.0MHz K14 s0: r=0.945 Δmax=+1.45Ω pattern_mae=0.030 max_ratio=0.884 real_max=12.4 gen_max=11 peak_err=63.0px
  450.0MHz K15 s0: r=0.953 Δmax=+4.26Ω pattern_mae=0.020 max_ratio=0.725 real_max=15.5 gen_max=11.2 peak_err=0.0px
  450.0MHz K16 s0: r=0.890 Δmax=-0.659Ω pattern_mae=0.027 max_ratio=1.052 real_max=12.6 gen_max=13.2 peak_err=0.0px
  450.0MHz K17 s0: r=0.922 Δmax=+0.869Ω pattern_mae=0.029 max_ratio=0.930 real_max=12.5 gen_max=11.6 peak_err=62.0px
  450.0MHz K18 s0: r=0.903 Δmax=+5.39Ω pattern_mae=0.039 max_ratio=0.663 real_max=16 gen_max=10.6 peak_err=33.6px
  450.0MHz K19 s0: r=0.841 Δmax=-0.603Ω pattern_mae=0.036 max_ratio=1.059 real_max=10.2 gen_max=10.8 peak_err=0.0px
  450.0MHz K20 s0: r=0.817 Δmax=+8.5Ω pattern_mae=0.049 max_ratio=0.579 real_max=20.2 gen_max=11.7 peak_err=63.0px
  500.0MHz K2 s0: r=0.559 Δmax=-5.47Ω pattern_mae=0.106 max_ratio=1.414 real_max=13.2 gen_max=18.7 peak_err=59.7px
  500.0MHz K3 s0: r=0.846 Δmax=-1.23Ω pattern_mae=0.084 max_ratio=1.075 real_max=16.4 gen_max=17.6 peak_err=70.2px
  500.0MHz K4 s0: r=0.437 Δmax=-7.12Ω pattern_mae=0.124 max_ratio=1.679 real_max=10.5 gen_max=17.6 peak_err=63.0px
  500.0MHz K5 s0: r=0.912 Δmax=+4.33Ω pattern_mae=0.056 max_ratio=0.767 real_max=18.6 gen_max=14.2 peak_err=88.4px
  500.0MHz K6 s0: r=0.942 Δmax=+1.51Ω pattern_mae=0.054 max_ratio=0.911 real_max=17 gen_max=15.5 peak_err=29.0px
  500.0MHz K7 s0: r=0.938 Δmax=+0.549Ω pattern_mae=0.065 max_ratio=0.961 real_max=14.2 gen_max=13.7 peak_err=0.0px
  500.0MHz K8 s0: r=0.917 Δmax=+1.86Ω pattern_mae=0.054 max_ratio=0.880 real_max=15.5 gen_max=13.6 peak_err=0.0px
  500.0MHz K9 s0: r=0.792 Δmax=+13.6Ω pattern_mae=0.074 max_ratio=0.464 real_max=25.4 gen_max=11.8 peak_err=71.1px
  500.0MHz K10 s0: r=0.933 Δmax=+0.902Ω pattern_mae=0.044 max_ratio=0.931 real_max=13 gen_max=12.1 peak_err=70.2px
  500.0MHz K11 s0: r=0.669 Δmax=+1.8Ω pattern_mae=0.080 max_ratio=0.875 real_max=14.4 gen_max=12.6 peak_err=32.0px
  500.0MHz K12 s0: r=0.887 Δmax=+6.43Ω pattern_mae=0.040 max_ratio=0.609 real_max=16.5 gen_max=10 peak_err=2.0px
  500.0MHz K13 s0: r=0.565 Δmax=-4.25Ω pattern_mae=0.102 max_ratio=1.352 real_max=12.1 gen_max=16.3 peak_err=63.0px
  500.0MHz K14 s0: r=0.703 Δmax=-0.661Ω pattern_mae=0.060 max_ratio=1.056 real_max=11.9 gen_max=12.5 peak_err=0.0px
  500.0MHz K15 s0: r=0.812 Δmax=-10.3Ω pattern_mae=0.073 max_ratio=1.805 real_max=12.8 gen_max=23.2 peak_err=87.7px
  500.0MHz K16 s0: r=0.784 Δmax=+11Ω pattern_mae=0.081 max_ratio=0.523 real_max=23 gen_max=12 peak_err=1.4px
  500.0MHz K17 s0: r=0.684 Δmax=-1.23Ω pattern_mae=0.084 max_ratio=1.100 real_max=12.3 gen_max=13.5 peak_err=70.7px
  500.0MHz K18 s0: r=0.950 Δmax=-0.315Ω pattern_mae=0.021 max_ratio=1.030 real_max=10.3 gen_max=10.7 peak_err=0.0px
  500.0MHz K19 s0: r=0.739 Δmax=-1.82Ω pattern_mae=0.049 max_ratio=1.174 real_max=10.4 gen_max=12.2 peak_err=0.0px
  500.0MHz K20 s0: r=0.859 Δmax=-0.129Ω pattern_mae=0.044 max_ratio=1.014 real_max=9.48 gen_max=9.61 peak_err=0.0px
```
