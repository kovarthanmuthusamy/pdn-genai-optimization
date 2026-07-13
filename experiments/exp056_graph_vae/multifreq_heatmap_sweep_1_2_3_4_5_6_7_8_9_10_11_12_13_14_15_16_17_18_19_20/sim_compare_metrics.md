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
10  | 2  | 0      | 0.9884    | 0.04278      | 0.008204    | 0.874     | 0               | 0.3396       | 0.2968     
10  | 3  | 0      | 0.9861    | 0.02912      | 0.006997    | 0.9181    | 63              | 0.3556       | 0.3265     
10  | 4  | 0      | 0.9942    | 0.03122      | 0.00191     | 0.9069    | 0               | 0.3352       | 0.304      
10  | 5  | 0      | 0.9902    | 0.03699      | 0.007701    | 0.8959    | 63              | 0.3554       | 0.3184     
10  | 6  | 0      | 0.9957    | 0.03323      | 0.002273    | 0.9008    | 0               | 0.3351       | 0.3018     
10  | 7  | 0      | 0.9689    | 0.04774      | 0.009561    | 0.8612    | 0               | 0.344        | 0.2962     
10  | 8  | 0      | 0.987     | 0.04431      | 0.01007     | 0.873     | 0               | 0.3489       | 0.3046     
10  | 9  | 0      | 0.9927    | 0.01804      | 0.009318    | 0.943     | 0               | 0.3163       | 0.2983     
10  | 10 | 0      | 0.9774    | 0.03323      | 0.01012     | 0.9006    | 0               | 0.3342       | 0.301      
10  | 11 | 0      | 0.9695    | 0.0209       | 0.01133     | 0.9341    | 0               | 0.3173       | 0.2964     
10  | 12 | 0      | 0.9804    | 0.03429      | 0.01514     | 0.898     | 0               | 0.3362       | 0.3019     
10  | 13 | 0      | 0.9739    | 0.0243       | 0.004134    | 0.9265    | 0               | 0.3304       | 0.3061     
10  | 14 | 0      | 0.9609    | 0.02486      | 0.0148      | 0.9228    | 0               | 0.322        | 0.2971     
10  | 15 | 0      | 0.9649    | 0.02673      | 0.007964    | 0.9171    | 63              | 0.3224       | 0.2956     
10  | 16 | 0      | 0.927     | 0.007371     | 0.02563     | 0.9742    | 0               | 0.2852       | 0.2778     
10  | 17 | 0      | 0.9834    | 0.02996      | 0.01972     | 0.9047    | 0               | 0.3145       | 0.2845     
10  | 18 | 0      | 0.9745    | 0.01802      | 0.009519    | 0.9415    | 0               | 0.3082       | 0.2902     
10  | 19 | 0      | 0.9673    | 0.01908      | 0.01609     | 0.9377    | 0               | 0.3063       | 0.2873     
10  | 20 | 0      | 0.9494    | 0.03331      | 0.01143     | 0.8872    | 0               | 0.2952       | 0.2619     
63  | 2  | 0      | 0.9827    | 0.2912       | 0.005044    | 0.9116    | 0               | 3.294        | 3.002      
63  | 3  | 0      | 0.9861    | -0.003442    | 0.01234     | 1.001     | 0               | 3.211        | 3.214      
63  | 4  | 0      | 0.9939    | 0.09702      | 0.0003527   | 0.9599    | 0               | 2.418        | 2.321      
63  | 5  | 0      | 0.9899    | 0.1429       | 0.01375     | 0.9464    | 0               | 2.663        | 2.521      
63  | 6  | 0      | 0.9956    | 0.2402       | 0.0007305   | 0.8967    | 0               | 2.326        | 2.086      
63  | 7  | 0      | 0.9863    | 0.2775       | 0.002869    | 0.8827    | 0               | 2.365        | 2.088      
63  | 8  | 0      | 0.9961    | 0.1534       | 0.0002841   | 0.9354    | 0               | 2.376        | 2.223      
63  | 9  | 0      | 0.9975    | 0.15         | 3.703e-05   | 0.9307    | 63              | 2.165        | 2.015      
63  | 10 | 0      | 0.9972    | 0.1164       | 0.0001551   | 0.9482    | 0               | 2.248        | 2.132      
63  | 11 | 0      | 0.9909    | 0.119        | 0.004994    | 0.945     | 63              | 2.164        | 2.045      
63  | 12 | 0      | 0.9903    | 0.07703      | 0.004636    | 0.9652    | 0               | 2.214        | 2.136      
63  | 13 | 0      | 0.9818    | 0.07846      | 0.009232    | 0.9648    | 0               | 2.232        | 2.154      
63  | 14 | 0      | 0.9923    | 0.08611      | 0.001591    | 0.9604    | 0               | 2.174        | 2.087      
63  | 15 | 0      | 0.992     | 0.1332       | 0.0002382   | 0.9388    | 0               | 2.175        | 2.042      
63  | 16 | 0      | 0.9843    | 0.1148       | 0.003154    | 0.9435    | 63              | 2.033        | 1.918      
63  | 17 | 0      | 0.996     | 0.1664       | 0.00447     | 0.9212    | 0               | 2.112        | 1.946      
63  | 18 | 0      | 0.9956    | 0.08716      | 3.489e-05   | 0.9583    | 0               | 2.088        | 2.001      
63  | 19 | 0      | 0.9878    | 0.1734       | 0.005626    | 0.9181    | 63              | 2.118        | 1.945      
63  | 20 | 0      | 0.9819    | 0.1412       | 0.005894    | 0.9283    | 0               | 1.97         | 1.828      
150 | 2  | 0      | 0.9198    | -3.013       | 0.08079     | 1.575     | 0               | 5.242        | 8.255      
150 | 3  | 0      | 0.9503    | -1.354       | 0.03817     | 1.265     | 0               | 5.102        | 6.456      
150 | 4  | 0      | 0.8088    | -2.751       | 0.1027      | 1.603     | 0               | 4.563        | 7.314      
150 | 5  | 0      | 0.7766    | -1.599       | 0.06561     | 1.369     | 0               | 4.329        | 5.928      
150 | 6  | 0      | 0.9218    | 11.91        | 0.06963     | 0.3304    | 9               | 17.79        | 5.88       
150 | 7  | 0      | 0.1174    | 13.21        | 0.1886      | 0.2525    | 63              | 17.67        | 4.46       
150 | 8  | 0      | 0.9553    | 6.8          | 0.0585      | 0.4294    | 0               | 11.92        | 5.118      
150 | 9  | 0      | 0.9803    | 0.1194       | 0.03924     | 0.9818    | 0               | 6.563        | 6.444      
150 | 10 | 0      | 0.9846    | 0.5032       | 0.02127     | 0.9289    | 0               | 7.081        | 6.577      
150 | 11 | 0      | 0.9717    | -0.603       | 0.01694     | 1.096     | 0               | 6.299        | 6.902      
150 | 12 | 0      | 0.9602    | -2.017       | 0.03009     | 1.296     | 1               | 6.818        | 8.834      
150 | 13 | 0      | 0.9813    | -0.9021      | 0.01149     | 1.132     | 0               | 6.83         | 7.732      
150 | 14 | 0      | 0.9684    | -0.4356      | 0.01023     | 1.071     | 0               | 6.166        | 6.602      
150 | 15 | 0      | 0.9876    | -0.2167      | 0.003853    | 1.035     | 0               | 6.177        | 6.394      
150 | 16 | 0      | 0.9625    | -0.4305      | 0.008184    | 1.083     | 0               | 5.182        | 5.612      
150 | 17 | 0      | 0.9926    | 0.1229       | 0.005069    | 0.9785    | 0               | 5.723        | 5.6        
150 | 18 | 0      | 0.993     | -0.1598      | 0.0002534   | 1.029     | 0               | 5.592        | 5.751      
150 | 19 | 0      | 0.9781    | -0.1097      | 0.00846     | 1.019     | 63              | 5.732        | 5.842      
150 | 20 | 0      | 0.9831    | 0.06572      | 0.003178    | 0.9867    | 0               | 4.952        | 4.886      
270 | 2  | 0      | 0.861     | -6.683       | 0.07443     | 1.579     | 6.708           | 11.54        | 18.23      
270 | 3  | 0      | 0.7824    | 18.88        | 0.101       | 0.5357    | 62.03           | 40.66        | 21.78      
270 | 4  | 0      | 0.8078    | 6.532        | 0.07916     | 0.6958    | 5.385           | 21.47        | 14.94      
270 | 5  | 0      | 0.68      | -3.554       | 0.1141      | 1.405     | 62              | 8.772        | 12.33      
270 | 6  | 0      | 0.8616    | -0.786       | 0.07931     | 1.057     | 5.831           | 13.89        | 14.68      
270 | 7  | 0      | 0.791     | -2.922       | 0.07167     | 1.347     | 88.39           | 8.417        | 11.34      
270 | 8  | 0      | 0.8652    | -1.236       | 0.06715     | 1.084     | 62              | 14.79        | 16.02      
270 | 9  | 0      | 0.815     | 25.1         | 0.07629     | 0.4138    | 3.162           | 42.82        | 17.72      
270 | 10 | 0      | 0.5818    | 2.398        | 0.08694     | 0.8467    | 88.39           | 15.64        | 13.25      
270 | 11 | 0      | 0.5224    | -10.91       | 0.128       | 2.401     | 62              | 7.789        | 18.7       
270 | 12 | 0      | 0.8313    | -1.981       | 0.09228     | 1.171     | 63              | 11.57        | 13.56      
270 | 13 | 0      | 0.804     | 19.52        | 0.06968     | 0.4335    | 0               | 34.46        | 14.94      
270 | 14 | 0      | 0.3589    | -9.683       | 0.1279      | 2.09      | 58.08           | 8.887        | 18.57      
270 | 15 | 0      | 0.5348    | -11.27       | 0.1622      | 2.341     | 62              | 8.403        | 19.67      
270 | 16 | 0      | 0.9283    | 0.008421     | 0.02795     | 0.9995    | 0               | 16.33        | 16.32      
270 | 17 | 0      | 0.8562    | 11.55        | 0.05832     | 0.6171    | 86.28           | 30.17        | 18.62      
270 | 18 | 0      | 0.9306    | 33.1         | 0.04122     | 0.3784    | 0               | 53.25        | 20.15      
270 | 19 | 0      | 0.7104    | 50.33        | 0.09202     | 0.2992    | 63              | 71.82        | 21.49      
270 | 20 | 0      | 0.9426    | 1.373        | 0.05617     | 0.9191    | 0               | 16.97        | 15.6       
450 | 2  | 0      | 0.9423    | 0.2663       | 0.02449     | 0.9757    | 0               | 10.96        | 10.7       
450 | 3  | 0      | 0.9328    | 0.1717       | 0.01995     | 0.9843    | 0               | 10.92        | 10.75      
450 | 4  | 0      | 0.9884    | 0.2773       | 0.01155     | 0.9746    | 0               | 10.92        | 10.65      
450 | 5  | 0      | 0.9691    | -0.6664      | 0.01391     | 1.082     | 76.03           | 8.147        | 8.813      
450 | 6  | 0      | 0.9955    | -0.2052      | 0.0004725   | 1.021     | 0               | 9.95         | 10.15      
450 | 7  | 0      | 0.7815    | -2.924       | 0.08622     | 1.411     | 29              | 7.105        | 10.03      
450 | 8  | 0      | 0.8958    | -1.688       | 0.05281     | 1.235     | 29              | 7.173        | 8.861      
450 | 9  | 0      | 0.4785    | -1.937       | 0.1387      | 1.289     | 34              | 6.705        | 8.642      
450 | 10 | 0      | 0.401     | 9.393        | 0.1229      | 0.5136    | 63              | 19.31        | 9.92       
450 | 11 | 0      | 0.9726    | 2.698        | 0.0193      | 0.8228    | 0               | 15.23        | 12.53      
450 | 12 | 0      | 0.9618    | 0.2426       | 0.01319     | 0.9669    | 70.21           | 7.329        | 7.087      
450 | 13 | 0      | 0.9033    | -0.2411      | 0.02438     | 1.021     | 62              | 11.34        | 11.58      
450 | 14 | 0      | 0.8818    | 5.79         | 0.03235     | 0.5981    | 76.03           | 14.41        | 8.617      
450 | 15 | 0      | 0.9277    | 1.373        | 0.02146     | 0.8801    | 0               | 11.45        | 10.07      
450 | 16 | 0      | 0.8717    | 6.497        | 0.03376     | 0.6177    | 0               | 16.99        | 10.5       
450 | 17 | 0      | 0.9471    | 1.967        | 0.02098     | 0.8374    | 0               | 12.1         | 10.13      
450 | 18 | 0      | 0.8708    | 6.371        | 0.05908     | 0.6213    | 63              | 16.82        | 10.45      
450 | 19 | 0      | 0.7964    | 0.07857      | 0.03344     | 0.9927    | 63              | 10.78        | 10.7       
450 | 20 | 0      | 0.9254    | 4.155        | 0.02423     | 0.6953    | 0               | 13.64        | 9.482      
500 | 2  | 0      | 0.9214    | 0.695        | 0.03688     | 0.9474    | 60.21           | 13.21        | 12.52      
500 | 3  | 0      | 0.7734    | 5.455        | 0.07758     | 0.667     | 70.21           | 16.38        | 10.92      
500 | 4  | 0      | 0.9651    | -0.1565      | 0.01504     | 1.015     | 0               | 10.49        | 10.65      
500 | 5  | 0      | 0.809     | 4.809        | 0.0871      | 0.7409    | 88.39           | 18.56        | 13.75      
500 | 6  | 0      | 0.8411    | 5.771        | 0.07842     | 0.6651    | 63              | 17.23        | 11.46      
500 | 7  | 0      | 0.9766    | -0.1938      | 0.0227      | 1.012     | 0               | 16.02        | 16.22      
500 | 8  | 0      | 0.9698    | 1.42         | 0.02693     | 0.9081    | 0               | 15.46        | 14.04      
500 | 9  | 0      | 0.8457    | 11.66        | 0.07224     | 0.4846    | 1               | 22.62        | 10.96      
500 | 10 | 0      | 0.9821    | -0.2964      | 0.008699    | 1.023     | 1               | 13           | 13.29      
500 | 11 | 0      | 0.8612    | 0.8805       | 0.04328     | 0.9349    | 63              | 13.52        | 12.64      
500 | 12 | 0      | 0.929     | 4.51         | 0.02877     | 0.713     | 32              | 15.72        | 11.21      
500 | 13 | 0      | 0.3901    | 1.331        | 0.1045      | 0.8942    | 63              | 12.58        | 11.25      
500 | 14 | 0      | 0.9501    | 0.7109       | 0.02432     | 0.9415    | 0               | 12.15        | 11.44      
500 | 15 | 0      | 0.7565    | 3.087        | 0.0487      | 0.7787    | 85.56           | 13.95        | 10.86      
500 | 16 | 0      | 0.8878    | -1.094       | 0.0502      | 1.101     | 0               | 10.85        | 11.94      
500 | 17 | 0      | 0.9528    | 1.104        | 0.0305      | 0.9102    | 70.66           | 12.29        | 11.19      
500 | 18 | 0      | 0.9408    | -1.356       | 0.05072     | 1.136     | 0               | 9.958        | 11.31      
500 | 19 | 0      | 0.7859    | -1.154       | 0.04815     | 1.109     | 0               | 10.57        | 11.73      
500 | 20 | 0      | 0.7724    | 1.406        | 0.05626     | 0.8772    | 0               | 11.45        | 10.04      
```

## Mean by MHz (primary)

```
mhz | n  | pearson_r_mean | max_diff_ohm_mean | pattern_mae_mean | max_ratio_mean | peak_loc_err_px_mean | real_max_ohm_mean | gen_max_ohm_mean
----|----|----------------|-------------------|------------------|----------------|----------------------|-------------------|-----------------
10  | 19 | 0.9754         | 0.02923           | 0.01063          | 0.9114         | 9.947                | 0.3264            | 0.2972          
63  | 19 | 0.9904         | 0.1391            | 0.00397          | 0.9398         | 13.26                | 2.334             | 2.195           
150 | 19 | 0.9049         | 1.007             | 0.04012          | 1.024          | 7.158                | 7.354             | 6.347           
270 | 19 | 0.7613         | 6.303             | 0.08452          | 1.085          | 40.96                | 23.03             | 16.73           
450 | 19 | 0.8655         | 1.664             | 0.03964          | 0.9232         | 29.75                | 11.65             | 9.982           
500 | 19 | 0.8585         | 2.031             | 0.04794          | 0.8873         | 31.48                | 14                | 11.97           
```

## Agent copy block

```
sim_compare_metrics: n=114
  10.0MHz K2 s0: r=0.988 Δmax=+0.0428Ω pattern_mae=0.008 max_ratio=0.874 real_max=0.34 gen_max=0.297 peak_err=0.0px
  10.0MHz K3 s0: r=0.986 Δmax=+0.0291Ω pattern_mae=0.007 max_ratio=0.918 real_max=0.356 gen_max=0.327 peak_err=63.0px
  10.0MHz K4 s0: r=0.994 Δmax=+0.0312Ω pattern_mae=0.002 max_ratio=0.907 real_max=0.335 gen_max=0.304 peak_err=0.0px
  10.0MHz K5 s0: r=0.990 Δmax=+0.037Ω pattern_mae=0.008 max_ratio=0.896 real_max=0.355 gen_max=0.318 peak_err=63.0px
  10.0MHz K6 s0: r=0.996 Δmax=+0.0332Ω pattern_mae=0.002 max_ratio=0.901 real_max=0.335 gen_max=0.302 peak_err=0.0px
  10.0MHz K7 s0: r=0.969 Δmax=+0.0477Ω pattern_mae=0.010 max_ratio=0.861 real_max=0.344 gen_max=0.296 peak_err=0.0px
  10.0MHz K8 s0: r=0.987 Δmax=+0.0443Ω pattern_mae=0.010 max_ratio=0.873 real_max=0.349 gen_max=0.305 peak_err=0.0px
  10.0MHz K9 s0: r=0.993 Δmax=+0.018Ω pattern_mae=0.009 max_ratio=0.943 real_max=0.316 gen_max=0.298 peak_err=0.0px
  10.0MHz K10 s0: r=0.977 Δmax=+0.0332Ω pattern_mae=0.010 max_ratio=0.901 real_max=0.334 gen_max=0.301 peak_err=0.0px
  10.0MHz K11 s0: r=0.969 Δmax=+0.0209Ω pattern_mae=0.011 max_ratio=0.934 real_max=0.317 gen_max=0.296 peak_err=0.0px
  10.0MHz K12 s0: r=0.980 Δmax=+0.0343Ω pattern_mae=0.015 max_ratio=0.898 real_max=0.336 gen_max=0.302 peak_err=0.0px
  10.0MHz K13 s0: r=0.974 Δmax=+0.0243Ω pattern_mae=0.004 max_ratio=0.926 real_max=0.33 gen_max=0.306 peak_err=0.0px
  10.0MHz K14 s0: r=0.961 Δmax=+0.0249Ω pattern_mae=0.015 max_ratio=0.923 real_max=0.322 gen_max=0.297 peak_err=0.0px
  10.0MHz K15 s0: r=0.965 Δmax=+0.0267Ω pattern_mae=0.008 max_ratio=0.917 real_max=0.322 gen_max=0.296 peak_err=63.0px
  10.0MHz K16 s0: r=0.927 Δmax=+0.00737Ω pattern_mae=0.026 max_ratio=0.974 real_max=0.285 gen_max=0.278 peak_err=0.0px
  10.0MHz K17 s0: r=0.983 Δmax=+0.03Ω pattern_mae=0.020 max_ratio=0.905 real_max=0.314 gen_max=0.285 peak_err=0.0px
  10.0MHz K18 s0: r=0.975 Δmax=+0.018Ω pattern_mae=0.010 max_ratio=0.942 real_max=0.308 gen_max=0.29 peak_err=0.0px
  10.0MHz K19 s0: r=0.967 Δmax=+0.0191Ω pattern_mae=0.016 max_ratio=0.938 real_max=0.306 gen_max=0.287 peak_err=0.0px
  10.0MHz K20 s0: r=0.949 Δmax=+0.0333Ω pattern_mae=0.011 max_ratio=0.887 real_max=0.295 gen_max=0.262 peak_err=0.0px
  63.0MHz K2 s0: r=0.983 Δmax=+0.291Ω pattern_mae=0.005 max_ratio=0.912 real_max=3.29 gen_max=3 peak_err=0.0px
  63.0MHz K3 s0: r=0.986 Δmax=-0.00344Ω pattern_mae=0.012 max_ratio=1.001 real_max=3.21 gen_max=3.21 peak_err=0.0px
  63.0MHz K4 s0: r=0.994 Δmax=+0.097Ω pattern_mae=0.000 max_ratio=0.960 real_max=2.42 gen_max=2.32 peak_err=0.0px
  63.0MHz K5 s0: r=0.990 Δmax=+0.143Ω pattern_mae=0.014 max_ratio=0.946 real_max=2.66 gen_max=2.52 peak_err=0.0px
  63.0MHz K6 s0: r=0.996 Δmax=+0.24Ω pattern_mae=0.001 max_ratio=0.897 real_max=2.33 gen_max=2.09 peak_err=0.0px
  63.0MHz K7 s0: r=0.986 Δmax=+0.278Ω pattern_mae=0.003 max_ratio=0.883 real_max=2.37 gen_max=2.09 peak_err=0.0px
  63.0MHz K8 s0: r=0.996 Δmax=+0.153Ω pattern_mae=0.000 max_ratio=0.935 real_max=2.38 gen_max=2.22 peak_err=0.0px
  63.0MHz K9 s0: r=0.998 Δmax=+0.15Ω pattern_mae=0.000 max_ratio=0.931 real_max=2.17 gen_max=2.02 peak_err=63.0px
  63.0MHz K10 s0: r=0.997 Δmax=+0.116Ω pattern_mae=0.000 max_ratio=0.948 real_max=2.25 gen_max=2.13 peak_err=0.0px
  63.0MHz K11 s0: r=0.991 Δmax=+0.119Ω pattern_mae=0.005 max_ratio=0.945 real_max=2.16 gen_max=2.05 peak_err=63.0px
  63.0MHz K12 s0: r=0.990 Δmax=+0.077Ω pattern_mae=0.005 max_ratio=0.965 real_max=2.21 gen_max=2.14 peak_err=0.0px
  63.0MHz K13 s0: r=0.982 Δmax=+0.0785Ω pattern_mae=0.009 max_ratio=0.965 real_max=2.23 gen_max=2.15 peak_err=0.0px
  63.0MHz K14 s0: r=0.992 Δmax=+0.0861Ω pattern_mae=0.002 max_ratio=0.960 real_max=2.17 gen_max=2.09 peak_err=0.0px
  63.0MHz K15 s0: r=0.992 Δmax=+0.133Ω pattern_mae=0.000 max_ratio=0.939 real_max=2.17 gen_max=2.04 peak_err=0.0px
  63.0MHz K16 s0: r=0.984 Δmax=+0.115Ω pattern_mae=0.003 max_ratio=0.944 real_max=2.03 gen_max=1.92 peak_err=63.0px
  63.0MHz K17 s0: r=0.996 Δmax=+0.166Ω pattern_mae=0.004 max_ratio=0.921 real_max=2.11 gen_max=1.95 peak_err=0.0px
  63.0MHz K18 s0: r=0.996 Δmax=+0.0872Ω pattern_mae=0.000 max_ratio=0.958 real_max=2.09 gen_max=2 peak_err=0.0px
  63.0MHz K19 s0: r=0.988 Δmax=+0.173Ω pattern_mae=0.006 max_ratio=0.918 real_max=2.12 gen_max=1.94 peak_err=63.0px
  63.0MHz K20 s0: r=0.982 Δmax=+0.141Ω pattern_mae=0.006 max_ratio=0.928 real_max=1.97 gen_max=1.83 peak_err=0.0px
  150.0MHz K2 s0: r=0.920 Δmax=-3.01Ω pattern_mae=0.081 max_ratio=1.575 real_max=5.24 gen_max=8.26 peak_err=0.0px
  150.0MHz K3 s0: r=0.950 Δmax=-1.35Ω pattern_mae=0.038 max_ratio=1.265 real_max=5.1 gen_max=6.46 peak_err=0.0px
  150.0MHz K4 s0: r=0.809 Δmax=-2.75Ω pattern_mae=0.103 max_ratio=1.603 real_max=4.56 gen_max=7.31 peak_err=0.0px
  150.0MHz K5 s0: r=0.777 Δmax=-1.6Ω pattern_mae=0.066 max_ratio=1.369 real_max=4.33 gen_max=5.93 peak_err=0.0px
  150.0MHz K6 s0: r=0.922 Δmax=+11.9Ω pattern_mae=0.070 max_ratio=0.330 real_max=17.8 gen_max=5.88 peak_err=9.0px
  150.0MHz K7 s0: r=0.117 Δmax=+13.2Ω pattern_mae=0.189 max_ratio=0.252 real_max=17.7 gen_max=4.46 peak_err=63.0px
  150.0MHz K8 s0: r=0.955 Δmax=+6.8Ω pattern_mae=0.058 max_ratio=0.429 real_max=11.9 gen_max=5.12 peak_err=0.0px
  150.0MHz K9 s0: r=0.980 Δmax=+0.119Ω pattern_mae=0.039 max_ratio=0.982 real_max=6.56 gen_max=6.44 peak_err=0.0px
  150.0MHz K10 s0: r=0.985 Δmax=+0.503Ω pattern_mae=0.021 max_ratio=0.929 real_max=7.08 gen_max=6.58 peak_err=0.0px
  150.0MHz K11 s0: r=0.972 Δmax=-0.603Ω pattern_mae=0.017 max_ratio=1.096 real_max=6.3 gen_max=6.9 peak_err=0.0px
  150.0MHz K12 s0: r=0.960 Δmax=-2.02Ω pattern_mae=0.030 max_ratio=1.296 real_max=6.82 gen_max=8.83 peak_err=1.0px
  150.0MHz K13 s0: r=0.981 Δmax=-0.902Ω pattern_mae=0.011 max_ratio=1.132 real_max=6.83 gen_max=7.73 peak_err=0.0px
  150.0MHz K14 s0: r=0.968 Δmax=-0.436Ω pattern_mae=0.010 max_ratio=1.071 real_max=6.17 gen_max=6.6 peak_err=0.0px
  150.0MHz K15 s0: r=0.988 Δmax=-0.217Ω pattern_mae=0.004 max_ratio=1.035 real_max=6.18 gen_max=6.39 peak_err=0.0px
  150.0MHz K16 s0: r=0.963 Δmax=-0.43Ω pattern_mae=0.008 max_ratio=1.083 real_max=5.18 gen_max=5.61 peak_err=0.0px
  150.0MHz K17 s0: r=0.993 Δmax=+0.123Ω pattern_mae=0.005 max_ratio=0.979 real_max=5.72 gen_max=5.6 peak_err=0.0px
  150.0MHz K18 s0: r=0.993 Δmax=-0.16Ω pattern_mae=0.000 max_ratio=1.029 real_max=5.59 gen_max=5.75 peak_err=0.0px
  150.0MHz K19 s0: r=0.978 Δmax=-0.11Ω pattern_mae=0.008 max_ratio=1.019 real_max=5.73 gen_max=5.84 peak_err=63.0px
  150.0MHz K20 s0: r=0.983 Δmax=+0.0657Ω pattern_mae=0.003 max_ratio=0.987 real_max=4.95 gen_max=4.89 peak_err=0.0px
  270.0MHz K2 s0: r=0.861 Δmax=-6.68Ω pattern_mae=0.074 max_ratio=1.579 real_max=11.5 gen_max=18.2 peak_err=6.7px
  270.0MHz K3 s0: r=0.782 Δmax=+18.9Ω pattern_mae=0.101 max_ratio=0.536 real_max=40.7 gen_max=21.8 peak_err=62.0px
  270.0MHz K4 s0: r=0.808 Δmax=+6.53Ω pattern_mae=0.079 max_ratio=0.696 real_max=21.5 gen_max=14.9 peak_err=5.4px
  270.0MHz K5 s0: r=0.680 Δmax=-3.55Ω pattern_mae=0.114 max_ratio=1.405 real_max=8.77 gen_max=12.3 peak_err=62.0px
  270.0MHz K6 s0: r=0.862 Δmax=-0.786Ω pattern_mae=0.079 max_ratio=1.057 real_max=13.9 gen_max=14.7 peak_err=5.8px
  270.0MHz K7 s0: r=0.791 Δmax=-2.92Ω pattern_mae=0.072 max_ratio=1.347 real_max=8.42 gen_max=11.3 peak_err=88.4px
  270.0MHz K8 s0: r=0.865 Δmax=-1.24Ω pattern_mae=0.067 max_ratio=1.084 real_max=14.8 gen_max=16 peak_err=62.0px
  270.0MHz K9 s0: r=0.815 Δmax=+25.1Ω pattern_mae=0.076 max_ratio=0.414 real_max=42.8 gen_max=17.7 peak_err=3.2px
  270.0MHz K10 s0: r=0.582 Δmax=+2.4Ω pattern_mae=0.087 max_ratio=0.847 real_max=15.6 gen_max=13.2 peak_err=88.4px
  270.0MHz K11 s0: r=0.522 Δmax=-10.9Ω pattern_mae=0.128 max_ratio=2.401 real_max=7.79 gen_max=18.7 peak_err=62.0px
  270.0MHz K12 s0: r=0.831 Δmax=-1.98Ω pattern_mae=0.092 max_ratio=1.171 real_max=11.6 gen_max=13.6 peak_err=63.0px
  270.0MHz K13 s0: r=0.804 Δmax=+19.5Ω pattern_mae=0.070 max_ratio=0.433 real_max=34.5 gen_max=14.9 peak_err=0.0px
  270.0MHz K14 s0: r=0.359 Δmax=-9.68Ω pattern_mae=0.128 max_ratio=2.090 real_max=8.89 gen_max=18.6 peak_err=58.1px
  270.0MHz K15 s0: r=0.535 Δmax=-11.3Ω pattern_mae=0.162 max_ratio=2.341 real_max=8.4 gen_max=19.7 peak_err=62.0px
  270.0MHz K16 s0: r=0.928 Δmax=+0.00842Ω pattern_mae=0.028 max_ratio=0.999 real_max=16.3 gen_max=16.3 peak_err=0.0px
  270.0MHz K17 s0: r=0.856 Δmax=+11.6Ω pattern_mae=0.058 max_ratio=0.617 real_max=30.2 gen_max=18.6 peak_err=86.3px
  270.0MHz K18 s0: r=0.931 Δmax=+33.1Ω pattern_mae=0.041 max_ratio=0.378 real_max=53.2 gen_max=20.1 peak_err=0.0px
  270.0MHz K19 s0: r=0.710 Δmax=+50.3Ω pattern_mae=0.092 max_ratio=0.299 real_max=71.8 gen_max=21.5 peak_err=63.0px
  270.0MHz K20 s0: r=0.943 Δmax=+1.37Ω pattern_mae=0.056 max_ratio=0.919 real_max=17 gen_max=15.6 peak_err=0.0px
  450.0MHz K2 s0: r=0.942 Δmax=+0.266Ω pattern_mae=0.024 max_ratio=0.976 real_max=11 gen_max=10.7 peak_err=0.0px
  450.0MHz K3 s0: r=0.933 Δmax=+0.172Ω pattern_mae=0.020 max_ratio=0.984 real_max=10.9 gen_max=10.8 peak_err=0.0px
  450.0MHz K4 s0: r=0.988 Δmax=+0.277Ω pattern_mae=0.012 max_ratio=0.975 real_max=10.9 gen_max=10.6 peak_err=0.0px
  450.0MHz K5 s0: r=0.969 Δmax=-0.666Ω pattern_mae=0.014 max_ratio=1.082 real_max=8.15 gen_max=8.81 peak_err=76.0px
  450.0MHz K6 s0: r=0.996 Δmax=-0.205Ω pattern_mae=0.000 max_ratio=1.021 real_max=9.95 gen_max=10.2 peak_err=0.0px
  450.0MHz K7 s0: r=0.782 Δmax=-2.92Ω pattern_mae=0.086 max_ratio=1.411 real_max=7.1 gen_max=10 peak_err=29.0px
  450.0MHz K8 s0: r=0.896 Δmax=-1.69Ω pattern_mae=0.053 max_ratio=1.235 real_max=7.17 gen_max=8.86 peak_err=29.0px
  450.0MHz K9 s0: r=0.479 Δmax=-1.94Ω pattern_mae=0.139 max_ratio=1.289 real_max=6.71 gen_max=8.64 peak_err=34.0px
  450.0MHz K10 s0: r=0.401 Δmax=+9.39Ω pattern_mae=0.123 max_ratio=0.514 real_max=19.3 gen_max=9.92 peak_err=63.0px
  450.0MHz K11 s0: r=0.973 Δmax=+2.7Ω pattern_mae=0.019 max_ratio=0.823 real_max=15.2 gen_max=12.5 peak_err=0.0px
  450.0MHz K12 s0: r=0.962 Δmax=+0.243Ω pattern_mae=0.013 max_ratio=0.967 real_max=7.33 gen_max=7.09 peak_err=70.2px
  450.0MHz K13 s0: r=0.903 Δmax=-0.241Ω pattern_mae=0.024 max_ratio=1.021 real_max=11.3 gen_max=11.6 peak_err=62.0px
  450.0MHz K14 s0: r=0.882 Δmax=+5.79Ω pattern_mae=0.032 max_ratio=0.598 real_max=14.4 gen_max=8.62 peak_err=76.0px
  450.0MHz K15 s0: r=0.928 Δmax=+1.37Ω pattern_mae=0.021 max_ratio=0.880 real_max=11.4 gen_max=10.1 peak_err=0.0px
  450.0MHz K16 s0: r=0.872 Δmax=+6.5Ω pattern_mae=0.034 max_ratio=0.618 real_max=17 gen_max=10.5 peak_err=0.0px
  450.0MHz K17 s0: r=0.947 Δmax=+1.97Ω pattern_mae=0.021 max_ratio=0.837 real_max=12.1 gen_max=10.1 peak_err=0.0px
  450.0MHz K18 s0: r=0.871 Δmax=+6.37Ω pattern_mae=0.059 max_ratio=0.621 real_max=16.8 gen_max=10.5 peak_err=63.0px
  450.0MHz K19 s0: r=0.796 Δmax=+0.0786Ω pattern_mae=0.033 max_ratio=0.993 real_max=10.8 gen_max=10.7 peak_err=63.0px
  450.0MHz K20 s0: r=0.925 Δmax=+4.16Ω pattern_mae=0.024 max_ratio=0.695 real_max=13.6 gen_max=9.48 peak_err=0.0px
  500.0MHz K2 s0: r=0.921 Δmax=+0.695Ω pattern_mae=0.037 max_ratio=0.947 real_max=13.2 gen_max=12.5 peak_err=60.2px
  500.0MHz K3 s0: r=0.773 Δmax=+5.45Ω pattern_mae=0.078 max_ratio=0.667 real_max=16.4 gen_max=10.9 peak_err=70.2px
  500.0MHz K4 s0: r=0.965 Δmax=-0.156Ω pattern_mae=0.015 max_ratio=1.015 real_max=10.5 gen_max=10.6 peak_err=0.0px
  500.0MHz K5 s0: r=0.809 Δmax=+4.81Ω pattern_mae=0.087 max_ratio=0.741 real_max=18.6 gen_max=13.7 peak_err=88.4px
  500.0MHz K6 s0: r=0.841 Δmax=+5.77Ω pattern_mae=0.078 max_ratio=0.665 real_max=17.2 gen_max=11.5 peak_err=63.0px
  500.0MHz K7 s0: r=0.977 Δmax=-0.194Ω pattern_mae=0.023 max_ratio=1.012 real_max=16 gen_max=16.2 peak_err=0.0px
  500.0MHz K8 s0: r=0.970 Δmax=+1.42Ω pattern_mae=0.027 max_ratio=0.908 real_max=15.5 gen_max=14 peak_err=0.0px
  500.0MHz K9 s0: r=0.846 Δmax=+11.7Ω pattern_mae=0.072 max_ratio=0.485 real_max=22.6 gen_max=11 peak_err=1.0px
  500.0MHz K10 s0: r=0.982 Δmax=-0.296Ω pattern_mae=0.009 max_ratio=1.023 real_max=13 gen_max=13.3 peak_err=1.0px
  500.0MHz K11 s0: r=0.861 Δmax=+0.88Ω pattern_mae=0.043 max_ratio=0.935 real_max=13.5 gen_max=12.6 peak_err=63.0px
  500.0MHz K12 s0: r=0.929 Δmax=+4.51Ω pattern_mae=0.029 max_ratio=0.713 real_max=15.7 gen_max=11.2 peak_err=32.0px
  500.0MHz K13 s0: r=0.390 Δmax=+1.33Ω pattern_mae=0.104 max_ratio=0.894 real_max=12.6 gen_max=11.2 peak_err=63.0px
  500.0MHz K14 s0: r=0.950 Δmax=+0.711Ω pattern_mae=0.024 max_ratio=0.941 real_max=12.2 gen_max=11.4 peak_err=0.0px
  500.0MHz K15 s0: r=0.756 Δmax=+3.09Ω pattern_mae=0.049 max_ratio=0.779 real_max=13.9 gen_max=10.9 peak_err=85.6px
  500.0MHz K16 s0: r=0.888 Δmax=-1.09Ω pattern_mae=0.050 max_ratio=1.101 real_max=10.8 gen_max=11.9 peak_err=0.0px
  500.0MHz K17 s0: r=0.953 Δmax=+1.1Ω pattern_mae=0.031 max_ratio=0.910 real_max=12.3 gen_max=11.2 peak_err=70.7px
  500.0MHz K18 s0: r=0.941 Δmax=-1.36Ω pattern_mae=0.051 max_ratio=1.136 real_max=9.96 gen_max=11.3 peak_err=0.0px
  500.0MHz K19 s0: r=0.786 Δmax=-1.15Ω pattern_mae=0.048 max_ratio=1.109 real_max=10.6 gen_max=11.7 peak_err=0.0px
  500.0MHz K20 s0: r=0.772 Δmax=+1.41Ω pattern_mae=0.056 max_ratio=0.877 real_max=11.4 gen_max=10 peak_err=0.0px
```
