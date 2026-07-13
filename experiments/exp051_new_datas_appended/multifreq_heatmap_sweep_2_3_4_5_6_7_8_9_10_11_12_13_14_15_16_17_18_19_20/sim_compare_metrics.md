# Simulated Real vs Generated — Heatmap Metrics

- **Rows**: 95 (one per freq × K × sample)
- **Source**: ECADStar `.map` in `K*/Real/` vs `data_sample_*/heatmap_physical.npy`
- **CSV**: `sim_compare_metrics.csv`
- **JSON**: `sim_compare_metrics.json`

## Per-sample metrics

```
mhz | k  | sample | mae_ohm  | rmse_ohm | pearson_r | pattern_mae | max_ratio | peak_loc_err_px
----|----|--------|----------|----------|-----------|-------------|-----------|----------------
10  | 2  | 0      | 0        | 0        | 0.7783    | 0.2028      | 0.5875    | 88.39          
10  | 3  | 0      | 0        | 0        | 0.8397    | 0.2234      | 0.5638    | 68.94          
10  | 4  | 0      | 0        | 0        | 0.9251    | 0.09452     | 0.543     | 63             
10  | 5  | 0      | 0        | 0        | 0.9503    | 0.09673     | 0.5558    | 63             
10  | 6  | 0      | 0        | 0        | 0.9333    | 0.1243      | 0.5832    | 0              
10  | 7  | 0      | 0        | 0        | 0.9371    | 0.1247      | 0.6101    | 0              
10  | 8  | 0      | 0        | 0        | 0.9312    | 0.1103      | 0.5712    | 0              
10  | 9  | 0      | 0        | 0        | 0.9527    | 0.08336     | 0.5689    | 0              
10  | 10 | 0      | 0        | 0        | 0.9139    | 0.1002      | 0.5942    | 0              
10  | 11 | 0      | 0        | 0        | 0.9334    | 0.05842     | 0.5695    | 0              
10  | 12 | 0      | 0        | 0        | 0.9474    | 0.05469     | 0.5345    | 0              
10  | 13 | 0      | 0        | 0        | 0.9354    | 0.0588      | 0.5199    | 0              
10  | 14 | 0      | 0        | 0        | 0.8415    | 0.1417      | 0.5329    | 63             
10  | 15 | 0      | 0        | 0        | 0.9516    | 0.04994     | 0.5723    | 0              
10  | 16 | 0      | 0        | 0        | 0.9397    | 0.05134     | 0.5707    | 63             
10  | 17 | 0      | 0        | 0        | 0.8952    | 0.05983     | 0.5808    | 63             
10  | 18 | 0      | 0        | 0        | 0.8692    | 0.04642     | 0.5406    | 63             
10  | 19 | 0      | 0        | 0        | 0.956     | 0.05082     | 0.5921    | 0              
10  | 20 | 0      | 0        | 0        | 0.8436    | 0.05076     | 0.6083    | 0              
70  | 2  | 0      | 0.8151   | 1.039    | 0.3674    | 0.2269      | 0.576     | 88.39          
70  | 3  | 0      | 0.1719   | 0.2853   | 0.8729    | 0.1308      | 0.8063    | 9              
70  | 4  | 0      | 0.05329  | 0.1316   | 0.8543    | 0.07182     | 0.6827    | 0              
70  | 5  | 0      | 0.02274  | 0.09001  | 0.9563    | 0.04257     | 0.7295    | 63             
70  | 6  | 0      | 0.01398  | 0.07005  | 0.9503    | 0.04573     | 0.7307    | 0              
70  | 7  | 0      | 0.09077  | 0.1992   | 0.9076    | 0.05287     | 0.7081    | 0              
70  | 8  | 0      | 0.02104  | 0.08583  | 0.981     | 0.01341     | 0.687     | 0              
70  | 9  | 0      | 0.007317 | 0.05063  | 0.9686    | 0.01016     | 0.7386    | 0              
70  | 10 | 0      | 0.02304  | 0.09318  | 0.9545    | 0.02433     | 0.7065    | 0              
70  | 11 | 0      | 0.05018  | 0.1375   | 0.9265    | 0.02786     | 0.7224    | 63             
70  | 12 | 0      | 0.0326   | 0.1095   | 0.9395    | 0.02082     | 0.7081    | 0              
70  | 13 | 0      | 0.06491  | 0.1635   | 0.9488    | 0.03046     | 0.6313    | 0              
70  | 14 | 0      | 0.02009  | 0.09106  | 0.9339    | 0.05503     | 0.7032    | 63             
70  | 15 | 0      | 0.01271  | 0.06662  | 0.9795    | 0.009597    | 0.7071    | 0              
70  | 16 | 0      | 0.02734  | 0.09594  | 0.9653    | 0.01143     | 0.6848    | 0              
70  | 17 | 0      | 0.04208  | 0.1307   | 0.8988    | 0.02877     | 0.7568    | 63             
70  | 18 | 0      | 0.03622  | 0.1182   | 0.9412    | 0.01935     | 0.6806    | 63             
70  | 19 | 0      | 0.01849  | 0.0781   | 0.9597    | 0.01275     | 0.7344    | 0              
70  | 20 | 0      | 0.03132  | 0.1094   | 0.913     | 0.02283     | 0.7405    | 0              
120 | 2  | 0      | 1.458    | 1.589    | 0.7081    | 0.1596      | 1.281     | 63.15          
120 | 3  | 0      | 1.574    | 1.881    | 0.2654    | 0.201       | 1.694     | 8.062          
120 | 4  | 0      | 1.935    | 2.268    | 0.02019   | 0.3592      | 0.7032    | 71.06          
120 | 5  | 0      | 0.9348   | 1.083    | 0.6633    | 0.1527      | 0.5513    | 58.86          
120 | 6  | 0      | 8.849    | 11.18    | 0.05983   | 0.3449      | 0.1318    | 87.68          
120 | 7  | 0      | 3.384    | 4.417    | 0.3267    | 0.3059      | 0.2952    | 75.17          
120 | 8  | 0      | 0.6824   | 0.803    | 0.7945    | 0.08469     | 0.5727    | 10.05          
120 | 9  | 0      | 0.4584   | 0.5785   | 0.8949    | 0.07736     | 0.4778    | 8              
120 | 10 | 0      | 0.5146   | 0.6531   | 0.8421    | 0.06779     | 0.5133    | 17             
120 | 11 | 0      | 0.515    | 0.9877   | 0.591     | 0.1093      | 1.098     | 12.21          
120 | 12 | 0      | 0.4534   | 0.9146   | 0.5746    | 0.1135      | 1.099     | 14.32          
120 | 13 | 0      | 0.8228   | 1.111    | 0.8572    | 0.07563     | 0.4946    | 63.15          
120 | 14 | 0      | 0.4528   | 0.8545   | 0.6935    | 0.1137      | 0.8204    | 61.66          
120 | 15 | 0      | 0.1854   | 0.3664   | 0.7916    | 0.06        | 0.6884    | 11.05          
120 | 16 | 0      | 0.2269   | 0.4283   | 0.7115    | 0.06783     | 0.7249    | 8.246          
120 | 17 | 0      | 0.2349   | 0.4807   | 0.6832    | 0.07224     | 0.7759    | 9.22           
120 | 18 | 0      | 0.1828   | 0.3691   | 0.8075    | 0.06589     | 0.5628    | 63             
120 | 19 | 0      | 0.2714   | 0.5602   | 0.6492    | 0.07643     | 0.8385    | 9.22           
120 | 20 | 0      | 0.3451   | 0.5398   | 0.8486    | 0.05154     | 0.5089    | 0              
270 | 2  | 0      | 3.084    | 4.015    | 0.8638    | 0.09175     | 1.028     | 62.29          
270 | 3  | 0      | 8.513    | 12.36    | 0.2892    | 0.2542      | 2.33      | 67.78          
270 | 4  | 0      | 3.944    | 7.015    | 0.2995    | 0.1405      | 1.166     | 63.35          
270 | 5  | 0      | 6.607    | 9.882    | 0.3765    | 0.2212      | 0.8512    | 35             
270 | 6  | 0      | 3.499    | 6.089    | 0.1685    | 0.1579      | 1.808     | 53.46          
270 | 7  | 0      | 4.248    | 8.032    | 0.3077    | 0.1618      | 1.983     | 60.41          
270 | 8  | 0      | 2.525    | 4.454    | 0.3273    | 0.1524      | 1.384     | 57.43          
270 | 9  | 0      | 3.848    | 6.467    | 0.5048    | 0.1375      | 0.7431    | 53.24          
270 | 10 | 0      | 3.361    | 5.83     | 0.07682   | 0.1458      | 1.22      | 81.57          
270 | 11 | 0      | 6.667    | 10.53    | 0.1284    | 0.224       | 1.374     | 65.3           
270 | 12 | 0      | 6.272    | 10.27    | -0.109    | 0.2185      | 2.829     | 34             
270 | 13 | 0      | 4.192    | 7.039    | 0.5666    | 0.1134      | 0.8432    | 74.06          
270 | 14 | 0      | 7.945    | 11.99    | -0.06829  | 0.3214      | 4.974     | 38             
270 | 15 | 0      | 2.56     | 4.384    | 0.3799    | 0.11        | 1.045     | 53             
270 | 16 | 0      | 6.659    | 11.16    | 0.1623    | 0.1691      | 0.5072    | 61.52          
270 | 17 | 0      | 4.486    | 7.938    | 0.3942    | 0.1747      | 3.072     | 40             
270 | 18 | 0      | 1.876    | 2.892    | 0.8455    | 0.08529     | 1.499     | 7              
270 | 19 | 0      | 4.524    | 7.952    | 0.1036    | 0.1793      | 1.695     | 67.07          
270 | 20 | 0      | 1.772    | 3.54     | 0.8726    | 0.04511     | 0.4657    | 0              
400 | 2  | 0      | 1.817    | 2.491    | 0.7296    | 0.08474     | 0.8115    | 88.39          
400 | 3  | 0      | 1.576    | 2.109    | 0.7331    | 0.0622      | 1.464     | 0              
400 | 4  | 0      | 2.772    | 3.781    | 0.5575    | 0.1319      | 0.791     | 63             
400 | 5  | 0      | 2.761    | 3.912    | 0.5255    | 0.1186      | 0.4844    | 63             
400 | 6  | 0      | 1.992    | 3.62     | 0.1734    | 0.1095      | 1.705     | 61.81          
400 | 7  | 0      | 1.617    | 2.331    | 0.512     | 0.07271     | 1.386     | 62             
400 | 8  | 0      | 0.6042   | 0.9812   | 0.8221    | 0.03785     | 0.9534    | 63             
400 | 9  | 0      | 0.8522   | 1.44     | 0.7365    | 0.05603     | 1.031     | 63             
400 | 10 | 0      | 1.545    | 2.145    | 0.573     | 0.06777     | 1.317     | 62             
400 | 11 | 0      | 3.196    | 5.383    | 0.06196   | 0.1507      | 2.064     | 65.79          
400 | 12 | 0      | 2.294    | 4.195    | 0.2033    | 0.1432      | 2.025     | 11.05          
400 | 13 | 0      | 1.821    | 2.822    | 0.2488    | 0.1034      | 1.512     | 47.04          
400 | 14 | 0      | 3.524    | 5.632    | 0.03592   | 0.1792      | 1.68      | 28             
400 | 15 | 0      | 1.255    | 2.115    | 0.3061    | 0.1132      | 1.277     | 0              
400 | 16 | 0      | 1.39     | 2.117    | 0.4392    | 0.1076      | 1.39      | 0              
400 | 17 | 0      | 3.778    | 5.85     | -0.1937   | 0.1984      | 1.026     | 68.94          
400 | 18 | 0      | 1.638    | 2.311    | 0.3769    | 0.107       | 1.232     | 62             
400 | 19 | 0      | 3.654    | 5.233    | 0.02108   | 0.2006      | 1.066     | 75.17          
400 | 20 | 0      | 1.216    | 1.778    | 0.4185    | 0.166       | 1.45      | 88.39          
```

## Mean by MHz

```
mhz | n  | mae_ohm_mean | rmse_ohm_mean | pearson_r_mean | pattern_mae_mean | max_ratio_mean
----|----|--------------|---------------|----------------|------------------|---------------
10  | 19 | 0            | 0             | 0.9092         | 0.09384          | 0.5684        
70  | 19 | 0.08185      | 0.1655        | 0.9063         | 0.04513          | 0.7071        
120 | 19 | 1.236        | 1.635         | 0.6201         | 0.1347           | 0.728         
270 | 19 | 4.557        | 7.465         | 0.3416         | 0.1634           | 1.622         
400 | 19 | 2.068        | 3.171         | 0.3832         | 0.1163           | 1.298         
```

## Agent copy block

```
sim_compare_metrics: n=95
  10.0MHz K2 s0: mae=0Ω rmse=0Ω r=0.778 pattern_mae=0.203 max_ratio=0.587 peak_err=88.4px
  10.0MHz K3 s0: mae=0Ω rmse=0Ω r=0.840 pattern_mae=0.223 max_ratio=0.564 peak_err=68.9px
  10.0MHz K4 s0: mae=0Ω rmse=0Ω r=0.925 pattern_mae=0.095 max_ratio=0.543 peak_err=63.0px
  10.0MHz K5 s0: mae=0Ω rmse=0Ω r=0.950 pattern_mae=0.097 max_ratio=0.556 peak_err=63.0px
  10.0MHz K6 s0: mae=0Ω rmse=0Ω r=0.933 pattern_mae=0.124 max_ratio=0.583 peak_err=0.0px
  10.0MHz K7 s0: mae=0Ω rmse=0Ω r=0.937 pattern_mae=0.125 max_ratio=0.610 peak_err=0.0px
  10.0MHz K8 s0: mae=0Ω rmse=0Ω r=0.931 pattern_mae=0.110 max_ratio=0.571 peak_err=0.0px
  10.0MHz K9 s0: mae=0Ω rmse=0Ω r=0.953 pattern_mae=0.083 max_ratio=0.569 peak_err=0.0px
  10.0MHz K10 s0: mae=0Ω rmse=0Ω r=0.914 pattern_mae=0.100 max_ratio=0.594 peak_err=0.0px
  10.0MHz K11 s0: mae=0Ω rmse=0Ω r=0.933 pattern_mae=0.058 max_ratio=0.570 peak_err=0.0px
  10.0MHz K12 s0: mae=0Ω rmse=0Ω r=0.947 pattern_mae=0.055 max_ratio=0.534 peak_err=0.0px
  10.0MHz K13 s0: mae=0Ω rmse=0Ω r=0.935 pattern_mae=0.059 max_ratio=0.520 peak_err=0.0px
  10.0MHz K14 s0: mae=0Ω rmse=0Ω r=0.842 pattern_mae=0.142 max_ratio=0.533 peak_err=63.0px
  10.0MHz K15 s0: mae=0Ω rmse=0Ω r=0.952 pattern_mae=0.050 max_ratio=0.572 peak_err=0.0px
  10.0MHz K16 s0: mae=0Ω rmse=0Ω r=0.940 pattern_mae=0.051 max_ratio=0.571 peak_err=63.0px
  10.0MHz K17 s0: mae=0Ω rmse=0Ω r=0.895 pattern_mae=0.060 max_ratio=0.581 peak_err=63.0px
  10.0MHz K18 s0: mae=0Ω rmse=0Ω r=0.869 pattern_mae=0.046 max_ratio=0.541 peak_err=63.0px
  10.0MHz K19 s0: mae=0Ω rmse=0Ω r=0.956 pattern_mae=0.051 max_ratio=0.592 peak_err=0.0px
  10.0MHz K20 s0: mae=0Ω rmse=0Ω r=0.844 pattern_mae=0.051 max_ratio=0.608 peak_err=0.0px
  70.0MHz K2 s0: mae=0.8151Ω rmse=1.039Ω r=0.367 pattern_mae=0.227 max_ratio=0.576 peak_err=88.4px
  70.0MHz K3 s0: mae=0.1719Ω rmse=0.2853Ω r=0.873 pattern_mae=0.131 max_ratio=0.806 peak_err=9.0px
  70.0MHz K4 s0: mae=0.05329Ω rmse=0.1316Ω r=0.854 pattern_mae=0.072 max_ratio=0.683 peak_err=0.0px
  70.0MHz K5 s0: mae=0.02274Ω rmse=0.09001Ω r=0.956 pattern_mae=0.043 max_ratio=0.730 peak_err=63.0px
  70.0MHz K6 s0: mae=0.01398Ω rmse=0.07005Ω r=0.950 pattern_mae=0.046 max_ratio=0.731 peak_err=0.0px
  70.0MHz K7 s0: mae=0.09077Ω rmse=0.1992Ω r=0.908 pattern_mae=0.053 max_ratio=0.708 peak_err=0.0px
  70.0MHz K8 s0: mae=0.02104Ω rmse=0.08583Ω r=0.981 pattern_mae=0.013 max_ratio=0.687 peak_err=0.0px
  70.0MHz K9 s0: mae=0.007317Ω rmse=0.05063Ω r=0.969 pattern_mae=0.010 max_ratio=0.739 peak_err=0.0px
  70.0MHz K10 s0: mae=0.02304Ω rmse=0.09318Ω r=0.954 pattern_mae=0.024 max_ratio=0.707 peak_err=0.0px
  70.0MHz K11 s0: mae=0.05018Ω rmse=0.1375Ω r=0.926 pattern_mae=0.028 max_ratio=0.722 peak_err=63.0px
  70.0MHz K12 s0: mae=0.0326Ω rmse=0.1095Ω r=0.939 pattern_mae=0.021 max_ratio=0.708 peak_err=0.0px
  70.0MHz K13 s0: mae=0.06491Ω rmse=0.1635Ω r=0.949 pattern_mae=0.030 max_ratio=0.631 peak_err=0.0px
  70.0MHz K14 s0: mae=0.02009Ω rmse=0.09106Ω r=0.934 pattern_mae=0.055 max_ratio=0.703 peak_err=63.0px
  70.0MHz K15 s0: mae=0.01271Ω rmse=0.06662Ω r=0.980 pattern_mae=0.010 max_ratio=0.707 peak_err=0.0px
  70.0MHz K16 s0: mae=0.02734Ω rmse=0.09594Ω r=0.965 pattern_mae=0.011 max_ratio=0.685 peak_err=0.0px
  70.0MHz K17 s0: mae=0.04208Ω rmse=0.1307Ω r=0.899 pattern_mae=0.029 max_ratio=0.757 peak_err=63.0px
  70.0MHz K18 s0: mae=0.03622Ω rmse=0.1182Ω r=0.941 pattern_mae=0.019 max_ratio=0.681 peak_err=63.0px
  70.0MHz K19 s0: mae=0.01849Ω rmse=0.0781Ω r=0.960 pattern_mae=0.013 max_ratio=0.734 peak_err=0.0px
  70.0MHz K20 s0: mae=0.03132Ω rmse=0.1094Ω r=0.913 pattern_mae=0.023 max_ratio=0.740 peak_err=0.0px
  120.0MHz K2 s0: mae=1.458Ω rmse=1.589Ω r=0.708 pattern_mae=0.160 max_ratio=1.281 peak_err=63.2px
  120.0MHz K3 s0: mae=1.574Ω rmse=1.881Ω r=0.265 pattern_mae=0.201 max_ratio=1.694 peak_err=8.1px
  120.0MHz K4 s0: mae=1.935Ω rmse=2.268Ω r=0.020 pattern_mae=0.359 max_ratio=0.703 peak_err=71.1px
  120.0MHz K5 s0: mae=0.9348Ω rmse=1.083Ω r=0.663 pattern_mae=0.153 max_ratio=0.551 peak_err=58.9px
  120.0MHz K6 s0: mae=8.849Ω rmse=11.18Ω r=0.060 pattern_mae=0.345 max_ratio=0.132 peak_err=87.7px
  120.0MHz K7 s0: mae=3.384Ω rmse=4.417Ω r=0.327 pattern_mae=0.306 max_ratio=0.295 peak_err=75.2px
  120.0MHz K8 s0: mae=0.6824Ω rmse=0.803Ω r=0.795 pattern_mae=0.085 max_ratio=0.573 peak_err=10.0px
  120.0MHz K9 s0: mae=0.4584Ω rmse=0.5785Ω r=0.895 pattern_mae=0.077 max_ratio=0.478 peak_err=8.0px
  120.0MHz K10 s0: mae=0.5146Ω rmse=0.6531Ω r=0.842 pattern_mae=0.068 max_ratio=0.513 peak_err=17.0px
  120.0MHz K11 s0: mae=0.515Ω rmse=0.9877Ω r=0.591 pattern_mae=0.109 max_ratio=1.098 peak_err=12.2px
  120.0MHz K12 s0: mae=0.4534Ω rmse=0.9146Ω r=0.575 pattern_mae=0.113 max_ratio=1.099 peak_err=14.3px
  120.0MHz K13 s0: mae=0.8228Ω rmse=1.111Ω r=0.857 pattern_mae=0.076 max_ratio=0.495 peak_err=63.2px
  120.0MHz K14 s0: mae=0.4528Ω rmse=0.8545Ω r=0.693 pattern_mae=0.114 max_ratio=0.820 peak_err=61.7px
  120.0MHz K15 s0: mae=0.1854Ω rmse=0.3664Ω r=0.792 pattern_mae=0.060 max_ratio=0.688 peak_err=11.0px
  120.0MHz K16 s0: mae=0.2269Ω rmse=0.4283Ω r=0.711 pattern_mae=0.068 max_ratio=0.725 peak_err=8.2px
  120.0MHz K17 s0: mae=0.2349Ω rmse=0.4807Ω r=0.683 pattern_mae=0.072 max_ratio=0.776 peak_err=9.2px
  120.0MHz K18 s0: mae=0.1828Ω rmse=0.3691Ω r=0.807 pattern_mae=0.066 max_ratio=0.563 peak_err=63.0px
  120.0MHz K19 s0: mae=0.2714Ω rmse=0.5602Ω r=0.649 pattern_mae=0.076 max_ratio=0.838 peak_err=9.2px
  120.0MHz K20 s0: mae=0.3451Ω rmse=0.5398Ω r=0.849 pattern_mae=0.052 max_ratio=0.509 peak_err=0.0px
  270.0MHz K2 s0: mae=3.084Ω rmse=4.015Ω r=0.864 pattern_mae=0.092 max_ratio=1.028 peak_err=62.3px
  270.0MHz K3 s0: mae=8.513Ω rmse=12.36Ω r=0.289 pattern_mae=0.254 max_ratio=2.330 peak_err=67.8px
  270.0MHz K4 s0: mae=3.944Ω rmse=7.015Ω r=0.299 pattern_mae=0.141 max_ratio=1.166 peak_err=63.3px
  270.0MHz K5 s0: mae=6.607Ω rmse=9.882Ω r=0.376 pattern_mae=0.221 max_ratio=0.851 peak_err=35.0px
  270.0MHz K6 s0: mae=3.499Ω rmse=6.089Ω r=0.168 pattern_mae=0.158 max_ratio=1.808 peak_err=53.5px
  270.0MHz K7 s0: mae=4.248Ω rmse=8.032Ω r=0.308 pattern_mae=0.162 max_ratio=1.983 peak_err=60.4px
  270.0MHz K8 s0: mae=2.525Ω rmse=4.454Ω r=0.327 pattern_mae=0.152 max_ratio=1.384 peak_err=57.4px
  270.0MHz K9 s0: mae=3.848Ω rmse=6.467Ω r=0.505 pattern_mae=0.137 max_ratio=0.743 peak_err=53.2px
  270.0MHz K10 s0: mae=3.361Ω rmse=5.83Ω r=0.077 pattern_mae=0.146 max_ratio=1.220 peak_err=81.6px
  270.0MHz K11 s0: mae=6.667Ω rmse=10.53Ω r=0.128 pattern_mae=0.224 max_ratio=1.374 peak_err=65.3px
  270.0MHz K12 s0: mae=6.272Ω rmse=10.27Ω r=-0.109 pattern_mae=0.219 max_ratio=2.829 peak_err=34.0px
  270.0MHz K13 s0: mae=4.192Ω rmse=7.039Ω r=0.567 pattern_mae=0.113 max_ratio=0.843 peak_err=74.1px
  270.0MHz K14 s0: mae=7.945Ω rmse=11.99Ω r=-0.068 pattern_mae=0.321 max_ratio=4.974 peak_err=38.0px
  270.0MHz K15 s0: mae=2.56Ω rmse=4.384Ω r=0.380 pattern_mae=0.110 max_ratio=1.045 peak_err=53.0px
  270.0MHz K16 s0: mae=6.659Ω rmse=11.16Ω r=0.162 pattern_mae=0.169 max_ratio=0.507 peak_err=61.5px
  270.0MHz K17 s0: mae=4.486Ω rmse=7.938Ω r=0.394 pattern_mae=0.175 max_ratio=3.072 peak_err=40.0px
  270.0MHz K18 s0: mae=1.876Ω rmse=2.892Ω r=0.845 pattern_mae=0.085 max_ratio=1.499 peak_err=7.0px
  270.0MHz K19 s0: mae=4.524Ω rmse=7.952Ω r=0.104 pattern_mae=0.179 max_ratio=1.695 peak_err=67.1px
  270.0MHz K20 s0: mae=1.772Ω rmse=3.54Ω r=0.873 pattern_mae=0.045 max_ratio=0.466 peak_err=0.0px
  400.0MHz K2 s0: mae=1.817Ω rmse=2.491Ω r=0.730 pattern_mae=0.085 max_ratio=0.811 peak_err=88.4px
  400.0MHz K3 s0: mae=1.576Ω rmse=2.109Ω r=0.733 pattern_mae=0.062 max_ratio=1.464 peak_err=0.0px
  400.0MHz K4 s0: mae=2.772Ω rmse=3.781Ω r=0.558 pattern_mae=0.132 max_ratio=0.791 peak_err=63.0px
  400.0MHz K5 s0: mae=2.761Ω rmse=3.912Ω r=0.526 pattern_mae=0.119 max_ratio=0.484 peak_err=63.0px
  400.0MHz K6 s0: mae=1.992Ω rmse=3.62Ω r=0.173 pattern_mae=0.110 max_ratio=1.705 peak_err=61.8px
  400.0MHz K7 s0: mae=1.617Ω rmse=2.331Ω r=0.512 pattern_mae=0.073 max_ratio=1.386 peak_err=62.0px
  400.0MHz K8 s0: mae=0.6042Ω rmse=0.9812Ω r=0.822 pattern_mae=0.038 max_ratio=0.953 peak_err=63.0px
  400.0MHz K9 s0: mae=0.8522Ω rmse=1.44Ω r=0.736 pattern_mae=0.056 max_ratio=1.031 peak_err=63.0px
  400.0MHz K10 s0: mae=1.545Ω rmse=2.145Ω r=0.573 pattern_mae=0.068 max_ratio=1.317 peak_err=62.0px
  400.0MHz K11 s0: mae=3.196Ω rmse=5.383Ω r=0.062 pattern_mae=0.151 max_ratio=2.064 peak_err=65.8px
  400.0MHz K12 s0: mae=2.294Ω rmse=4.195Ω r=0.203 pattern_mae=0.143 max_ratio=2.025 peak_err=11.0px
  400.0MHz K13 s0: mae=1.821Ω rmse=2.822Ω r=0.249 pattern_mae=0.103 max_ratio=1.512 peak_err=47.0px
  400.0MHz K14 s0: mae=3.524Ω rmse=5.632Ω r=0.036 pattern_mae=0.179 max_ratio=1.680 peak_err=28.0px
  400.0MHz K15 s0: mae=1.255Ω rmse=2.115Ω r=0.306 pattern_mae=0.113 max_ratio=1.277 peak_err=0.0px
  400.0MHz K16 s0: mae=1.39Ω rmse=2.117Ω r=0.439 pattern_mae=0.108 max_ratio=1.390 peak_err=0.0px
  400.0MHz K17 s0: mae=3.778Ω rmse=5.85Ω r=-0.194 pattern_mae=0.198 max_ratio=1.026 peak_err=68.9px
  400.0MHz K18 s0: mae=1.638Ω rmse=2.311Ω r=0.377 pattern_mae=0.107 max_ratio=1.232 peak_err=62.0px
  400.0MHz K19 s0: mae=3.654Ω rmse=5.233Ω r=0.021 pattern_mae=0.201 max_ratio=1.066 peak_err=75.2px
  400.0MHz K20 s0: mae=1.216Ω rmse=1.778Ω r=0.419 pattern_mae=0.166 max_ratio=1.450 peak_err=88.4px
```
