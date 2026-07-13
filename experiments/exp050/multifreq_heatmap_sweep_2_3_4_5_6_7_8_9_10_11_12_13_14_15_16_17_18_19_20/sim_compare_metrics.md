# Simulated Real vs Generated — Heatmap Metrics

- **Rows**: 95 (one per freq × K × sample)
- **Source**: ECADStar `.map` in `K*/Real/` vs `data_sample_*/heatmap_physical.npy`
- **CSV**: `sim_compare_metrics.csv`
- **JSON**: `sim_compare_metrics.json`

## Per-sample metrics

```
mhz | k  | sample | mae_ohm  | rmse_ohm | pearson_r | pattern_mae | max_ratio | peak_loc_err_px
----|----|--------|----------|----------|-----------|-------------|-----------|----------------
10  | 2  | 0      | 0        | 0        | 0.8377    | 0.1387      | 0.8078    | 0              
10  | 3  | 0      | 0        | 0        | 0.841     | 0.1259      | 0.7861    | 0              
10  | 4  | 0      | 0        | 0        | 0.7633    | 0.08162     | 0.8767    | 63             
10  | 5  | 0      | 0        | 0        | 0.7756    | 0.06623     | 0.8901    | 0              
10  | 6  | 0      | 0        | 0        | 0.8515    | 0.03661     | 0.8854    | 63             
10  | 7  | 0      | 0        | 0        | 0.9582    | 0.01202     | 1.02      | 0              
10  | 8  | 0      | 0        | 0        | 0.676     | 0.04365     | 1.088     | 63             
10  | 9  | 0      | 0        | 0        | 0.7946    | 0.04881     | 1.065     | 88.39          
10  | 10 | 0      | 0        | 0        | 0.8243    | 0.05097     | 1.146     | 88.39          
10  | 11 | 0      | 0        | 0        | 0.9479    | 0.01293     | 1.027     | 0              
10  | 12 | 0      | 0        | 0        | 0.9822    | 0.01114     | 1.029     | 0              
10  | 13 | 0      | 0        | 0        | 0.9585    | 0.01439     | 0.986     | 0              
10  | 14 | 0      | 0        | 0        | 0.8558    | 0.05785     | 0.7547    | 0              
10  | 15 | 0      | 0        | 0        | 0.851     | 0.02683     | 0.7857    | 0              
10  | 16 | 0      | 0        | 0        | 0.894     | 0.03596     | 0.708     | 0              
10  | 17 | 0      | 0        | 0        | 0.7912    | 0.04325     | 0.7843    | 63             
10  | 18 | 0      | 0        | 0        | 0.9539    | 0.01128     | 0.9573    | 0              
10  | 19 | 0      | 0        | 0        | 0.8432    | 0.03671     | 1.021     | 0              
10  | 20 | 0      | 0        | 0        | 0.8929    | 0.02891     | 0.9205    | 63             
70  | 2  | 0      | 0.9993   | 1.174    | 0.3564    | 0.3456      | 0.9528    | 55.97          
70  | 3  | 0      | 3.427    | 3.785    | 0.517     | 0.3455      | 0.3747    | 53.94          
70  | 4  | 0      | 1.017    | 1.181    | 0.6133    | 0.1947      | 0.485     | 63             
70  | 5  | 0      | 0.02707  | 0.09227  | 0.9848    | 0.03344     | 0.7362    | 0              
70  | 6  | 0      | 0.1805   | 0.3334   | 0.6915    | 0.08809     | 0.6032    | 63             
70  | 7  | 0      | 0.06338  | 0.1681   | 0.9181    | 0.03243     | 0.7922    | 0              
70  | 8  | 0      | 0.06549  | 0.1646   | 0.6918    | 0.06268     | 0.827     | 0              
70  | 9  | 0      | 0.005874 | 0.04015  | 0.8971    | 0.0862      | 0.8813    | 62             
70  | 10 | 0      | 0.01272  | 0.06023  | 0.8438    | 0.07632     | 0.8486    | 62             
70  | 11 | 0      | 0.01022  | 0.05604  | 0.958     | 0.01684     | 0.7511    | 0              
70  | 12 | 0      | 0.00861  | 0.05294  | 0.9866    | 0.005487    | 0.7839    | 0              
70  | 13 | 0      | 0.03384  | 0.1062   | 0.9754    | 0.003045    | 0.7474    | 0              
70  | 14 | 0      | 0.3546   | 0.5171   | 0.8048    | 0.09757     | 0.5713    | 0              
70  | 15 | 0      | 0.04131  | 0.1332   | 0.8166    | 0.04617     | 0.6753    | 0              
70  | 16 | 0      | 0.03664  | 0.1117   | 0.9576    | 0.01356     | 0.7293    | 0              
70  | 17 | 0      | 0.0725   | 0.1885   | 0.8799    | 0.03054     | 0.6422    | 0              
70  | 18 | 0      | 0.008795 | 0.05414  | 0.9813    | 0.007499    | 0.7816    | 0              
70  | 19 | 0      | 0.04071  | 0.1249   | 0.9216    | 0.02451     | 0.7625    | 0              
70  | 20 | 0      | 0.07695  | 0.1798   | 0.911     | 0.03047     | 0.7112    | 63             
120 | 2  | 0      | 0.9109   | 1.088    | 0.7434    | 0.09986     | 1.458     | 0              
120 | 3  | 0      | 1.128    | 1.287    | 0.6539    | 0.1801      | 1.267     | 19             
120 | 4  | 0      | 0.2958   | 0.4715   | 0.5684    | 0.08196     | 0.7089    | 0              
120 | 5  | 0      | 1.8      | 1.978    | 0.7114    | 0.1431      | 0.3773    | 19             
120 | 6  | 0      | 1.391    | 1.708    | 0.4121    | 0.1335      | 0.337     | 63             
120 | 7  | 0      | 0.9552   | 1.064    | 0.9206    | 0.0558      | 0.478     | 0              
120 | 8  | 0      | 0.802    | 0.8868   | 0.4105    | 0.114       | 0.5213    | 44             
120 | 9  | 0      | 0.7205   | 0.7463   | 0.7749    | 0.1305      | 0.5371    | 62             
120 | 10 | 0      | 0.4782   | 0.5375   | 0.7563    | 0.1354      | 0.7694    | 64.85          
120 | 11 | 0      | 0.06915  | 0.1699   | 0.9239    | 0.03811     | 0.7204    | 0              
120 | 12 | 0      | 0.1191   | 0.2397   | 0.9468    | 0.02069     | 0.7959    | 84.22          
120 | 13 | 0      | 0.6177   | 0.7017   | 0.9655    | 0.01043     | 0.512     | 0              
120 | 14 | 0      | 6.349    | 8.925    | 0.6217    | 0.3025      | 0.1443    | 6.403          
120 | 15 | 0      | 0.2696   | 0.3355   | 0.8741    | 0.03388     | 0.6244    | 0              
120 | 16 | 0      | 0.3221   | 0.5158   | 0.8759    | 0.02858     | 0.5044    | 0              
120 | 17 | 0      | 0.5186   | 0.7006   | 0.8155    | 0.04273     | 0.4234    | 63             
120 | 18 | 0      | 0.2067   | 0.3211   | 0.972     | 0.0136      | 0.7138    | 1              
120 | 19 | 0      | 0.2502   | 0.4121   | 0.912     | 0.01999     | 0.5573    | 0              
120 | 20 | 0      | 0.3696   | 0.5724   | 0.888     | 0.03789     | 0.5046    | 63             
270 | 2  | 0      | 5.179    | 8.681    | 0.3254    | 0.1869      | 6.762     | 52.09          
270 | 3  | 0      | 3.775    | 6.065    | 0.4335    | 0.1448      | 3.057     | 56             
270 | 4  | 0      | 0.8758   | 1.35     | 0.742     | 0.1056      | 1.398     | 56.08          
270 | 5  | 0      | 1.499    | 2.228    | 0.6275    | 0.1195      | 0.886     | 83.45          
270 | 6  | 0      | 1.516    | 2.561    | 0.5268    | 0.08971     | 0.6417    | 88.39          
270 | 7  | 0      | 5.238    | 10.62    | 0.4854    | 0.1198      | 0.2498    | 52             
270 | 8  | 0      | 1.143    | 1.812    | 0.4477    | 0.1161      | 1.113     | 56.08          
270 | 9  | 0      | 1.1      | 1.427    | 0.7845    | 0.09262     | 0.8532    | 62             
270 | 10 | 0      | 0.4877   | 0.7094   | 0.8428    | 0.1014      | 1.274     | 63.89          
270 | 11 | 0      | 4.786    | 9.196    | 0.8467    | 0.06081     | 0.405     | 62.07          
270 | 12 | 0      | 3.072    | 5.28     | 0.302     | 0.1034      | 1.289     | 62.39          
270 | 13 | 0      | 2.493    | 4.135    | 0.76      | 0.07061     | 0.5313    | 44             
270 | 14 | 0      | 0.669    | 1.277    | 0.8401    | 0.1241      | 1.75      | 61             
270 | 15 | 0      | 0.5492   | 0.7173   | 0.8552    | 0.07351     | 1.016     | 0              
270 | 16 | 0      | 6.508    | 11.87    | 0.5595    | 0.09698     | 0.2129    | 0              
270 | 17 | 0      | 1.598    | 2.979    | 0.4561    | 0.09368     | 0.5527    | 88.39          
270 | 18 | 0      | 1.758    | 3.841    | 0.004834  | 0.1752      | 2.581     | 59.01          
270 | 19 | 0      | 0.8053   | 1.058    | 0.7862    | 0.08544     | 1.051     | 0              
270 | 20 | 0      | 1.282    | 2.664    | 0.7793    | 0.0468      | 0.5091    | 0              
400 | 2  | 0      | 0.689    | 1.041    | 0.7687    | 0.06961     | 1.199     | 63             
400 | 3  | 0      | 0.5801   | 0.9012   | 0.7762    | 0.05801     | 1.093     | 0              
400 | 4  | 0      | 0.7953   | 1.176    | 0.6914    | 0.05999     | 0.9827    | 63             
400 | 5  | 0      | 0.509    | 0.8019   | 0.9253    | 0.01447     | 0.7981    | 0              
400 | 6  | 0      | 3.545    | 5.199    | 0.5109    | 0.1152      | 0.4799    | 0              
400 | 7  | 0      | 0.2113   | 0.37     | 0.9559    | 0.01932     | 1.027     | 0              
400 | 8  | 0      | 1.307    | 2.967    | 0.2583    | 0.1212      | 0.4971    | 61.01          
400 | 9  | 0      | 2.439    | 3.513    | 0.4322    | 0.1268      | 0.5576    | 64.85          
400 | 10 | 0      | 0.6738   | 0.9852   | 0.7793    | 0.04153     | 0.9002    | 0              
400 | 11 | 0      | 0.5798   | 0.9983   | 0.7015    | 0.04797     | 0.9469    | 88.39          
400 | 12 | 0      | 0.4293   | 0.6606   | 0.8424    | 0.06966     | 1.199     | 0              
400 | 13 | 0      | 0.7563   | 1.014    | 0.8202    | 0.1212      | 1.462     | 88.39          
400 | 14 | 0      | 2.539    | 4.354    | 0.2375    | 0.128       | 0.5256    | 62.29          
400 | 15 | 0      | 0.5841   | 0.7831   | 0.8195    | 0.04528     | 0.8708    | 0              
400 | 16 | 0      | 1.945    | 3.02     | 0.6054    | 0.0844      | 0.5633    | 62             
400 | 17 | 0      | 0.689    | 1.06     | 0.5705    | 0.09954     | 1.103     | 0              
400 | 18 | 0      | 1.639    | 3.111    | 0.7188    | 0.08228     | 0.4331    | 19             
400 | 19 | 0      | 0.6007   | 0.8733   | 0.8039    | 0.03603     | 0.8209    | 0              
400 | 20 | 0      | 0.6796   | 0.9332   | 0.7159    | 0.09342     | 1.156     | 0              
```

## Mean by MHz

```
mhz | n  | mae_ohm_mean | rmse_ohm_mean | pearson_r_mean | pattern_mae_mean | max_ratio_mean
----|----|--------------|---------------|----------------|------------------|---------------
10  | 19 | 0            | 0             | 0.8575         | 0.04651          | 0.923         
70  | 19 | 0.3412       | 0.4486        | 0.8267         | 0.08109          | 0.7188        
120 | 19 | 0.925        | 1.193         | 0.7761         | 0.0854           | 0.6292        
270 | 19 | 2.333        | 4.13          | 0.6003         | 0.1056           | 1.375         
400 | 19 | 1.115        | 1.777         | 0.6807         | 0.07547          | 0.8745        
```

## Agent copy block

```
sim_compare_metrics: n=95
  10.0MHz K2 s0: mae=0Ω rmse=0Ω r=0.838 pattern_mae=0.139 max_ratio=0.808 peak_err=0.0px
  10.0MHz K3 s0: mae=0Ω rmse=0Ω r=0.841 pattern_mae=0.126 max_ratio=0.786 peak_err=0.0px
  10.0MHz K4 s0: mae=0Ω rmse=0Ω r=0.763 pattern_mae=0.082 max_ratio=0.877 peak_err=63.0px
  10.0MHz K5 s0: mae=0Ω rmse=0Ω r=0.776 pattern_mae=0.066 max_ratio=0.890 peak_err=0.0px
  10.0MHz K6 s0: mae=0Ω rmse=0Ω r=0.852 pattern_mae=0.037 max_ratio=0.885 peak_err=63.0px
  10.0MHz K7 s0: mae=0Ω rmse=0Ω r=0.958 pattern_mae=0.012 max_ratio=1.020 peak_err=0.0px
  10.0MHz K8 s0: mae=0Ω rmse=0Ω r=0.676 pattern_mae=0.044 max_ratio=1.088 peak_err=63.0px
  10.0MHz K9 s0: mae=0Ω rmse=0Ω r=0.795 pattern_mae=0.049 max_ratio=1.065 peak_err=88.4px
  10.0MHz K10 s0: mae=0Ω rmse=0Ω r=0.824 pattern_mae=0.051 max_ratio=1.146 peak_err=88.4px
  10.0MHz K11 s0: mae=0Ω rmse=0Ω r=0.948 pattern_mae=0.013 max_ratio=1.027 peak_err=0.0px
  10.0MHz K12 s0: mae=0Ω rmse=0Ω r=0.982 pattern_mae=0.011 max_ratio=1.029 peak_err=0.0px
  10.0MHz K13 s0: mae=0Ω rmse=0Ω r=0.958 pattern_mae=0.014 max_ratio=0.986 peak_err=0.0px
  10.0MHz K14 s0: mae=0Ω rmse=0Ω r=0.856 pattern_mae=0.058 max_ratio=0.755 peak_err=0.0px
  10.0MHz K15 s0: mae=0Ω rmse=0Ω r=0.851 pattern_mae=0.027 max_ratio=0.786 peak_err=0.0px
  10.0MHz K16 s0: mae=0Ω rmse=0Ω r=0.894 pattern_mae=0.036 max_ratio=0.708 peak_err=0.0px
  10.0MHz K17 s0: mae=0Ω rmse=0Ω r=0.791 pattern_mae=0.043 max_ratio=0.784 peak_err=63.0px
  10.0MHz K18 s0: mae=0Ω rmse=0Ω r=0.954 pattern_mae=0.011 max_ratio=0.957 peak_err=0.0px
  10.0MHz K19 s0: mae=0Ω rmse=0Ω r=0.843 pattern_mae=0.037 max_ratio=1.021 peak_err=0.0px
  10.0MHz K20 s0: mae=0Ω rmse=0Ω r=0.893 pattern_mae=0.029 max_ratio=0.920 peak_err=63.0px
  70.0MHz K2 s0: mae=0.9993Ω rmse=1.174Ω r=0.356 pattern_mae=0.346 max_ratio=0.953 peak_err=56.0px
  70.0MHz K3 s0: mae=3.427Ω rmse=3.785Ω r=0.517 pattern_mae=0.346 max_ratio=0.375 peak_err=53.9px
  70.0MHz K4 s0: mae=1.017Ω rmse=1.181Ω r=0.613 pattern_mae=0.195 max_ratio=0.485 peak_err=63.0px
  70.0MHz K5 s0: mae=0.02707Ω rmse=0.09227Ω r=0.985 pattern_mae=0.033 max_ratio=0.736 peak_err=0.0px
  70.0MHz K6 s0: mae=0.1805Ω rmse=0.3334Ω r=0.692 pattern_mae=0.088 max_ratio=0.603 peak_err=63.0px
  70.0MHz K7 s0: mae=0.06338Ω rmse=0.1681Ω r=0.918 pattern_mae=0.032 max_ratio=0.792 peak_err=0.0px
  70.0MHz K8 s0: mae=0.06549Ω rmse=0.1646Ω r=0.692 pattern_mae=0.063 max_ratio=0.827 peak_err=0.0px
  70.0MHz K9 s0: mae=0.005874Ω rmse=0.04015Ω r=0.897 pattern_mae=0.086 max_ratio=0.881 peak_err=62.0px
  70.0MHz K10 s0: mae=0.01272Ω rmse=0.06023Ω r=0.844 pattern_mae=0.076 max_ratio=0.849 peak_err=62.0px
  70.0MHz K11 s0: mae=0.01022Ω rmse=0.05604Ω r=0.958 pattern_mae=0.017 max_ratio=0.751 peak_err=0.0px
  70.0MHz K12 s0: mae=0.00861Ω rmse=0.05294Ω r=0.987 pattern_mae=0.005 max_ratio=0.784 peak_err=0.0px
  70.0MHz K13 s0: mae=0.03384Ω rmse=0.1062Ω r=0.975 pattern_mae=0.003 max_ratio=0.747 peak_err=0.0px
  70.0MHz K14 s0: mae=0.3546Ω rmse=0.5171Ω r=0.805 pattern_mae=0.098 max_ratio=0.571 peak_err=0.0px
  70.0MHz K15 s0: mae=0.04131Ω rmse=0.1332Ω r=0.817 pattern_mae=0.046 max_ratio=0.675 peak_err=0.0px
  70.0MHz K16 s0: mae=0.03664Ω rmse=0.1117Ω r=0.958 pattern_mae=0.014 max_ratio=0.729 peak_err=0.0px
  70.0MHz K17 s0: mae=0.0725Ω rmse=0.1885Ω r=0.880 pattern_mae=0.031 max_ratio=0.642 peak_err=0.0px
  70.0MHz K18 s0: mae=0.008795Ω rmse=0.05414Ω r=0.981 pattern_mae=0.007 max_ratio=0.782 peak_err=0.0px
  70.0MHz K19 s0: mae=0.04071Ω rmse=0.1249Ω r=0.922 pattern_mae=0.025 max_ratio=0.763 peak_err=0.0px
  70.0MHz K20 s0: mae=0.07695Ω rmse=0.1798Ω r=0.911 pattern_mae=0.030 max_ratio=0.711 peak_err=63.0px
  120.0MHz K2 s0: mae=0.9109Ω rmse=1.088Ω r=0.743 pattern_mae=0.100 max_ratio=1.458 peak_err=0.0px
  120.0MHz K3 s0: mae=1.128Ω rmse=1.287Ω r=0.654 pattern_mae=0.180 max_ratio=1.267 peak_err=19.0px
  120.0MHz K4 s0: mae=0.2958Ω rmse=0.4715Ω r=0.568 pattern_mae=0.082 max_ratio=0.709 peak_err=0.0px
  120.0MHz K5 s0: mae=1.8Ω rmse=1.978Ω r=0.711 pattern_mae=0.143 max_ratio=0.377 peak_err=19.0px
  120.0MHz K6 s0: mae=1.391Ω rmse=1.708Ω r=0.412 pattern_mae=0.134 max_ratio=0.337 peak_err=63.0px
  120.0MHz K7 s0: mae=0.9552Ω rmse=1.064Ω r=0.921 pattern_mae=0.056 max_ratio=0.478 peak_err=0.0px
  120.0MHz K8 s0: mae=0.802Ω rmse=0.8868Ω r=0.411 pattern_mae=0.114 max_ratio=0.521 peak_err=44.0px
  120.0MHz K9 s0: mae=0.7205Ω rmse=0.7463Ω r=0.775 pattern_mae=0.130 max_ratio=0.537 peak_err=62.0px
  120.0MHz K10 s0: mae=0.4782Ω rmse=0.5375Ω r=0.756 pattern_mae=0.135 max_ratio=0.769 peak_err=64.8px
  120.0MHz K11 s0: mae=0.06915Ω rmse=0.1699Ω r=0.924 pattern_mae=0.038 max_ratio=0.720 peak_err=0.0px
  120.0MHz K12 s0: mae=0.1191Ω rmse=0.2397Ω r=0.947 pattern_mae=0.021 max_ratio=0.796 peak_err=84.2px
  120.0MHz K13 s0: mae=0.6177Ω rmse=0.7017Ω r=0.965 pattern_mae=0.010 max_ratio=0.512 peak_err=0.0px
  120.0MHz K14 s0: mae=6.349Ω rmse=8.925Ω r=0.622 pattern_mae=0.303 max_ratio=0.144 peak_err=6.4px
  120.0MHz K15 s0: mae=0.2696Ω rmse=0.3355Ω r=0.874 pattern_mae=0.034 max_ratio=0.624 peak_err=0.0px
  120.0MHz K16 s0: mae=0.3221Ω rmse=0.5158Ω r=0.876 pattern_mae=0.029 max_ratio=0.504 peak_err=0.0px
  120.0MHz K17 s0: mae=0.5186Ω rmse=0.7006Ω r=0.815 pattern_mae=0.043 max_ratio=0.423 peak_err=63.0px
  120.0MHz K18 s0: mae=0.2067Ω rmse=0.3211Ω r=0.972 pattern_mae=0.014 max_ratio=0.714 peak_err=1.0px
  120.0MHz K19 s0: mae=0.2502Ω rmse=0.4121Ω r=0.912 pattern_mae=0.020 max_ratio=0.557 peak_err=0.0px
  120.0MHz K20 s0: mae=0.3696Ω rmse=0.5724Ω r=0.888 pattern_mae=0.038 max_ratio=0.505 peak_err=63.0px
  270.0MHz K2 s0: mae=5.179Ω rmse=8.681Ω r=0.325 pattern_mae=0.187 max_ratio=6.762 peak_err=52.1px
  270.0MHz K3 s0: mae=3.775Ω rmse=6.065Ω r=0.433 pattern_mae=0.145 max_ratio=3.057 peak_err=56.0px
  270.0MHz K4 s0: mae=0.8758Ω rmse=1.35Ω r=0.742 pattern_mae=0.106 max_ratio=1.398 peak_err=56.1px
  270.0MHz K5 s0: mae=1.499Ω rmse=2.228Ω r=0.627 pattern_mae=0.119 max_ratio=0.886 peak_err=83.5px
  270.0MHz K6 s0: mae=1.516Ω rmse=2.561Ω r=0.527 pattern_mae=0.090 max_ratio=0.642 peak_err=88.4px
  270.0MHz K7 s0: mae=5.238Ω rmse=10.62Ω r=0.485 pattern_mae=0.120 max_ratio=0.250 peak_err=52.0px
  270.0MHz K8 s0: mae=1.143Ω rmse=1.812Ω r=0.448 pattern_mae=0.116 max_ratio=1.113 peak_err=56.1px
  270.0MHz K9 s0: mae=1.1Ω rmse=1.427Ω r=0.785 pattern_mae=0.093 max_ratio=0.853 peak_err=62.0px
  270.0MHz K10 s0: mae=0.4877Ω rmse=0.7094Ω r=0.843 pattern_mae=0.101 max_ratio=1.274 peak_err=63.9px
  270.0MHz K11 s0: mae=4.786Ω rmse=9.196Ω r=0.847 pattern_mae=0.061 max_ratio=0.405 peak_err=62.1px
  270.0MHz K12 s0: mae=3.072Ω rmse=5.28Ω r=0.302 pattern_mae=0.103 max_ratio=1.289 peak_err=62.4px
  270.0MHz K13 s0: mae=2.493Ω rmse=4.135Ω r=0.760 pattern_mae=0.071 max_ratio=0.531 peak_err=44.0px
  270.0MHz K14 s0: mae=0.669Ω rmse=1.277Ω r=0.840 pattern_mae=0.124 max_ratio=1.750 peak_err=61.0px
  270.0MHz K15 s0: mae=0.5492Ω rmse=0.7173Ω r=0.855 pattern_mae=0.074 max_ratio=1.016 peak_err=0.0px
  270.0MHz K16 s0: mae=6.508Ω rmse=11.87Ω r=0.559 pattern_mae=0.097 max_ratio=0.213 peak_err=0.0px
  270.0MHz K17 s0: mae=1.598Ω rmse=2.979Ω r=0.456 pattern_mae=0.094 max_ratio=0.553 peak_err=88.4px
  270.0MHz K18 s0: mae=1.758Ω rmse=3.841Ω r=0.005 pattern_mae=0.175 max_ratio=2.581 peak_err=59.0px
  270.0MHz K19 s0: mae=0.8053Ω rmse=1.058Ω r=0.786 pattern_mae=0.085 max_ratio=1.051 peak_err=0.0px
  270.0MHz K20 s0: mae=1.282Ω rmse=2.664Ω r=0.779 pattern_mae=0.047 max_ratio=0.509 peak_err=0.0px
  400.0MHz K2 s0: mae=0.689Ω rmse=1.041Ω r=0.769 pattern_mae=0.070 max_ratio=1.199 peak_err=63.0px
  400.0MHz K3 s0: mae=0.5801Ω rmse=0.9012Ω r=0.776 pattern_mae=0.058 max_ratio=1.093 peak_err=0.0px
  400.0MHz K4 s0: mae=0.7953Ω rmse=1.176Ω r=0.691 pattern_mae=0.060 max_ratio=0.983 peak_err=63.0px
  400.0MHz K5 s0: mae=0.509Ω rmse=0.8019Ω r=0.925 pattern_mae=0.014 max_ratio=0.798 peak_err=0.0px
  400.0MHz K6 s0: mae=3.545Ω rmse=5.199Ω r=0.511 pattern_mae=0.115 max_ratio=0.480 peak_err=0.0px
  400.0MHz K7 s0: mae=0.2113Ω rmse=0.37Ω r=0.956 pattern_mae=0.019 max_ratio=1.027 peak_err=0.0px
  400.0MHz K8 s0: mae=1.307Ω rmse=2.967Ω r=0.258 pattern_mae=0.121 max_ratio=0.497 peak_err=61.0px
  400.0MHz K9 s0: mae=2.439Ω rmse=3.513Ω r=0.432 pattern_mae=0.127 max_ratio=0.558 peak_err=64.8px
  400.0MHz K10 s0: mae=0.6738Ω rmse=0.9852Ω r=0.779 pattern_mae=0.042 max_ratio=0.900 peak_err=0.0px
  400.0MHz K11 s0: mae=0.5798Ω rmse=0.9983Ω r=0.702 pattern_mae=0.048 max_ratio=0.947 peak_err=88.4px
  400.0MHz K12 s0: mae=0.4293Ω rmse=0.6606Ω r=0.842 pattern_mae=0.070 max_ratio=1.199 peak_err=0.0px
  400.0MHz K13 s0: mae=0.7563Ω rmse=1.014Ω r=0.820 pattern_mae=0.121 max_ratio=1.462 peak_err=88.4px
  400.0MHz K14 s0: mae=2.539Ω rmse=4.354Ω r=0.237 pattern_mae=0.128 max_ratio=0.526 peak_err=62.3px
  400.0MHz K15 s0: mae=0.5841Ω rmse=0.7831Ω r=0.819 pattern_mae=0.045 max_ratio=0.871 peak_err=0.0px
  400.0MHz K16 s0: mae=1.945Ω rmse=3.02Ω r=0.605 pattern_mae=0.084 max_ratio=0.563 peak_err=62.0px
  400.0MHz K17 s0: mae=0.689Ω rmse=1.06Ω r=0.571 pattern_mae=0.100 max_ratio=1.103 peak_err=0.0px
  400.0MHz K18 s0: mae=1.639Ω rmse=3.111Ω r=0.719 pattern_mae=0.082 max_ratio=0.433 peak_err=19.0px
  400.0MHz K19 s0: mae=0.6007Ω rmse=0.8733Ω r=0.804 pattern_mae=0.036 max_ratio=0.821 peak_err=0.0px
  400.0MHz K20 s0: mae=0.6796Ω rmse=0.9332Ω r=0.716 pattern_mae=0.093 max_ratio=1.156 peak_err=0.0px
```
