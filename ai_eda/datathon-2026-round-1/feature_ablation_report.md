# Feature Ablation Report

## A. Features Actually Used
| model       |   position | feature                   |
|:------------|-----------:|:--------------------------|
| Revenue     |          1 | Revenue_seasonal_base     |
| Revenue     |          2 | AOV_lag_1                 |
| Revenue     |          3 | Revenue_lag_1             |
| Revenue     |          4 | Revenue_lag_7             |
| Revenue     |          5 | order_count_lag_90        |
| Revenue     |          6 | order_count_lag_7         |
| Revenue     |          7 | order_count_rolling_std_7 |
| Revenue     |          8 | AOV_lag_364               |
| Revenue     |          9 | Revenue_lag_365           |
| Revenue     |         10 | order_count_seasonal_base |
| Revenue     |         11 | AOV_lag_730               |
| Revenue     |         12 | Revenue_lag_364           |
| COGS        |          1 | COGS_seasonal_base        |
| COGS        |          2 | COGS_lag_1                |
| COGS        |          3 | COGS_lag_364              |
| COGS        |          4 | COGS_lag_7                |
| COGS        |          5 | COGS_lag_365              |
| COGS        |          6 | COGS_lag_90               |
| COGS        |          7 | COGS_lag_730              |
| COGS        |          8 | COGS_lag_14               |
| COGS        |          9 | COGS_rolling_std_7        |
| COGS        |         10 | COGS_lag_30               |
| COGS        |         11 | COGS_lag_366              |
| COGS        |         12 | time_index                |
| order_count |          1 | order_count_seasonal_base |
| order_count |          2 | order_count_lag_1         |
| order_count |          3 | order_count_lag_7         |
| order_count |          4 | order_count_lag_365       |
| order_count |          5 | order_count_lag_364       |
| order_count |          6 | order_count_lag_730       |
| order_count |          7 | order_count_lag_366       |
| order_count |          8 | order_count_lag_90        |
| order_count |          9 | order_count_lag_14        |
| order_count |         10 | year_cos_3                |
| order_count |         11 | order_count_lag_30        |
| order_count |         12 | order_count_rolling_std_7 |
| AOV         |          1 | AOV_seasonal_base         |
| AOV         |          2 | AOV_lag_1                 |
| AOV         |          3 | AOV_lag_30                |
| AOV         |          4 | AOV_lag_730               |
| AOV         |          5 | AOV_lag_90                |
| AOV         |          6 | AOV_lag_366               |
| AOV         |          7 | AOV_rolling_std_7         |
| AOV         |          8 | AOV_lag_365               |
| AOV         |          9 | AOV_lag_364               |
| AOV         |         10 | AOV_lag_14                |
| AOV         |         11 | AOV_lag_7                 |
| AOV         |         12 | AOV_rolling_mean_7        |
| cogs_ratio  |          1 | cogs_ratio_seasonal_base  |
| cogs_ratio  |          2 | cogs_ratio_lag_1          |
| cogs_ratio  |          3 | cogs_ratio_lag_730        |
| cogs_ratio  |          4 | cogs_ratio_rolling_std_7  |
| cogs_ratio  |          5 | cogs_ratio_lag_364        |
| cogs_ratio  |          6 | cogs_ratio_lag_30         |
| cogs_ratio  |          7 | cogs_ratio_lag_90         |
| cogs_ratio  |          8 | cogs_ratio_lag_365        |
| cogs_ratio  |          9 | cogs_ratio_rolling_mean_7 |
| cogs_ratio  |         10 | cogs_ratio_yoy_lag_mean   |
| cogs_ratio  |         11 | cogs_ratio_lag_366        |
| cogs_ratio  |         12 | time_index                |

## B. Top LightGBM Feature Importance
Averaged across the baseline models trained on Fold C and Fold D training windows.
| model       | feature                   | mean_importance   |   mean_importance_share |   folds_seen |
|:------------|:--------------------------|:------------------|------------------------:|-------------:|
| AOV         | AOV_seasonal_base         | 2,101.0000        |                  0.1077 |            2 |
| AOV         | AOV_rolling_std_7         | 1,676.5000        |                  0.086  |            2 |
| AOV         | AOV_lag_90                | 1,663.0000        |                  0.0853 |            2 |
| AOV         | AOV_lag_30                | 1,631.0000        |                  0.0836 |            2 |
| AOV         | AOV_lag_14                | 1,618.0000        |                  0.083  |            2 |
| AOV         | AOV_lag_1                 | 1,596.0000        |                  0.0818 |            2 |
| AOV         | AOV_lag_730               | 1,586.0000        |                  0.0813 |            2 |
| AOV         | AOV_lag_7                 | 1,580.0000        |                  0.081  |            2 |
| AOV         | AOV_lag_366               | 1,574.0000        |                  0.0807 |            2 |
| AOV         | AOV_lag_364               | 1,536.5000        |                  0.0788 |            2 |
| AOV         | AOV_lag_365               | 1,524.0000        |                  0.0782 |            2 |
| AOV         | AOV_rolling_mean_7        | 1,414.0000        |                  0.0725 |            2 |
| COGS        | COGS_seasonal_base        | 2,321.5000        |                  0.1191 |            2 |
| COGS        | COGS_lag_1                | 1,830.5000        |                  0.0939 |            2 |
| COGS        | COGS_lag_90               | 1,706.5000        |                  0.0875 |            2 |
| COGS        | COGS_lag_365              | 1,703.5000        |                  0.0874 |            2 |
| COGS        | time_index                | 1,643.0000        |                  0.0843 |            2 |
| COGS        | COGS_lag_366              | 1,558.5000        |                  0.0799 |            2 |
| COGS        | COGS_lag_730              | 1,532.0000        |                  0.0786 |            2 |
| COGS        | COGS_lag_7                | 1,498.0000        |                  0.0768 |            2 |
| COGS        | COGS_rolling_std_7        | 1,490.0000        |                  0.0764 |            2 |
| COGS        | COGS_lag_364              | 1,484.0000        |                  0.0761 |            2 |
| COGS        | COGS_lag_30               | 1,457.0000        |                  0.0747 |            2 |
| COGS        | COGS_lag_14               | 1,275.5000        |                  0.0654 |            2 |
| Revenue     | order_count_lag_90        | 1,877.5000        |                  0.0963 |            2 |
| Revenue     | AOV_lag_730               | 1,758.5000        |                  0.0902 |            2 |
| Revenue     | Revenue_lag_1             | 1,739.5000        |                  0.0892 |            2 |
| Revenue     | AOV_lag_1                 | 1,681.0000        |                  0.0862 |            2 |
| Revenue     | AOV_lag_364               | 1,622.5000        |                  0.0832 |            2 |
| Revenue     | order_count_rolling_std_7 | 1,618.0000        |                  0.083  |            2 |
| Revenue     | Revenue_seasonal_base     | 1,616.0000        |                  0.0829 |            2 |
| Revenue     | Revenue_lag_365           | 1,599.5000        |                  0.082  |            2 |
| Revenue     | order_count_seasonal_base | 1,560.0000        |                  0.08   |            2 |
| Revenue     | Revenue_lag_364           | 1,554.0000        |                  0.0797 |            2 |
| Revenue     | Revenue_lag_7             | 1,472.5000        |                  0.0755 |            2 |
| Revenue     | order_count_lag_7         | 1,401.0000        |                  0.0718 |            2 |
| cogs_ratio  | cogs_ratio_seasonal_base  | 2,450.5000        |                  0.1282 |            2 |
| cogs_ratio  | cogs_ratio_lag_1          | 1,954.5000        |                  0.1023 |            2 |
| cogs_ratio  | cogs_ratio_rolling_mean_7 | 1,727.5000        |                  0.0904 |            2 |
| cogs_ratio  | cogs_ratio_lag_730        | 1,696.5000        |                  0.0888 |            2 |
| cogs_ratio  | cogs_ratio_lag_30         | 1,694.5000        |                  0.0887 |            2 |
| cogs_ratio  | cogs_ratio_rolling_std_7  | 1,562.0000        |                  0.0817 |            2 |
| cogs_ratio  | time_index                | 1,541.0000        |                  0.0806 |            2 |
| cogs_ratio  | cogs_ratio_lag_90         | 1,526.5000        |                  0.0799 |            2 |
| cogs_ratio  | cogs_ratio_lag_365        | 1,287.0000        |                  0.0673 |            2 |
| cogs_ratio  | cogs_ratio_lag_364        | 1,286.5000        |                  0.0673 |            2 |
| cogs_ratio  | cogs_ratio_lag_366        | 1,236.5000        |                  0.0647 |            2 |
| cogs_ratio  | cogs_ratio_yoy_lag_mean   | 1,146.5000        |                  0.06   |            2 |
| order_count | order_count_seasonal_base | 1,963.5000        |                  0.1007 |            2 |
| order_count | year_cos_3                | 1,872.5000        |                  0.096  |            2 |
| order_count | order_count_lag_1         | 1,693.0000        |                  0.0868 |            2 |
| order_count | order_count_lag_90        | 1,692.5000        |                  0.0868 |            2 |
| order_count | order_count_lag_730       | 1,671.0000        |                  0.0857 |            2 |
| order_count | order_count_lag_7         | 1,657.0000        |                  0.085  |            2 |
| order_count | order_count_lag_364       | 1,582.5000        |                  0.0812 |            2 |
| order_count | order_count_lag_366       | 1,579.5000        |                  0.081  |            2 |
| order_count | order_count_lag_365       | 1,501.5000        |                  0.077  |            2 |
| order_count | order_count_lag_30        | 1,499.5000        |                  0.0769 |            2 |
| order_count | order_count_rolling_std_7 | 1,417.5000        |                  0.0727 |            2 |
| order_count | order_count_lag_14        | 1,370.0000        |                  0.0703 |            2 |

## C. Ablation Feature Counts
| fold   | experiment               |   Revenue_feature_count |   COGS_feature_count |   order_count_feature_count |   AOV_feature_count |   cogs_ratio_feature_count |
|:-------|:-------------------------|------------------------:|---------------------:|----------------------------:|--------------------:|---------------------------:|
| C      | baseline_current         |                      12 |                   12 |                          12 |                  12 |                         12 |
| C      | hand_picked_importance   |                      12 |                   11 |                          11 |                  11 |                         11 |
| C      | top_10                   |                      10 |                   10 |                          10 |                  10 |                         10 |
| C      | top_8                    |                       8 |                    8 |                           8 |                   8 |                          8 |
| C      | top_6                    |                       6 |                    6 |                           6 |                   6 |                          6 |
| C      | remove_low_importance    |                      10 |                   10 |                          10 |                  11 |                          8 |
| C      | remove_redundant_obvious |                      11 |                   10 |                          10 |                  10 |                          9 |
| D      | baseline_current         |                      12 |                   12 |                          12 |                  12 |                         12 |
| D      | hand_picked_importance   |                      12 |                   11 |                          11 |                  11 |                         11 |
| D      | top_10                   |                      10 |                   10 |                          10 |                  10 |                         10 |
| D      | top_8                    |                       8 |                    8 |                           8 |                   8 |                          8 |
| D      | top_6                    |                       6 |                    6 |                           6 |                   6 |                          6 |
| D      | remove_low_importance    |                      11 |                    8 |                           8 |                  11 |                          8 |
| D      | remove_redundant_obvious |                      11 |                   10 |                          10 |                  10 |                          9 |

## D. Validation Metrics
Fold-level metrics:
| experiment               | fold   | target   | MAE            | RMSE           |      R2 |
|:-------------------------|:-------|:---------|:---------------|:---------------|--------:|
| baseline_current         | C      | COGS     | 614,720.8831   | 851,313.6331   |  0.6958 |
| baseline_current         | C      | Revenue  | 763,120.1262   | 1,051,232.0665 |  0.6449 |
| baseline_current         | D      | COGS     | 553,229.8418   | 751,816.2233   |  0.7343 |
| baseline_current         | D      | Revenue  | 683,628.8723   | 918,167.9236   |  0.6991 |
| hand_picked_importance   | C      | COGS     | 618,267.1577   | 858,629.4103   |  0.6906 |
| hand_picked_importance   | C      | Revenue  | 790,312.6867   | 1,091,708.3220 |  0.6171 |
| hand_picked_importance   | D      | COGS     | 552,061.9868   | 752,078.5102   |  0.7341 |
| hand_picked_importance   | D      | Revenue  | 694,705.5890   | 919,179.8111   |  0.6984 |
| remove_low_importance    | C      | COGS     | 609,171.5224   | 840,992.9899   |  0.7031 |
| remove_low_importance    | C      | Revenue  | 772,271.9330   | 1,051,146.3280 |  0.645  |
| remove_low_importance    | D      | COGS     | 550,896.7340   | 750,309.9552   |  0.7354 |
| remove_low_importance    | D      | Revenue  | 713,302.0094   | 956,826.4247   |  0.6732 |
| remove_redundant_obvious | C      | COGS     | 610,957.8666   | 851,097.1837   |  0.696  |
| remove_redundant_obvious | C      | Revenue  | 772,788.9778   | 1,065,525.4874 |  0.6352 |
| remove_redundant_obvious | D      | COGS     | 660,848.2396   | 918,599.7205   |  0.6034 |
| remove_redundant_obvious | D      | Revenue  | 780,042.4774   | 1,073,346.9438 |  0.5888 |
| top_10                   | C      | COGS     | 610,319.4947   | 840,483.6956   |  0.7035 |
| top_10                   | C      | Revenue  | 772,271.9330   | 1,051,146.3280 |  0.645  |
| top_10                   | D      | COGS     | 544,521.4245   | 742,627.7595   |  0.7408 |
| top_10                   | D      | Revenue  | 703,220.8545   | 931,614.5482   |  0.6902 |
| top_6                    | C      | COGS     | 680,763.3156   | 933,115.0616   |  0.6345 |
| top_6                    | C      | Revenue  | 779,503.3482   | 1,075,700.9094 |  0.6282 |
| top_6                    | D      | COGS     | 731,868.8778   | 1,014,074.6092 |  0.5166 |
| top_6                    | D      | Revenue  | 1,627,310.4058 | 2,268,689.4261 | -0.8371 |
| top_8                    | C      | COGS     | 627,484.3864   | 873,316.6798   |  0.6799 |
| top_8                    | C      | Revenue  | 776,329.7865   | 1,083,505.1511 |  0.6228 |
| top_8                    | D      | COGS     | 549,992.0422   | 752,278.5756   |  0.734  |
| top_8                    | D      | Revenue  | 704,342.5996   | 942,541.0144   |  0.6829 |

Average across Fold C and Fold D:
| experiment               | target   | MAE            | RMSE           |      R2 | MAE_delta_vs_baseline   |   MAE_pct_delta_vs_baseline | status_vs_baseline   |
|:-------------------------|:---------|:---------------|:---------------|--------:|:------------------------|----------------------------:|:---------------------|
| top_10                   | COGS     | 577,420.4596   | 791,555.7276   |  0.7221 | -6,554.9028             |                     -0.0112 | improves             |
| remove_low_importance    | COGS     | 580,034.1282   | 795,651.4725   |  0.7193 | -3,941.2343             |                     -0.0067 | stays similar        |
| baseline_current         | COGS     | 583,975.3624   | 801,564.9282   |  0.7151 | 0.0000                  |                      0      | baseline             |
| hand_picked_importance   | COGS     | 585,164.5722   | 805,353.9603   |  0.7123 | 1,189.2098              |                      0.002  | stays similar        |
| top_8                    | COGS     | 588,738.2143   | 812,797.6277   |  0.7069 | 4,762.8519              |                      0.0082 | stays similar        |
| remove_redundant_obvious | COGS     | 635,903.0531   | 884,848.4521   |  0.6497 | 51,927.6907             |                      0.0889 | gets worse           |
| top_6                    | COGS     | 706,316.0967   | 973,594.8354   |  0.5756 | 122,340.7343            |                      0.2095 | gets worse           |
| baseline_current         | Revenue  | 723,374.4993   | 984,699.9951   |  0.672  | 0.0000                  |                      0      | baseline             |
| top_10                   | Revenue  | 737,746.3937   | 991,380.4381   |  0.6676 | 14,371.8945             |                      0.0199 | gets worse           |
| top_8                    | Revenue  | 740,336.1930   | 1,013,023.0827 |  0.6528 | 16,961.6938             |                      0.0234 | gets worse           |
| hand_picked_importance   | Revenue  | 742,509.1378   | 1,005,444.0666 |  0.6577 | 19,134.6386             |                      0.0265 | gets worse           |
| remove_low_importance    | Revenue  | 742,786.9712   | 1,003,986.3764 |  0.6591 | 19,412.4719             |                      0.0268 | gets worse           |
| remove_redundant_obvious | Revenue  | 776,415.7276   | 1,069,436.2156 |  0.612  | 53,041.2284             |                      0.0733 | gets worse           |
| top_6                    | Revenue  | 1,203,406.8770 | 1,672,195.1678 | -0.1045 | 480,032.3777            |                      0.6636 | gets worse           |

## E. Recommended Feature Set
Recommended set: `baseline_current`.

Reducing features did not produce a clearly safer validation result; keep the current selected features.

## F. Features That Can Probably Be Removed
These are features removed in all Fold C/D versions of a reduced experiment. Treat them as candidates, not automatic production deletions.
| experiment               | model       | feature_removed_in_all_folds   |
|:-------------------------|:------------|:-------------------------------|
| hand_picked_importance   | COGS        | COGS_lag_14                    |
| hand_picked_importance   | order_count | order_count_lag_14             |
| hand_picked_importance   | AOV         | AOV_rolling_mean_7             |
| hand_picked_importance   | cogs_ratio  | cogs_ratio_yoy_lag_mean        |
| top_10                   | Revenue     | order_count_lag_7              |
| top_10                   | COGS        | COGS_lag_14                    |
| top_10                   | order_count | order_count_lag_14             |
| top_10                   | AOV         | AOV_rolling_mean_7             |
| top_10                   | cogs_ratio  | cogs_ratio_yoy_lag_mean        |
| top_8                    | Revenue     | Revenue_lag_7                  |
| top_8                    | Revenue     | order_count_lag_7              |
| top_8                    | COGS        | COGS_lag_14                    |
| top_8                    | COGS        | COGS_lag_30                    |
| top_8                    | order_count | order_count_lag_14             |
| top_8                    | order_count | order_count_rolling_std_7      |
| top_8                    | AOV         | AOV_rolling_mean_7             |
| top_8                    | cogs_ratio  | cogs_ratio_lag_364             |
| top_8                    | cogs_ratio  | cogs_ratio_lag_365             |
| top_8                    | cogs_ratio  | cogs_ratio_lag_366             |
| top_8                    | cogs_ratio  | cogs_ratio_yoy_lag_mean        |
| top_6                    | Revenue     | Revenue_lag_365                |
| top_6                    | Revenue     | Revenue_lag_7                  |
| top_6                    | Revenue     | order_count_lag_7              |
| top_6                    | Revenue     | order_count_seasonal_base      |
| top_6                    | COGS        | COGS_lag_14                    |
| top_6                    | COGS        | COGS_lag_30                    |
| top_6                    | COGS        | COGS_lag_364                   |
| top_6                    | COGS        | COGS_lag_7                     |
| top_6                    | COGS        | COGS_rolling_std_7             |
| top_6                    | order_count | order_count_lag_14             |
| top_6                    | order_count | order_count_lag_30             |
| top_6                    | order_count | order_count_lag_365            |
| top_6                    | order_count | order_count_lag_366            |
| top_6                    | order_count | order_count_rolling_std_7      |
| top_6                    | AOV         | AOV_lag_364                    |
| top_6                    | AOV         | AOV_lag_365                    |
| top_6                    | AOV         | AOV_lag_366                    |
| top_6                    | AOV         | AOV_rolling_mean_7             |
| top_6                    | cogs_ratio  | cogs_ratio_lag_364             |
| top_6                    | cogs_ratio  | cogs_ratio_lag_365             |
| top_6                    | cogs_ratio  | cogs_ratio_lag_366             |
| top_6                    | cogs_ratio  | cogs_ratio_yoy_lag_mean        |
| remove_low_importance    | Revenue     | order_count_lag_7              |
| remove_low_importance    | COGS        | COGS_lag_14                    |
| remove_low_importance    | COGS        | COGS_lag_30                    |
| remove_low_importance    | order_count | order_count_lag_14             |
| remove_low_importance    | order_count | order_count_rolling_std_7      |
| remove_low_importance    | AOV         | AOV_rolling_mean_7             |
| remove_low_importance    | cogs_ratio  | cogs_ratio_lag_364             |
| remove_low_importance    | cogs_ratio  | cogs_ratio_lag_365             |
| remove_low_importance    | cogs_ratio  | cogs_ratio_lag_366             |
| remove_low_importance    | cogs_ratio  | cogs_ratio_yoy_lag_mean        |
| remove_redundant_obvious | COGS        | COGS_lag_364                   |
| remove_redundant_obvious | COGS        | COGS_lag_366                   |
| remove_redundant_obvious | order_count | order_count_lag_365            |
| remove_redundant_obvious | AOV         | AOV_lag_364                    |
| remove_redundant_obvious | AOV         | AOV_lag_365                    |
| remove_redundant_obvious | cogs_ratio  | cogs_ratio_lag_366             |
| remove_redundant_obvious | cogs_ratio  | cogs_ratio_yoy_lag_mean        |

The `hand_picked_importance` set is intentionally conservative: it keeps all Revenue features because top-N pruning worsened recursive Revenue MAE, and removes only the consistently weaker auxiliary features `COGS_lag_14`, `order_count_lag_14`, `AOV_rolling_mean_7`, and `cogs_ratio_yoy_lag_mean`.

## G. Recursive Forecasting Warning
Feature reductions are evaluated recursively: Revenue, COGS, order_count, AOV, and cogs_ratio predictions are fed back into later validation days. If a reduced feature set looks fine on direct training fit but worsens Fold C/D recursive MAE or R2, it should not replace the current allowlist.

## H. Runtime
- Started: `2026-05-01T15:07:29`
- Finished: `2026-05-01T15:10:09`
- Runtime seconds: `160.2`
- Folds: `C, D`
- Model mode: `full`
- Cache directory: `diagnostic_cache_feature_ablation`

Cache status:
| fold   | experiment               | cache_status   |
|:-------|:-------------------------|:---------------|
| C      | baseline_current         | loaded         |
| C      | hand_picked_importance   | recomputed     |
| C      | top_10                   | loaded         |
| C      | top_8                    | loaded         |
| C      | top_6                    | loaded         |
| C      | remove_low_importance    | loaded         |
| C      | remove_redundant_obvious | loaded         |
| D      | baseline_current         | loaded         |
| D      | hand_picked_importance   | recomputed     |
| D      | top_10                   | loaded         |
| D      | top_8                    | loaded         |
| D      | top_6                    | loaded         |
| D      | remove_low_importance    | loaded         |
| D      | remove_redundant_obvious | loaded         |