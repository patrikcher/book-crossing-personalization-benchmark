# Significance: popularity vs itemcf on hit@10

Paired Wilcoxon signed-rank test on per-user metric deltas, per train-history-depth bucket, run before treating the raw metric lift as real rather than noise — sparse buckets especially need this. No multiple-comparison correction applied across the 4 buckets.

| bucket | n_users | mean_popularity | mean_itemcf | mean_delta | wilcoxon_stat | p_value | note |
|---|---|---|---|---|---|---|---|
| 1-2 | 19,036 | 0.4892 | 0.6765 | 0.1873 | 6061770.0000 | <0.0001 | significant (p<0.05) |
| 3-5 | 9,698 | 0.4462 | 0.6163 | 0.1701 | 1808583.0000 | <0.0001 | significant (p<0.05) |
| 6-20 | 10,562 | 0.3674 | 0.4725 | 0.1051 | 1499763.0000 | <0.0001 | significant (p<0.05) |
| 21+ | 6,821 | 0.2174 | 0.3061 | 0.0887 | 303660.0000 | <0.0001 | significant (p<0.05) |