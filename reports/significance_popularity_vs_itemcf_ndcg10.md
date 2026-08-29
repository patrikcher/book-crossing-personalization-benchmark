# Significance: popularity vs itemcf on ndcg@10

Paired Wilcoxon signed-rank test on per-user metric deltas, per train-history-depth bucket, run before treating the raw metric lift as real rather than noise — sparse buckets especially need this. No multiple-comparison correction applied across the 4 buckets.

| bucket | n_users | mean_popularity | mean_itemcf | mean_delta | wilcoxon_stat | p_value | note |
|---|---|---|---|---|---|---|---|
| 1-2 | 19,036 | 0.3450 | 0.4847 | 0.1397 | 24171014.0000 | 0.0000 | significant (p<0.05) |
| 3-5 | 9,698 | 0.3048 | 0.4135 | 0.1087 | 6878025.5000 | 0.0000 | significant (p<0.05) |
| 6-20 | 10,562 | 0.2435 | 0.3022 | 0.0587 | 5573289.5000 | 0.0000 | significant (p<0.05) |
| 21+ | 6,821 | 0.1249 | 0.1904 | 0.0655 | 823352.5000 | 0.0000 | significant (p<0.05) |