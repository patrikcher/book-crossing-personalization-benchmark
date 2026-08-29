# Significance: popularity vs mf on ndcg@10

Paired Wilcoxon signed-rank test on per-user metric deltas, per train-history-depth bucket, run before treating the raw metric lift as real rather than noise — sparse buckets especially need this. No multiple-comparison correction applied across the 4 buckets.

| bucket | n_users | mean_popularity | mean_mf | mean_delta | wilcoxon_stat | p_value | note |
|---|---|---|---|---|---|---|---|
| 1-2 | 19,036 | 0.3450 | 0.3292 | -0.0157 | 16214815.5000 | 0.0000 | significant (p<0.05) |
| 3-5 | 9,698 | 0.3048 | 0.3498 | 0.0449 | 2924401.5000 | 0.0000 | significant (p<0.05) |
| 6-20 | 10,562 | 0.2435 | 0.3221 | 0.0786 | 2065736.5000 | 0.0000 | significant (p<0.05) |
| 21+ | 6,821 | 0.1249 | 0.2170 | 0.0920 | 442772.0000 | 0.0000 | significant (p<0.05) |