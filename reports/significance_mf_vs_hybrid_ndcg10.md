# Significance: mf vs hybrid on ndcg@10

Paired Wilcoxon signed-rank test on per-user metric deltas, per train-history-depth bucket, run before treating the raw metric lift as real rather than noise — sparse buckets especially need this. No multiple-comparison correction applied across the 4 buckets.

| bucket | n_users | mean_mf | mean_hybrid | mean_delta | wilcoxon_stat | p_value | note |
|---|---|---|---|---|---|---|---|
| 1-2 | 19,036 | 0.3292 | 0.3292 | 0.0000 | 0.0000 | 0.3173 | not significant (p>=0.05) |
| 3-5 | 9,698 | 0.3498 | 0.3498 | 0.0001 | 2.0000 | 0.1362 | not significant (p>=0.05) |
| 6-20 | 10,562 | 0.3221 | 0.3221 | -0.0000 | 12.5000 | 0.4406 | not significant (p>=0.05) |
| 21+ | 6,821 | 0.2170 | 0.2169 | -0.0001 | 180.0000 | 0.1819 | not significant (p>=0.05) |