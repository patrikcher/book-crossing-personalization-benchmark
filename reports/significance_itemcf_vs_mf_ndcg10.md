# Significance: itemcf vs mf on ndcg@10

Paired Wilcoxon signed-rank test on per-user metric deltas, per train-history-depth bucket, run before treating the raw metric lift as real rather than noise — sparse buckets especially need this. No multiple-comparison correction applied across the 4 buckets.

| bucket | n_users | mean_itemcf | mean_mf | mean_delta | wilcoxon_stat | p_value | note |
|---|---|---|---|---|---|---|---|
| 1-2 | 19,036 | 0.4847 | 0.3292 | -0.1554 | 19991350.5000 | <0.0001 | significant (p<0.05) |
| 3-5 | 9,698 | 0.4135 | 0.3498 | -0.0637 | 6924261.0000 | <0.0001 | significant (p<0.05) |
| 6-20 | 10,562 | 0.3022 | 0.3221 | 0.0199 | 5762966.0000 | <0.0001 | significant (p<0.05) |
| 21+ | 6,821 | 0.1904 | 0.2170 | 0.0265 | 1009962.5000 | <0.0001 | significant (p<0.05) |