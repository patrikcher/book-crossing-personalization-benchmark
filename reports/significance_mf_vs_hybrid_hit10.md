# Significance: mf vs hybrid on hit@10

Paired Wilcoxon signed-rank test on per-user metric deltas, per train-history-depth bucket, run before treating the raw metric lift as real rather than noise — sparse buckets especially need this. No multiple-comparison correction applied across the 4 buckets.

| bucket | n_users | mean_mf | mean_hybrid | mean_delta | wilcoxon_stat | p_value | note |
|---|---|---|---|---|---|---|---|
| 1-2 | 19,036 | 0.4234 | 0.4235 | 0.0001 | 0.0000 | 0.3173 | not significant (p>=0.05) |
| 3-5 | 9,698 | 0.4543 | 0.4545 | 0.0002 | 2.5000 | 0.3173 | not significant (p>=0.05) |
| 6-20 | 10,562 | 0.4289 | 0.4288 | -0.0001 | 0.0000 | 0.3173 | not significant (p>=0.05) |
| 21+ | 6,821 | 0.3118 | 0.3117 | -0.0001 | 6.0000 | 0.6547 | not significant (p>=0.05) |