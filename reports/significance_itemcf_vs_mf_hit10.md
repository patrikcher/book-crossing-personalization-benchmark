# Significance: itemcf vs mf on hit@10

Paired Wilcoxon signed-rank test on per-user metric deltas, per train-history-depth bucket, run before treating the raw metric lift as real rather than noise — sparse buckets especially need this. No multiple-comparison correction applied across the 4 buckets.

| bucket | n_users | mean_itemcf | mean_mf | mean_delta | wilcoxon_stat | p_value | note |
|---|---|---|---|---|---|---|---|
| 1-2 | 19,036 | 0.6765 | 0.4234 | -0.2530 | 4632802.0000 | <0.0001 | significant (p<0.05) |
| 3-5 | 9,698 | 0.6163 | 0.4543 | -0.1620 | 1509651.0000 | <0.0001 | significant (p<0.05) |
| 6-20 | 10,562 | 0.4725 | 0.4289 | -0.0436 | 1478722.0000 | <0.0001 | significant (p<0.05) |
| 21+ | 6,821 | 0.3061 | 0.3118 | 0.0057 | 314621.0000 | 0.2483 | not significant (p>=0.05) |