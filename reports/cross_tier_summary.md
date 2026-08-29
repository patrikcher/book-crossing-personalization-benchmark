# Cross-Tier Synthesis

All 4 tiers, same leave-1-out split, same sampled-negatives eval protocol (99 negatives, seed 42), same history-depth buckets. Pulled from each tier's own report — nothing refit here.

## hit@10 by bucket

| bucket | Tier 1: Popularity | Tier 2: Item-CF | Tier 3: Matrix Factorization | Tier 4: Hybrid/Content-Aware |
|---|---|---|---|---|
| 1-2 | 0.4892 | 0.6765 | 0.4234 | 0.4235 |
| 3-5 | 0.4462 | 0.6163 | 0.4543 | 0.4545 |
| 6-20 | 0.3674 | 0.4725 | 0.4289 | 0.4288 |
| 21+ | 0.2174 | 0.3061 | 0.3118 | 0.3117 |

## ndcg@10 by bucket

| bucket | Tier 1: Popularity | Tier 2: Item-CF | Tier 3: Matrix Factorization | Tier 4: Hybrid/Content-Aware |
|---|---|---|---|---|
| 1-2 | 0.3450 | 0.4847 | 0.3292 | 0.3292 |
| 3-5 | 0.3048 | 0.4135 | 0.3498 | 0.3498 |
| 6-20 | 0.2435 | 0.3022 | 0.3221 | 0.3221 |
| 21+ | 0.1249 | 0.1904 | 0.2170 | 0.2169 |

## Coverage: personalized vs. popularity-fallback

| bucket | Tier 1: Popularity | Tier 2: Item-CF | Tier 3: Matrix Factorization | Tier 4: Hybrid/Content-Aware |
|---|---|---|---|---|
| 1-2 | 0% | 88% | 100% | 100% |
| 3-5 | 0% | 100% | 100% | 100% |
| 6-20 | 0% | 100% | 100% | 100% |
| 21+ | 0% | 100% | 100% | 100% |

Note: Tier 4's coverage number here reflects MF's personalization (inherited, ~100%), not content-profile coverage, which is far lower and reported separately in `reports/eval_hybrid.md`.

## Cost proxies

| tier | training time | inference latency (ms/request) |
|---|---|---|
| Tier 1: Popularity | 76 ms | 0.101 |
| Tier 2: Item-CF | 1.15 s | 0.222 |
| Tier 3: Matrix Factorization | 18.04 s | 0.050 |
| Tier 4: Hybrid/Content-Aware | 36.25 s | 0.081 |

## Significance (paired Wilcoxon, ndcg@10, per bucket)

| comparison | 1-2 | 3-5 | 6-20 | 21+ |
|---|---|---|---|---|
| popularity vs itemcf | p<0.0001 (significant) | p<0.0001 (significant) | p<0.0001 (significant) | p<0.0001 (significant) |
| itemcf vs mf | p<0.0001 (significant) | p<0.0001 (significant) | p<0.0001 (significant) | p<0.0001 (significant) |
| mf vs hybrid | p=0.3173 (not significant) | p=0.1362 (not significant) | p=0.4406 (not significant) | p=0.1819 (not significant) |

## Significance (paired Wilcoxon, hit@10, per bucket)

| comparison | 1-2 | 3-5 | 6-20 | 21+ |
|---|---|---|---|---|
| popularity vs itemcf | p<0.0001 (significant) | p<0.0001 (significant) | p<0.0001 (significant) | p<0.0001 (significant) |
| itemcf vs mf | p<0.0001 (significant) | p<0.0001 (significant) | p<0.0001 (significant) | p=0.2483 (not significant) |
| mf vs hybrid | p=0.3173 (not significant) | p=0.3173 (not significant) | p=0.3173 (not significant) | p=0.6547 (not significant) |

Note: hit@10 and ndcg@10 don't always agree on which tier wins a bucket (they don't for item-CF vs. MF in 6-20/21+ — see README `## Results` for why that's a real finding, not an inconsistency to paper over).
