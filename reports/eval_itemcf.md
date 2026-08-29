# Eval: Tier 2 — Item-Based Collaborative Filtering

Evaluated on 46,117 users (leave-1-out, 99 sampled negatives per user, seed 42). See `src/bxbench/eval.py` docstring for the sampled-negatives protocol and its stated limitations.

## Cost proxies

- Training time: 1.15 s (build item x user sparse matrix + signal-coverage precheck over 1,044,497 rows)
- Mean inference latency: 0.222 ms/request (scoring 100 candidates against a user's train history via sparse cosine similarity, or a popularity-fallback lookup when signal-based fallback triggers)
- Retrain cadence: needs the full item x user matrix + signal precheck rebuilt on new data (not an O(1) incremental update like Tier 1) — still cheap at this scale, but this is the first tier where retrain cost is nonzero and worth tracking as the benchmark moves to bigger tiers.

## Coverage

- Personalized (vs. popularity-fallback): 95.1% overall. Fallback rule is signal-based (see `src/bxbench/models/item_cf.py` docstring), not a fixed interaction-count cutoff — see per-bucket breakdown below for where the fallback actually concentrates.

## Metrics — overall

| scope | n_users | personalized | hit@5 | precision@5 | recall@5 | ndcg@5 | hit@10 | precision@10 | recall@10 | ndcg@10 | hit@20 | precision@20 | recall@20 | ndcg@20 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| overall | 46,117 | 95% | 0.4513 | 0.0903 | 0.4513 | 0.3483 | 0.5623 | 0.0562 | 0.5623 | 0.3844 | 0.6794 | 0.0340 | 0.6794 | 0.4140 |

## Metrics — by train-history-depth bucket

This is the finding, not the row above — the aggregate hides the bucketed breakdown.

| bucket | n_users | personalized | hit@5 | precision@5 | recall@5 | ndcg@5 | hit@10 | precision@10 | recall@10 | ndcg@10 | hit@20 | precision@20 | recall@20 | ndcg@20 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1-2 | 19,036 | 88% | 0.5652 | 0.1130 | 0.5652 | 0.4485 | 0.6765 | 0.0676 | 0.6765 | 0.4847 | 0.7861 | 0.0393 | 0.7861 | 0.5126 |
| 3-5 | 9,698 | 100% | 0.4854 | 0.0971 | 0.4854 | 0.3709 | 0.6163 | 0.0616 | 0.6163 | 0.4135 | 0.7502 | 0.0375 | 0.7502 | 0.4474 |
| 6-20 | 10,562 | 100% | 0.3602 | 0.0720 | 0.3602 | 0.2658 | 0.4725 | 0.0473 | 0.4725 | 0.3022 | 0.6005 | 0.0300 | 0.6005 | 0.3343 |
| 21+ | 6,821 | 100% | 0.2261 | 0.0452 | 0.2261 | 0.1644 | 0.3061 | 0.0306 | 0.3061 | 0.1904 | 0.4032 | 0.0202 | 0.4032 | 0.2148 |
