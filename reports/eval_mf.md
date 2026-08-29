# Eval: Tier 3 — Matrix Factorization (Implicit-Feedback ALS)

Evaluated on 46,117 users (leave-1-out, 99 sampled negatives per user, seed 42). See `src/bxbench/eval.py` docstring for the sampled-negatives protocol and its stated limitations.

## Cost proxies

- Training time: 18.04 s (ALS, factors=64, regularization=0.01, alpha=40.0, iterations=15, over 1,044,497 rows)
- Mean inference latency: 0.050 ms/request (scoring 100 candidates as a single 64-dim dot product per candidate against the user's learned factor vector)
- Retrain cadence: full ALS refit needed on new data (not incremental) — the most expensive tier to retrain so far; how much this cost scales with catalog/user growth is worth tracking against the lift it buys over Tier 2.

## Coverage

- Personalized (vs. popularity-fallback): 100.0% overall. Unlike Tier 2's signal-based fallback, ALS assigns every user seen during training a real (if noisy, for sparse users) latent vector, so fallback here only triggers for a user absent from train entirely — expected to be ~0% given the split guarantees every included user has >=1 train interaction.

## Metrics — overall

| scope | n_users | personalized | hit@5 | precision@5 | recall@5 | ndcg@5 | hit@10 | precision@10 | recall@10 | ndcg@10 | hit@20 | precision@20 | recall@20 | ndcg@20 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| overall | 46,117 | 100% | 0.3640 | 0.0728 | 0.3640 | 0.2989 | 0.4147 | 0.0415 | 0.4147 | 0.3153 | 0.4638 | 0.0232 | 0.4638 | 0.3277 |

## Metrics — by train-history-depth bucket

This is the finding, not the row above — the aggregate hides the bucketed breakdown.

| bucket | n_users | personalized | hit@5 | precision@5 | recall@5 | ndcg@5 | hit@10 | precision@10 | recall@10 | ndcg@10 | hit@20 | precision@20 | recall@20 | ndcg@20 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1-2 | 19,036 | 100% | 0.3747 | 0.0749 | 0.3747 | 0.3135 | 0.4234 | 0.0423 | 0.4234 | 0.3292 | 0.4724 | 0.0236 | 0.4724 | 0.3416 |
| 3-5 | 9,698 | 100% | 0.4007 | 0.0801 | 0.4007 | 0.3325 | 0.4543 | 0.0454 | 0.4543 | 0.3498 | 0.5027 | 0.0251 | 0.5027 | 0.3620 |
| 6-20 | 10,562 | 100% | 0.3778 | 0.0756 | 0.3778 | 0.3055 | 0.4289 | 0.0429 | 0.4289 | 0.3221 | 0.4752 | 0.0238 | 0.4752 | 0.3338 |
| 21+ | 6,821 | 100% | 0.2605 | 0.0521 | 0.2605 | 0.2003 | 0.3118 | 0.0312 | 0.3118 | 0.2170 | 0.3670 | 0.0183 | 0.3670 | 0.2308 |
