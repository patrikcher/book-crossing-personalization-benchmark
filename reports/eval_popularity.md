# Eval: Tier 1 — Popularity Baseline

Evaluated on 46,117 users (leave-1-out, 99 sampled negatives per user, seed 42). See `src/bxbench/eval.py` docstring for the sampled-negatives protocol and its stated limitations.

## Cost proxies

- Training time: 76.1 ms (single groupby over 1,044,497 rows)
- Mean inference latency: 0.101 ms/request (model.score_items() only, scoring 100 candidates — excludes negative-sampling overhead, which is an eval-harness cost, not something a production system pays; a real request scores real candidates, it doesn't sample fake ones)
- Retrain cadence: trivial to retrain on every new interaction (O(1) count update); no meaningful staleness cost

## Coverage

- Personalized (vs. popularity-fallback): 0.0% — Tier 1 *is* the popularity model, so this is 0% by construction across every bucket. Recorded here as the reference point Tiers 2-4 get compared against.

## Metrics — overall

| scope | n_users | personalized | hit@5 | precision@5 | recall@5 | ndcg@5 | hit@10 | precision@10 | recall@10 | ndcg@10 | hit@20 | precision@20 | recall@20 | ndcg@20 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| overall | 46,117 | 0% | 0.3371 | 0.0674 | 0.3371 | 0.2565 | 0.4121 | 0.0412 | 0.4121 | 0.2807 | 0.4915 | 0.0246 | 0.4915 | 0.3008 |

## Metrics — by train-history-depth bucket

This is the finding, not the row above — the aggregate hides the bucketed breakdown.

| bucket | n_users | personalized | hit@5 | precision@5 | recall@5 | ndcg@5 | hit@10 | precision@10 | recall@10 | ndcg@10 | hit@20 | precision@20 | recall@20 | ndcg@20 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1-2 | 19,036 | 0% | 0.4084 | 0.0817 | 0.4084 | 0.3188 | 0.4892 | 0.0489 | 0.4892 | 0.3450 | 0.5679 | 0.0284 | 0.5679 | 0.3649 |
| 3-5 | 9,698 | 0% | 0.3696 | 0.0739 | 0.3696 | 0.2799 | 0.4462 | 0.0446 | 0.4462 | 0.3048 | 0.5241 | 0.0262 | 0.5241 | 0.3245 |
| 6-20 | 10,562 | 0% | 0.2966 | 0.0593 | 0.2966 | 0.2206 | 0.3674 | 0.0367 | 0.3674 | 0.2435 | 0.4500 | 0.0225 | 0.4500 | 0.2644 |
| 21+ | 6,821 | 0% | 0.1548 | 0.0310 | 0.1548 | 0.1048 | 0.2174 | 0.0217 | 0.2174 | 0.1249 | 0.2961 | 0.0148 | 0.2961 | 0.1447 |
