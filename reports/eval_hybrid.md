# Eval: Tier 4 — Hybrid / Content-Aware

Evaluated on 46,117 users (leave-1-out, 99 sampled negatives per user, seed 42). See `src/bxbench/eval.py` docstring for the sampled-negatives protocol and its stated limitations.

## Cost proxies

- Training time: 36.25 s (MF fit + embedding 6,842 book descriptions with 'all-MiniLM-L6-v2' + building per-user content profiles)
- Mean inference latency: 0.081 ms/request (MF dot products for 100 candidates + a content dot product for whichever of those candidates have a description)
- Retrain cadence: needs both a full MF refit and a full re-embedding pass over any new/changed descriptions -- the most expensive tier to retrain, though re-embedding only ~7K short descriptions is itself fast; the MF refit dominates.

## Coverage

- Personalized (vs. popularity-fallback): 100.0% overall -- inherited from Tier 3's MF signal, effectively always on. The coverage number that actually matters for this tier is content-profile coverage, broken out below, not this one. Content weight=1.0 (z-score-normalized MF score + weight * cosine similarity, where content exists).

## Metrics — overall

| scope | n_users | personalized | hit@5 | precision@5 | recall@5 | ndcg@5 | hit@10 | precision@10 | recall@10 | ndcg@10 | hit@20 | precision@20 | recall@20 | ndcg@20 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| overall | 46,117 | 100% | 0.3639 | 0.0728 | 0.3639 | 0.2989 | 0.4147 | 0.0415 | 0.4147 | 0.3153 | 0.4640 | 0.0232 | 0.4640 | 0.3278 |

## Metrics — by train-history-depth bucket

This is the finding, not the row above — the aggregate hides the bucketed breakdown.

| bucket | n_users | personalized | hit@5 | precision@5 | recall@5 | ndcg@5 | hit@10 | precision@10 | recall@10 | ndcg@10 | hit@20 | precision@20 | recall@20 | ndcg@20 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1-2 | 19,036 | 100% | 0.3747 | 0.0749 | 0.3747 | 0.3135 | 0.4235 | 0.0423 | 0.4235 | 0.3292 | 0.4724 | 0.0236 | 0.4724 | 0.3416 |
| 3-5 | 9,698 | 100% | 0.4007 | 0.0801 | 0.4007 | 0.3325 | 0.4545 | 0.0455 | 0.4545 | 0.3498 | 0.5027 | 0.0251 | 0.5027 | 0.3620 |
| 6-20 | 10,562 | 100% | 0.3778 | 0.0756 | 0.3778 | 0.3055 | 0.4288 | 0.0429 | 0.4288 | 0.3221 | 0.4756 | 0.0238 | 0.4756 | 0.3339 |
| 21+ | 6,821 | 100% | 0.2599 | 0.0520 | 0.2599 | 0.2000 | 0.3117 | 0.0312 | 0.3117 | 0.2169 | 0.3674 | 0.0184 | 0.3674 | 0.2309 |

## Content-profile coverage by history-depth bucket

A user needs >=1 *described* book in their train history to get a content profile at all. Since description coverage doesn't depend on a book's popularity (flat ~2.6% regardless), a user's odds of hitting one scale with how many books they've read -- so this concentrates in exactly the bucket that needs it least.

| bucket | users with content profile | n_users | coverage |
|---|---|---|---|
| 1-2 | 457 | 19,036 | 2.4% |
| 3-5 | 662 | 9,698 | 6.8% |
| 6-20 | 1,951 | 10,562 | 18.5% |
| 21+ | 4,395 | 6,821 | 64.4% |

- Content-boost rate (fraction of all scored user/candidate pairs that actually got a content-similarity term added, vs. falling through to MF alone): 0.35%
