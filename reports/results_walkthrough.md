# Results Walkthrough

A narrative look at the same numbers behind `reports/cross_tier_summary.md` -- real readers, real books, real recommendations. Run `notebooks/results_walkthrough.ipynb` for the full per-reader detail; this file just captures the summary table and the takeaway.

## hit@10 by bucket

| bucket | Tier 1 -- Popularity | Tier 2 -- Item-CF | Tier 3 -- Matrix Factorization | Tier 4 -- Hybrid/Content-Aware |
|---|---|---|---|---|
| 1-2 | 0.489 | 0.676 | 0.423 | 0.423 |
| 3-5 | 0.446 | 0.616 | 0.454 | 0.455 |
| 6-20 | 0.367 | 0.473 | 0.429 | 0.429 |
| 21+ | 0.217 | 0.306 | 0.312 | 0.312 |

Item-CF wins or ties every bucket on hit@10 -- matrix factorization is never significantly ahead of it here, even for deep-history readers. Matrix factorization does pull significantly ahead on ndcg@10 (rank position within the top 10, not just presence in it) from 6+ books of history on -- a real but narrower advantage than a single 'better tier' story would suggest. The hybrid tier makes essentially no visible difference over plain matrix factorization on either metric, because almost none of the books a typical reader has touched have a description on file to draw on.
