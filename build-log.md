# Book-Crossing Personalization-Tier Benchmark — Build Log

**Project:** How much does personalization complexity actually pay off for a book recommender, and where does it stop being worth it, under realistic data sparsity? (Empirical Benchmark)

**Constraint (per §1.4 step 2):** At what point does added personalization complexity — popularity baseline → item-based CF → matrix factorization → hybrid/content-aware — stop earning its keep, given realistic data sparsity and cold-start users? Measured as marginal lift (Precision@K / Recall@K / NDCG@K) per user-history-depth bucket, checked against a cost proxy (training time, inference latency, retrain cadence) and validated with paired significance testing (Wilcoxon signed-rank) per bucket, since sparse buckets will have high variance.

**Dataset:** Book-Crossing, Kaggle "book-crossing-descriptions-from-google-api" — BX-Users (User-ID, Location, Age), BX-Books (ISBN, Title, Author, Year, Publisher, + Google Books API description), BX-Book-Ratings (User-ID, ISBN, Book-Rating 0–10, where 0 = implicit interaction / no rating given, 1–10 = explicit rating). ~1.15M ratings, ~279K users, ~271K books, heavily sparse/long-tailed.

**Methodology notes carried from scoping:**
- Treat as an implicit-feedback ranking problem (collapse 0–10 down to interacted/not-interacted) rather than a rating-prediction problem — closer analogue to a real borrow/checkout signal than to star ratings. Use implicit-feedback methods (ALS-for-implicit or BPR), not classic RMSE-optimized SVD.
- No true negative signal exists in this dataset (only "read," never "shown and skipped") — negative sampling for ranking eval is an assumption to disclose explicitly in the write-up, not bury.
- Findings speak to borrow/checkout-level personalization value, not click/browse-level — the dataset has no clickstream or dwell-time granularity. State this as an explicit boundary of the conclusion.
- Bucket users by training-set history size (e.g. 1–2 / 3–5 / 6–20 / 21+ interactions) and report metrics, coverage (personalized vs. popularity-fallback), and lift *per bucket*, not just in aggregate — the aggregate number would hide the actual finding.

**Employment-safety verdict (§0): CLEAR.** This project is the sanctioned pattern itself — a real work question (required personalization level for a recommendation engine) rebuilt from scratch on fully public data (Book-Crossing), with no employer data, metrics, or screenshots involved at any point. Discipline required going forward: the published piece must never name the employer or the internal system, and must be framed only as personal curiosity ("I wanted to understand how much personalization actually pays off, so I built a toy version") — never as "this informed a real decision at work," even though internally it will. Re-check this framing at draft time.

**Archetype:** Empirical Benchmark

**Target cadence (per §3.3, aimed at a Sep 2026 finish):**
- Week 1: repo + data profiling, train/test split (leave-k-out per user), popularity baseline, eval harness (Precision/Recall/NDCG@K, history-depth buckets, coverage tracking).
- Week 2: item-based CF and matrix factorization (implicit-feedback variants), first per-bucket lift results.
- Week 3: hybrid/content-aware tier using the Google Books descriptions (e.g. content embeddings blended with CF signal); full cost-vs-lift analysis; Wilcoxon significance checks per bucket.
- Week 4: finalize repo (README, one-line verdict, reproducible run instructions per §3.4), draft article from this log, publish, then repurpose for LinkedIn.

---

## Attempts log

*(Nothing yet — entries go here as work happens: timestamp, what was tried, why, and the actual number/result, including dead ends.)*
