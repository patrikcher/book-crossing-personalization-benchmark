# CLAUDE.md — Book-Crossing Personalization-Tier Benchmark

This file orients any Claude session working in this project folder. Read it before writing code or drafting anything here.

## What this project is

An Empirical Benchmark testing how much personalization complexity actually pays off for a book recommender, and where it stops being worth it. Four model tiers, same eval set throughout:

1. Popularity baseline
2. Item-based collaborative filtering
3. Matrix factorization (implicit-feedback variant)
4. Hybrid / content-aware (adds book description embeddings)

**The one real constraint driving the whole project (do not lose sight of this):** at what point does added personalization complexity stop earning its keep, given realistic data sparsity and cold-start users? Every result should ultimately serve answering that, not just "which model has the highest score."

## Dataset

Kaggle: `book-crossing-descriptions-from-google-api` (Book-Crossing + Google Books API descriptions merged into `BX-Books.csv`).

- `BX-Users.csv` — User-ID, Location, Age (Age is sparse/self-reported, expect many nulls)
- `BX-Books.csv` — ISBN, Book-Title, Book-Author, Year-Of-Publication, Publisher, Image-URL-S/M/L, + Description (Google Books API)
- `BX-Book-Ratings.csv` — User-ID, ISBN, Book-Rating (0–10). **0 means an implicit interaction (logged, not rated); 1–10 is an explicit rating.**
- ~1.15M ratings, ~279K users, ~271K books. Extremely sparse and long-tailed — most users have only a few interactions. This skew is the point of the dataset for this project, not noise to filter out.

## Methodology decisions already made (don't re-litigate without a stated reason)

- **Treat this as implicit-feedback ranking, not rating prediction.** Collapse Book-Rating 0–10 down to a binary "interacted / did not interact" signal for the main benchmark. Use implicit-feedback methods (ALS-for-implicit, or BPR) — not RMSE-optimized SVD. Reasoning: the production system this benchmark is meant to generalize to has borrow/checkout-style signals, not star ratings. The 1–10 explicit subset can be used for a clearly-labeled secondary analysis, never the headline comparison.
- **Eval split:** leave-k-out per user (hold out each user's most recent 1–2 interactions as test).
- **Metrics:** Precision@K, Recall@K, NDCG@K — computed **per user-history-depth bucket** (e.g. 1–2 / 3–5 / 6–20 / 21+ training interactions), not just in aggregate. The aggregate number hides the actual finding; the bucketed breakdown is the finding.
- **Coverage tracking:** for every user, log whether their top-K list came from real personalized signal or a popularity fallback (many CF/MF implementations silently degrade to popularity for near-cold users — measure how often, per bucket, don't assume).
- **Cost proxies:** training time, inference latency per request, retrain cadence — logged per tier, to plot against the lift numbers.
- **Significance:** paired test (Wilcoxon signed-rank on per-user metric deltas) between adjacent tiers, per bucket, before claiming a lift is real — sparse buckets will have high variance and need this check more than dense ones.
- **Known limitation to keep stating, not hide:** no true negative signal exists in this dataset (only "read," never "shown and skipped") — ranking eval requires sampled negatives, which is an assumption, not a fact. Findings speak to borrow/checkout-level personalization value, not click/browse-level, since there's no clickstream or dwell-time data here.

## Employment-safety rule — read this every session

This project was scoped specifically to build intuition for a real personalization decision at Patrick's employer (a library recommendation engine). That is fine as private motivation but **must never appear in any artifact this project produces** — code comments, README, article draft, commit messages, or conversation summaries meant for publishing. Rules:

- Never reference the employer, the internal system, or any internal metric/screenshot/dataset by name or by identifiable description, anywhere in this repo or its outputs.
- The public framing is always personal curiosity: "I wanted to understand how much personalization actually pays off in a recommender, so I built a toy version on Book-Crossing." Never "at work we..." or "this informed a decision at..."
- If in doubt whether something is safe to write into a public-facing file (README, article draft), don't — flag it and ask instead of guessing.

## Content strategy conventions (see the full knowledge base for detail)

Full reference: `../Brand-Voice-Audience-Content-Strategy.md` in this Drive folder. Key things that apply directly to this project:

- **Article skeleton (§1.4):** one-line verdict with a number first → the constraint → what was tried, including what failed → the evidence (full comparison table, not just the winner) → the trade-off and its boundary condition ("use X below threshold Y, switch to Z above it") → so what.
- **Empirical Benchmark credibility bar (§3.2):** state the exact task and eval metric before any results; show the full comparison table, not just the winning row; disclose sample size and eval-design limitations; state what result *would have* changed the conclusion.
- **Build-log habit (§3.1):** log every attempt, dead end, and number to `build-log.md` in this folder *as it happens*, timestamped — not reconstructed afterward. The eventual article is an edit of that log.
- **Voice (§1.2):** precise over hedgy (numbers, not "significant improvement"); skeptical of its own results before presenting a win; economical; opinionated on the trade-off actually earned by the experiment, not on frameworks/vendors in the abstract.

## Repo hygiene

- Keep `build-log.md` updated before moving to the next step, not after.
- README (when written, in Week 4 per the cadence) should itself lead with a one-line verdict, per §3.4.
- Code should be runnable end-to-end from the README instructions — a reader needs to be able to reproduce every number.
