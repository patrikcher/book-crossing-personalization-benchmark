# Book-Crossing Personalization-Tier Benchmark

**Status: in progress — Tiers 1-2 of 4 complete.** The verdict, full comparison table, and trade-off section below get filled in once all four tiers are done.

## What this is

A benchmark testing how much personalization complexity actually pays off for a book recommender, and where it stops being worth it. This is built out of personal curiosity, on public data. Four model tiers, evaluated on the same eval set throughout:

1. Popularity baseline
2. Item-based collaborative filtering
3. Matrix factorization (implicit-feedback variant)
4. Hybrid / content-aware (adds book description embeddings)

The question driving it: at what point does added personalization complexity stop earning its keep, given realistic data sparsity and cold-start users?

It's a common pattern in recommender-system work to reach straight for the most sophisticated approach available, embeddings, hybrid models, real-time re-ranking, without first testing whether the added complexity earns its keep over something much simpler. That complexity has real, measurable costs (training time, inference latency, retrain cadence, engineering surface area), and those costs are easy to justify in the abstract and much harder to justify against an actual measured lift. This benchmark exists to make that trade-off concrete on a public dataset, rather than assert it.

## Model tiers

Each tier is a strictly bigger step in modeling complexity than the last, evaluated with the exact same harness so the comparison is apples-to-apples.

1. **Popularity baseline** — recommends the same globally most-interacted-with books to every user, ranked by raw train-set interaction count. Zero personalization by construction; this is the floor every other tier has to beat, and the reference point coverage tracking gets measured against. Results: `reports/eval_popularity.md`

2. **Item-based collaborative filtering** — scores a candidate book as the summed cosine similarity between it and every book already in a user's train history, where each book's "vector" is which users interacted with it. Falls back to the popularity score for the minority of users whose history has literally zero co-occurrence with anything else in the catalog (a measured fallback rate, not an assumed one). Results: `reports/eval_itemcf.md`

3. **Matrix factorization (implicit-feedback variant)** — learns latent user and item vectors directly from the interaction matrix (ALS-for-implicit) instead of hand-computing similarity. Can pick up patterns item-CF structurally can't — e.g. two books that were never read by the same person but appeal to the same kind of reader. Not yet built.

4. **Hybrid / content-aware** — blends the matrix-factorization signal with content embeddings of book descriptions, aimed at making reasonable recommendations for books with little or no interaction history by leaning on what a book is about rather than who's read it. Bounded by the description-coverage limitation noted below. Not yet built.

## Dataset

[Book-Crossing + Google Books API descriptions](https://www.kaggle.com/datasets/mostafanofal/book-crossing-descriptions-from-google-api) (Kaggle).

- `BX-Users.csv` - User-ID, Location, Age (self-reported, ~40% null)
- `BX-Books.csv` — ISBN, Book-Title, Book-Author, Year-Of-Publication, Publisher, Image-URL-S/M/L
- `Books_Descriptions.csv` — ISBN, description (Google Books API), joined by ISBN — covers only 2.6% of the catalog
- `BX-Book-Ratings.csv` — User-ID, ISBN, Book-Rating (0-10); 0 = implicit interaction, 1-10 = explicit rating
- ~1.15M ratings, ~279K users, ~271K books. Extremely sparse and long-tailed — most users have only a few interactions, and that skew is the point of the dataset here, not noise to filter out.

## Task and eval design

Treated as **implicit-feedback ranking**, not rating prediction: Book-Rating is collapsed to a binary interacted/did-not-interact signal, since the real-world analogue this is meant to generalize to is a borrow/checkout-style signal rather than a star rating.

- **Split:** leave-1-out per user. Each user's single most-recent interaction (by row order; the dataset has no timestamp) held out as test. Users with fewer than 2 total interactions can't be split at all and are excluded, but tracked separately as "pure cold" users rather than silently dropped.
- **Metrics:** Precision@K, Recall@K, NDCG@K, computed **per user-history-depth bucket** (1-2 / 3-5 / 6-20 / 21+ training interactions), not just in aggregate. The bucketed breakdown is the finding; the aggregate number hides it.
- **Ranking protocol:** each test user's held-out item is ranked against 99 negatives sampled uniformly at random from items they never interacted with. This is a standard but assumption-laden protocol. See Limitations below.
- **Coverage tracking:** every model reports, per user, whether its recommendation came from real personalized signal or fell back to popularity. Measured directly, not assumed.
- **Cost proxies:** training time, inference latency, retrain cadence — logged per tier.
- **Significance:** paired Wilcoxon signed-rank test on per-user metric deltas, per bucket, before any lift between tiers is treated as real rather than noise.

## Limitations

- **No true negative signal exists in this dataset** — only "read", never "shown and skipped". Ranking eval therefore relies on sampled negatives, which is an assumption, not a fact. Findings speak to borrow/checkout-level personalization value, not click/browse-level, since there's no clickstream or dwell-time data here.
- **Book descriptions cover only 7,021 of 271,379 books (2.6%)** of the catalog, capping how much of the catalog Tier 4 can use content signal for. Coverage was checked and found to be roughly flat across popularity buckets (not concentrated on popular books), so it isn't structurally biased away from the cold-start tail. But it's still a real ceiling on Tier 4's reach.
- **"Most recent" interaction is approximated by row order** in the raw CSV, since there's no timestamp column.

## Reproducing the results

```bash
# 1. Set up environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt   # add -r requirements-content.txt too for Tier 4

# 2. Download the dataset (requires a Kaggle account + API token, ~/.kaggle/kaggle.json)
kaggle datasets download mostafanofal/book-crossing-descriptions-from-google-api -p data/raw --unzip

# 3. Profile the raw data (writes reports/data_profile.md)
jupyter nbconvert --to notebook --execute --inplace notebooks/profile_data.ipynb

# 4. Build the leave-1-out train/test split (writes data/processed/, reports/split_summary.md)
python scripts/make_split.py

# 5. Fit + evaluate each tier (writes reports/eval_<tier>.md and data/processed/eval_<tier>_per_user.parquet)
python scripts/eval_baseline.py   # Tier 1: popularity
python scripts/eval_itemcf.py     # Tier 2: item-based CF

# 6. Paired significance test between two tiers, per bucket
python scripts/compare_tiers.py popularity itemcf --metric ndcg@10
```

Every number in the eventual write-up should be reproducible from this sequence. Raw data and generated `data/processed/` outputs are gitignored (~123MB). Step 2 regenerates them.

## Repo layout

- `src/bxbench/` — importable library: data loaders, split logic, eval harness, models, significance testing, report rendering
- `scripts/` — pipeline runners (split, per-tier eval, tier comparison)
- `notebooks/` — exploratory/profiling work
- `reports/` — generated markdown reports (the numbers behind the write-up)
- `data/` — raw and processed data (gitignored)

## Results

*(Coming as tiers complete — see `reports/` for what's generated so far.)*
