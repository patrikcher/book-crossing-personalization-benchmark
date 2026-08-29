# Book-Crossing Personalization-Tier Benchmark

**Verdict: one upgrade pays off, the rest don't.** Moving from "recommend whatever's popular" to a simple collaborative-filtering model gets a big, reliable win, about 30-40% more relevant picks, for barely any extra training cost. Two more upgrades were tried after that, a fancier model, then one that also reads book descriptions, and neither is a clear win: the fancier model only helps a small slice of heavy readers, and the one reading descriptions doesn't measurably help anyone. Full numbers and the exact trade-off in [Results](#results).

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

3. **Matrix factorization (implicit-feedback variant)** — learns latent user and item vectors directly from the interaction matrix (ALS-for-implicit) instead of hand-computing similarity. Can pick up patterns item-CF structurally can't — e.g. two books that were never read by the same person but appeal to the same kind of reader. Results: `reports/eval_mf.md`

4. **Hybrid / content-aware** — blends the matrix-factorization signal with content embeddings of book descriptions, aimed at making reasonable recommendations for books with little or no interaction history by leaning on what a book is about rather than who's read it. Bounded by the description-coverage limitation noted below. Results: `reports/eval_hybrid.md`

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
- **Some ISBNs in `BX-Book-Ratings.csv` don't exist in `BX-Books.csv`.** Train has 318,264 unique ISBNs against a 271,379-book catalog; some are malformed identifiers (e.g. `100940/86`, not a real ISBN format). These interactions are still used (the eval only needs an ISBN to key on, not a valid catalog entry), but a title/metadata lookup for them will come back empty. A data-quality issue in the source dataset, not something this pipeline introduces.

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
python scripts/eval_mf.py         # Tier 3: matrix factorization
python scripts/eval_hybrid.py     # Tier 4: hybrid/content-aware (needs requirements-content.txt)

# 6. Paired significance test between any two tiers, per bucket
python scripts/compare_tiers.py popularity itemcf --metric ndcg@10
python scripts/compare_tiers.py itemcf mf --metric ndcg@10
python scripts/compare_tiers.py mf hybrid --metric ndcg@10

# 7. Cross-tier synthesis (writes reports/cross_tier_summary.md)
python scripts/synthesize_tiers.py
```

To see what the numbers above actually look like in practice — real readers, real book titles, real recommendation lists from each tier — run `notebooks/results_walkthrough.ipynb`. It refits all 4 models and reruns the full eval (so it takes 15-20 minutes), and follows a handful of real readers through the exact same 100-candidate rankings the numbers in `reports/` are built from (see `reports/results_walkthrough.md` for the summary).

Every number in the eventual write-up should be reproducible from this sequence. Raw data and generated `data/processed/` outputs are gitignored (~123MB). Step 2 regenerates them.

## Repo layout

- `src/bxbench/` — importable library: data loaders, split logic, eval harness, models, significance testing, report rendering
- `scripts/` — pipeline runners (split, per-tier eval, tier comparison)
- `notebooks/` — exploratory/profiling work
- `reports/` — generated markdown reports (the numbers behind the write-up)
- `data/` — raw and processed data (gitignored)

## Results

**The question:** how much does personalization complexity actually pay off for a book recommender, and where does it stop being worth it?

**What was tried:** all four tiers, evaluated the same way.

**The evidence:** how often the book a reader actually picked up next showed up anywhere in a tier's top 10 recommendations, split by how many books that reader had already read (full numbers, plus a stricter version of this measurement, in `reports/cross_tier_summary.md`):

| readers with... | share of everyone who's read at least 1 book | Popularity | Simple model (item-CF) | Fancier model (matrix factorization) | Reads descriptions too (hybrid) |
|---|---|---|---|---|---|
| only 1 book ever | 56% | — | — | — | — |
| 1-2 books | 18% | 0.49 | **0.68** | 0.42 | 0.42 |
| 3-5 books | 9% | 0.45 | **0.62** | 0.45 | 0.45 |
| 6-20 books | 10% | 0.37 | **0.47** | 0.43 | 0.43 |
| 21+ books | 6% | 0.22 | 0.31 | 0.31 | 0.31 |

(Readers with only 1 book ever logged — 56% of everyone — can't be evaluated at all: there's nothing to learn from a single data point, for any of the four tiers. The other rows are the fraction of readers where the right book actually showed up in the top 10; higher is better. 21+ is left unbolded on purpose, see below — the two numbers there are too close to call a winner.)

**The simple model is the strongest or tied-strongest choice for every group of readers, on this measurement.** Neither fancier tier ever clearly beats it here.

A *stricter* way of scoring the same recommendations — one that also credits *how close to the top* the right book landed, not just whether it made the top 10 at all — tells a different story for readers with 6+ books of history: there, the fancier model does pull ahead. So for heavy readers, whether the simple model or the fancier one is "better" depends on whether a product needs the right book to show up at all, or to show up *near the top*.

The fourth tier, which also reads book descriptions, doesn't win under either way of scoring, for anyone. Across thousands of readers per group, adding descriptions changed the actual recommendation list for **at most 5 readers in any group** because only 2.6% of books in this dataset have a description to read in the first place.

**The trade-off:** the simple model is the right default. Cheap to build, and never clearly beaten on the most basic measurement (did the right book show up at all). Upgrading to the fancier model costs about 15x more to train. That cost only pays off for the 16.5% of readers with 6+ books of history, and only if a product specifically needs the right book ranked *near the top*, not just present somewhere in the list. For everyone else, it isn't worth it. The description-reading tier isn't worth building at all on this dataset: it costs even more to train (a large new dependency, an extra processing step) and doesn't move a single number that matters.

**So what:** most readers (56%) have logged only one interaction, so there's nothing to learn from and no model can personalize for them. Of everyone else, the simple model already does the job on the measurement that matters most. The expensive model only helps a slice of heavy readers, and only for products that care more about how high a recommendation ranks than whether it shows up at all. Whether an upgrade is worth building depends on who it actually helps, what's being measured, and how much data feeds it — not on how sophisticated it sounds.
