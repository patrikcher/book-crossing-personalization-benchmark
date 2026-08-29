"""Fit + evaluate the Tier 1 popularity baseline on the leave-k-out split.

Requires scripts/make_split.py to have been run first (reads
data/processed/{train,test}.parquet).

Run: python scripts/eval_baseline.py
"""

import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bxbench.eval import DEFAULT_N_NEG, DEFAULT_SEED, evaluate  # noqa: E402
from bxbench.models.popularity import PopularityModel  # noqa: E402
from bxbench.report import render_eval_report  # noqa: E402

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
REPORT_PATH = Path(__file__).resolve().parents[1] / "reports" / "eval_popularity.md"
PER_USER_PATH = DATA_DIR / "eval_popularity_per_user.parquet"


def main() -> None:
    train = pd.read_parquet(DATA_DIR / "train.parquet")
    test = pd.read_parquet(DATA_DIR / "test.parquet")

    model = PopularityModel()
    t0 = time.perf_counter()
    model.fit(train)
    train_time_s = time.perf_counter() - t0

    result = evaluate(model, train, test)

    report = render_eval_report(
        title="Eval: Tier 1 — Popularity Baseline",
        result=result,
        n_neg=DEFAULT_N_NEG,
        seed=DEFAULT_SEED,
        cost_proxy_lines=[
            f"- Training time: {train_time_s*1000:.1f} ms (single groupby over {len(train):,} rows)",
            f"- Mean inference latency: {result.mean_inference_latency_ms:.3f} ms/request "
            f"(model.score_items() only, scoring 100 candidates — excludes negative-sampling "
            f"overhead, which is an eval-harness cost, not something a production system pays; "
            f"a real request scores real candidates, it doesn't sample fake ones)",
            "- Retrain cadence: trivial to retrain on every new interaction (O(1) count update); no meaningful staleness cost",
        ],
        coverage_note=(
            f"- Personalized (vs. popularity-fallback): {result.overall['personalized']*100:.1f}% "
            f"— Tier 1 *is* the popularity model, so this is 0% by construction across every bucket. "
            f"Recorded here as the reference point Tiers 2-4 get compared against."
        ),
    )

    REPORT_PATH.parent.mkdir(exist_ok=True)
    REPORT_PATH.write_text(report)
    result.per_user.to_parquet(PER_USER_PATH, index=False)
    print(report)
    print(f"\nWrote {REPORT_PATH}")
    print(f"Wrote {PER_USER_PATH}")


if __name__ == "__main__":
    main()
