"""Fit + evaluate Tier 2 (item-based CF) on the leave-k-out split.

Requires scripts/make_split.py to have been run first (reads
data/processed/{train,test}.parquet).

Run: python scripts/eval_itemcf.py
"""

import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bxbench.eval import DEFAULT_N_NEG, DEFAULT_SEED, evaluate  # noqa: E402
from bxbench.models.item_cf import ItemCFModel  # noqa: E402
from bxbench.report import render_eval_report  # noqa: E402

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
REPORT_PATH = Path(__file__).resolve().parents[1] / "reports" / "eval_itemcf.md"
PER_USER_PATH = DATA_DIR / "eval_itemcf_per_user.parquet"


def main() -> None:
    train = pd.read_parquet(DATA_DIR / "train.parquet")
    test = pd.read_parquet(DATA_DIR / "test.parquet")

    model = ItemCFModel()
    t0 = time.perf_counter()
    model.fit(train)
    train_time_s = time.perf_counter() - t0

    result = evaluate(model, train, test)

    report = render_eval_report(
        title="Eval: Tier 2 — Item-Based Collaborative Filtering",
        result=result,
        n_neg=DEFAULT_N_NEG,
        seed=DEFAULT_SEED,
        cost_proxy_lines=[
            f"- Training time: {train_time_s:.2f} s (build item x user sparse matrix + "
            f"signal-coverage precheck over {len(train):,} rows)",
            f"- Mean inference latency: {result.mean_inference_latency_ms:.3f} ms/request "
            f"(scoring 100 candidates against a user's train history via sparse cosine "
            f"similarity, or a popularity-fallback lookup when signal-based fallback triggers)",
            "- Retrain cadence: needs the full item x user matrix + signal precheck rebuilt "
            "on new data (not an O(1) incremental update like Tier 1) — still cheap at this "
            "scale, but this is the first tier where retrain cost is nonzero and worth tracking "
            "as the benchmark moves to bigger tiers.",
        ],
        coverage_note=(
            f"- Personalized (vs. popularity-fallback): {result.overall['personalized']*100:.1f}% "
            f"overall. Fallback rule is signal-based (see `src/bxbench/models/item_cf.py` "
            f"docstring), not a fixed interaction-count cutoff — see per-bucket breakdown below "
            f"for where the fallback actually concentrates."
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
