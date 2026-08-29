"""Fit + evaluate Tier 3 (implicit-feedback matrix factorization) on the
leave-k-out split.

Requires scripts/make_split.py to have been run first (reads
data/processed/{train,test}.parquet).

Run: python scripts/eval_mf.py
"""

import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bxbench.eval import DEFAULT_N_NEG, DEFAULT_SEED, evaluate  # noqa: E402
from bxbench.models.mf import (  # noqa: E402
    DEFAULT_ALPHA,
    DEFAULT_FACTORS,
    DEFAULT_ITERATIONS,
    DEFAULT_REGULARIZATION,
    MFModel,
)
from bxbench.report import render_eval_report  # noqa: E402

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
REPORT_PATH = Path(__file__).resolve().parents[1] / "reports" / "eval_mf.md"
PER_USER_PATH = DATA_DIR / "eval_mf_per_user.parquet"


def main() -> None:
    train = pd.read_parquet(DATA_DIR / "train.parquet")
    test = pd.read_parquet(DATA_DIR / "test.parquet")

    model = MFModel()
    t0 = time.perf_counter()
    model.fit(train)
    train_time_s = time.perf_counter() - t0

    result = evaluate(model, train, test)

    report = render_eval_report(
        title="Eval: Tier 3 — Matrix Factorization (Implicit-Feedback ALS)",
        result=result,
        n_neg=DEFAULT_N_NEG,
        seed=DEFAULT_SEED,
        cost_proxy_lines=[
            f"- Training time: {train_time_s:.2f} s (ALS, factors={DEFAULT_FACTORS}, "
            f"regularization={DEFAULT_REGULARIZATION}, alpha={DEFAULT_ALPHA}, "
            f"iterations={DEFAULT_ITERATIONS}, over {len(train):,} rows)",
            f"- Mean inference latency: {result.mean_inference_latency_ms:.3f} ms/request "
            f"(scoring 100 candidates as a single 64-dim dot product per candidate against "
            f"the user's learned factor vector)",
            "- Retrain cadence: full ALS refit needed on new data (not incremental) — "
            "the most expensive tier to retrain so far; how much this cost scales with "
            "catalog/user growth is worth tracking against the lift it buys over Tier 2.",
        ],
        coverage_note=(
            f"- Personalized (vs. popularity-fallback): {result.overall['personalized']*100:.1f}% "
            f"overall. Unlike Tier 2's signal-based fallback, ALS assigns every user seen "
            f"during training a real (if noisy, for sparse users) latent vector, so fallback "
            f"here only triggers for a user absent from train entirely — expected to be ~0% "
            f"given the split guarantees every included user has >=1 train interaction."
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
