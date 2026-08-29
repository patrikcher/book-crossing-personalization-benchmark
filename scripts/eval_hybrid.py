"""Fit + evaluate Tier 4 (hybrid / content-aware) on the leave-k-out split.

Requires scripts/make_split.py to have been run first (reads
data/processed/{train,test}.parquet).

Run: python scripts/eval_hybrid.py
"""

import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bxbench.data import load_descriptions  # noqa: E402
from bxbench.eval import DEFAULT_N_NEG, DEFAULT_SEED, evaluate  # noqa: E402
from bxbench.models.hybrid import (  # noqa: E402
    DEFAULT_CONTENT_WEIGHT,
    DEFAULT_EMBEDDING_MODEL,
    HybridModel,
)
from bxbench.report import render_eval_report  # noqa: E402
from bxbench.split import BUCKET_LABELS  # noqa: E402

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
REPORT_PATH = Path(__file__).resolve().parents[1] / "reports" / "eval_hybrid.md"
PER_USER_PATH = DATA_DIR / "eval_hybrid_per_user.parquet"


def render_content_coverage(model: HybridModel, per_user: pd.DataFrame) -> str:
    per_user = per_user.copy()
    per_user["has_content_profile"] = per_user["User-ID"].map(model.has_content_profile)
    by_bucket = (
        per_user.groupby("bucket", observed=False)["has_content_profile"]
        .agg(["sum", "count"])
        .reindex(BUCKET_LABELS)
    )
    lines = ["## Content-profile coverage by history-depth bucket\n"]
    lines.append(
        "A user needs >=1 *described* book in their train history to get a "
        "content profile at all. Since description coverage doesn't depend "
        "on a book's popularity (flat ~2.6% regardless), a user's odds of "
        "hitting one scale with how many books they've read -- so this "
        "concentrates in exactly the bucket that needs it least.\n"
    )
    lines.append("| bucket | users with content profile | n_users | coverage |")
    lines.append("|---|---|---|---|")
    for label in BUCKET_LABELS:
        if label in by_bucket.index:
            s, c = by_bucket.loc[label, "sum"], by_bucket.loc[label, "count"]
            lines.append(f"| {label} | {int(s):,} | {int(c):,} | {s/c*100:.1f}% |")
    lines.append(
        f"\n- Content-boost rate (fraction of all scored user/candidate pairs "
        f"that actually got a content-similarity term added, vs. falling "
        f"through to MF alone): {model.content_boost_rate()*100:.2f}%\n"
    )
    return "\n".join(lines)


def main() -> None:
    train = pd.read_parquet(DATA_DIR / "train.parquet")
    test = pd.read_parquet(DATA_DIR / "test.parquet")
    descriptions = load_descriptions()

    model = HybridModel()
    t0 = time.perf_counter()
    model.fit(train, descriptions)
    train_time_s = time.perf_counter() - t0

    result = evaluate(model, train, test)

    report = render_eval_report(
        title="Eval: Tier 4 — Hybrid / Content-Aware",
        result=result,
        n_neg=DEFAULT_N_NEG,
        seed=DEFAULT_SEED,
        cost_proxy_lines=[
            f"- Training time: {train_time_s:.2f} s (MF fit + embedding "
            f"{len(model.item_embeddings):,} book descriptions with "
            f"'{DEFAULT_EMBEDDING_MODEL}' + building per-user content profiles)",
            f"- Mean inference latency: {result.mean_inference_latency_ms:.3f} ms/request "
            f"(MF dot products for 100 candidates + a content dot product for "
            f"whichever of those candidates have a description)",
            "- Retrain cadence: needs both a full MF refit and a full re-embedding pass "
            "over any new/changed descriptions -- the most expensive tier to retrain, "
            "though re-embedding only ~7K short descriptions is itself fast; the MF "
            "refit dominates.",
        ],
        coverage_note=(
            f"- Personalized (vs. popularity-fallback): {result.overall['personalized']*100:.1f}% "
            f"overall -- inherited from Tier 3's MF signal, effectively always on. "
            f"The coverage number that actually matters for this tier is content-profile "
            f"coverage, broken out below, not this one. Content weight={DEFAULT_CONTENT_WEIGHT} "
            f"(z-score-normalized MF score + weight * cosine similarity, where content exists)."
        ),
    )
    report += "\n" + render_content_coverage(model, result.per_user)

    REPORT_PATH.parent.mkdir(exist_ok=True)
    REPORT_PATH.write_text(report)
    result.per_user.to_parquet(PER_USER_PATH, index=False)
    print(report)
    print(f"\nWrote {REPORT_PATH}")
    print(f"Wrote {PER_USER_PATH}")


if __name__ == "__main__":
    main()
