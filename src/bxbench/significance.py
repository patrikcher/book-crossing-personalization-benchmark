"""Paired significance testing between two tiers' per-user eval results.

A raw metric lift between tiers isn't trustworthy on its own -- sparse
buckets especially can show a large mean difference that's just noise from
a handful of users. This runs a paired Wilcoxon signed-rank test on
per-user metric deltas, per bucket, so a lift can be called real rather
than asserted.

No multiple-comparison correction is applied across the 4 buckets -- state
that explicitly if it matters for a borderline p-value.
"""

import pandas as pd
from scipy import stats

from .split import BUCKET_LABELS

MIN_N_FOR_TEST = 10  # below this, a Wilcoxon p-value isn't meaningfully interpretable


def compare_tiers(
    per_user_a: pd.DataFrame,
    per_user_b: pd.DataFrame,
    metric: str,
    label_a: str = "a",
    label_b: str = "b",
) -> pd.DataFrame:
    """Paired Wilcoxon signed-rank test on `metric`, per bucket, between two
    tiers' per-user eval results. Both frames must come from the same split
    (same User-ID set, same bucket assignment) -- this is checked, not assumed.
    """
    a = per_user_a[["User-ID", "bucket", metric]].rename(columns={metric: "a"})
    b = per_user_b[["User-ID", "bucket", metric]]
    b = b[["User-ID", metric]].rename(columns={metric: "b"})
    merged = a.merge(b, on="User-ID", how="inner")
    if len(merged) != len(a) or len(merged) != len(b):
        raise ValueError(
            f"per-user frames don't match 1:1 on User-ID ({len(a)} vs {len(b)} vs "
            f"{len(merged)} merged) -- are these from the same train/test split?"
        )

    rows = []
    for bucket in BUCKET_LABELS:
        sub = merged[merged["bucket"] == bucket]
        n = len(sub)
        delta = sub["b"] - sub["a"]
        mean_a, mean_b = sub["a"].mean(), sub["b"].mean()
        stat = p = None

        if n < MIN_N_FOR_TEST:
            note = f"n={n} < {MIN_N_FOR_TEST} -- too small for a meaningful Wilcoxon result, not tested"
        elif (delta == 0).all():
            note = "all per-user deltas are exactly zero -- no measurable difference"
        else:
            try:
                stat, p = stats.wilcoxon(delta)
            except ValueError as e:
                note = f"wilcoxon failed: {e}"
            else:
                note = "significant (p<0.05)" if p < 0.05 else "not significant (p>=0.05)"

        rows.append(
            {
                "bucket": bucket,
                "n_users": n,
                f"mean_{label_a}": mean_a,
                f"mean_{label_b}": mean_b,
                "mean_delta": mean_b - mean_a,
                "wilcoxon_stat": stat,
                "p_value": p,
                "note": note,
            }
        )
    return pd.DataFrame(rows)
