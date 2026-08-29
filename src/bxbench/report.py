"""Shared eval-report rendering, used by every scripts/eval_*.py.

One source of truth for the report format so Tier 1-4 output tables line up
and are directly comparable.
"""

import pandas as pd

from .eval import EvalResult

METRIC_ORDER = [
    "hit@5", "precision@5", "recall@5", "ndcg@5",
    "hit@10", "precision@10", "recall@10", "ndcg@10",
    "hit@20", "precision@20", "recall@20", "ndcg@20",
]


def render_metrics_table(df: pd.DataFrame, index_col: str) -> list[str]:
    cols = [index_col, "n_users", "personalized"] + METRIC_ORDER
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, row in df.iterrows():
        cells = [
            str(row[index_col]),
            f"{int(row['n_users']):,}",
            f"{row['personalized']*100:.0f}%",
        ]
        cells += [f"{row[m]:.4f}" for m in METRIC_ORDER]
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def render_eval_report(
    title: str,
    result: EvalResult,
    n_neg: int,
    seed: int,
    cost_proxy_lines: list[str],
    coverage_note: str,
) -> str:
    lines = [f"# {title}\n"]
    lines.append(
        f"Evaluated on {result.n_users_evaluated:,} users (leave-1-out, "
        f"{n_neg} sampled negatives per user, seed {seed}). "
        f"See `src/bxbench/eval.py` docstring for the sampled-negatives "
        f"protocol and its stated limitations.\n"
    )
    lines.append("## Cost proxies\n")
    lines += cost_proxy_lines
    lines.append("")
    lines.append("## Coverage\n")
    lines.append(coverage_note)
    lines.append("")
    lines.append("## Metrics — overall\n")
    overall_df = pd.DataFrame([result.overall])
    overall_df.insert(0, "scope", "overall")
    lines += render_metrics_table(overall_df, "scope")
    lines.append("")
    lines.append("## Metrics — by train-history-depth bucket\n")
    lines.append(
        "This is the finding, not the row above — the aggregate hides the "
        "bucketed breakdown.\n"
    )
    lines += render_metrics_table(result.per_bucket, "bucket")
    lines.append("")
    return "\n".join(lines)
