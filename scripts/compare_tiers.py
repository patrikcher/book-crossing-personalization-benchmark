"""Paired Wilcoxon significance test between two tiers' saved per-user eval
results, per train-history-depth bucket, on a chosen metric.

Requires both tiers' eval_*.py scripts to have been run first (each saves
data/processed/eval_<tier>_per_user.parquet).

Run: python scripts/compare_tiers.py popularity itemcf --metric ndcg@10
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bxbench.significance import compare_tiers  # noqa: E402

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
REPORT_DIR = Path(__file__).resolve().parents[1] / "reports"


def render(result: pd.DataFrame, tier_a: str, tier_b: str, metric: str) -> str:
    lines = [f"# Significance: {tier_a} vs {tier_b} on {metric}\n"]
    lines.append(
        "Paired Wilcoxon signed-rank test on per-user metric deltas, per "
        "train-history-depth bucket, run before treating the raw metric lift "
        "as real rather than noise — sparse buckets especially need this. "
        "No multiple-comparison correction applied across the 4 buckets.\n"
    )
    cols = list(result.columns)
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("|" + "---|" * len(cols))
    for _, row in result.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            if c == "n_users":
                cells.append(f"{int(v):,}")
            elif c == "p_value" and pd.notna(v) and v < 0.0001:
                # "0.0000" reads as exactly zero; a paired Wilcoxon p-value
                # never actually is, so say "very small" instead of rounding
                # it away.
                cells.append("<0.0001")
            elif isinstance(v, float):
                cells.append(f"{v:.4f}" if pd.notna(v) else "—")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("tier_a", help="e.g. popularity")
    parser.add_argument("tier_b", help="e.g. itemcf")
    parser.add_argument("--metric", default="ndcg@10")
    args = parser.parse_args()

    a = pd.read_parquet(DATA_DIR / f"eval_{args.tier_a}_per_user.parquet")
    b = pd.read_parquet(DATA_DIR / f"eval_{args.tier_b}_per_user.parquet")

    result = compare_tiers(a, b, args.metric, label_a=args.tier_a, label_b=args.tier_b)
    report = render(result, args.tier_a, args.tier_b, args.metric)

    metric_slug = args.metric.replace("@", "")
    out_path = REPORT_DIR / f"significance_{args.tier_a}_vs_{args.tier_b}_{metric_slug}.md"
    out_path.write_text(report)
    print(report)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
