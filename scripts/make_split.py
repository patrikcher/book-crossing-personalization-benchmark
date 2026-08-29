"""Build the leave-k-out train/test split and write it to data/processed/.

Run: python scripts/make_split.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bxbench.data import load_ratings  # noqa: E402
from bxbench.split import bucketize_counts, leave_k_out_split  # noqa: E402

OUT_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
REPORT_PATH = Path(__file__).resolve().parents[1] / "reports" / "split_summary.md"


def main() -> None:
    ratings = load_ratings()
    result = leave_k_out_split(ratings)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    result.train.to_parquet(OUT_DIR / "train.parquet", index=False)
    result.test.to_parquet(OUT_DIR / "test.parquet", index=False)
    result.pure_cold_users.to_parquet(OUT_DIR / "pure_cold.parquet", index=False)

    train_counts = result.train.groupby("User-ID").size()
    bucket_counts = bucketize_counts(train_counts).value_counts().reindex(
        ["1-2", "3-5", "6-20", "21+"]
    )

    lines = ["# Split Summary\n"]
    lines.append(f"- Total users with any rating: {result.n_users_total:,}")
    lines.append(
        f"- Users included in split (>=2 interactions): {result.n_users_split:,} "
        f"({result.n_users_split/result.n_users_total*100:.1f}%)"
    )
    lines.append(
        f"- Pure-cold users excluded (exactly 1 interaction, no train possible): "
        f"{result.n_users_pure_cold:,} "
        f"({result.n_users_pure_cold/result.n_users_total*100:.1f}%)\n"
    )
    lines.append(f"- Train rows: {len(result.train):,}")
    lines.append(f"- Test rows: {len(result.test):,}\n")
    lines.append("## Train-history-depth buckets (of split-included users)\n")
    lines.append("| bucket | # users | % |")
    lines.append("|---|---|---|")
    for label, cnt in bucket_counts.items():
        pct = cnt / result.n_users_split * 100
        lines.append(f"| {label} | {cnt:,} | {pct:.1f}% |")

    REPORT_PATH.parent.mkdir(exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"\nWrote train/test/pure_cold parquet to {OUT_DIR}")
    print(f"Wrote {REPORT_PATH}")


if __name__ == "__main__":
    main()
