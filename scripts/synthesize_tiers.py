"""Cross-tier synthesis: pull Tier 1-4 metrics, coverage, cost proxies, and
already-computed significance results into one comparison view.

Reads (doesn't refit anything -- all 4 tiers were already fit and evaluated
by their own scripts/eval_<tier>.py):
- data/processed/eval_<tier>_per_user.parquet -- per-user metrics
- reports/eval_<tier>.md -- parses the Training time / Mean inference
  latency lines (cost proxies aren't saved anywhere more structured yet)
- reports/significance_<a>_vs_<b>_ndcg10.md -- parses the per-bucket
  p-values from the Wilcoxon tests already run this session

Requires all of scripts/eval_baseline.py, eval_itemcf.py, eval_mf.py,
eval_hybrid.py, and compare_tiers.py (popularity vs itemcf, itemcf vs mf,
mf vs hybrid) to have been run first.

Run: python scripts/synthesize_tiers.py
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bxbench.split import BUCKET_LABELS  # noqa: E402

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
REPORT_DIR = Path(__file__).resolve().parents[1] / "reports"
OUT_PATH = REPORT_DIR / "cross_tier_summary.md"

TIERS = [
    ("popularity", "Tier 1: Popularity"),
    ("itemcf", "Tier 2: Item-CF"),
    ("mf", "Tier 3: Matrix Factorization"),
    ("hybrid", "Tier 4: Hybrid/Content-Aware"),
]

SIGNIFICANCE_PAIRS = [
    ("popularity", "itemcf"),
    ("itemcf", "mf"),
    ("mf", "hybrid"),
]


def parse_cost_proxies(tier_slug: str) -> dict:
    path = REPORT_DIR / f"eval_{tier_slug}.md"
    if not path.exists():
        return {"train_ms": None, "latency_ms": None}
    text = path.read_text()

    train_ms = latency_ms = None
    for line in text.splitlines():
        if "Training time:" in line and train_ms is None:
            token = line.split("Training time:")[1].strip().split()[0:2]
            value, unit = token[0], token[1]
            train_ms = float(value) * (1000 if unit == "s" else 1)
        if "Mean inference latency:" in line and latency_ms is None:
            token = line.split("Mean inference latency:")[1].strip().split()[0]
            latency_ms = float(token)
    return {"train_ms": train_ms, "latency_ms": latency_ms}


def format_train_time(train_ms: float | None) -> str:
    if train_ms is None:
        return "—"
    return f"{train_ms:.0f} ms" if train_ms < 1000 else f"{train_ms/1000:.2f} s"


def load_bucket_metrics(tier_slug: str) -> pd.DataFrame:
    df = pd.read_parquet(DATA_DIR / f"eval_{tier_slug}_per_user.parquet")
    metric_cols = [c for c in df.columns if "@" in c] + ["personalized"]
    return df.groupby("bucket", observed=False)[metric_cols].mean().reindex(BUCKET_LABELS)


def parse_significance(tier_a: str, tier_b: str, metric_slug: str = "ndcg10") -> dict:
    path = REPORT_DIR / f"significance_{tier_a}_vs_{tier_b}_{metric_slug}.md"
    if not path.exists():
        return {}
    rows: dict[str, dict] = {}
    for line in path.read_text().splitlines():
        if not line.startswith("|") or line.startswith("|---"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 8 or cells[0] not in BUCKET_LABELS:
            continue
        bucket, _n, _a, _b, _delta, _stat, p, note = cells[:8]
        rows[bucket] = {"p": p, "significant": note.lower().startswith("significant")}
    return rows


def render_metric_table(bucket_data: dict, metric: str, title: str) -> list[str]:
    lines = [f"## {title}\n"]
    lines.append("| bucket | " + " | ".join(name for _, name in TIERS) + " |")
    lines.append("|---|" + "---|" * len(TIERS))
    for b in BUCKET_LABELS:
        cells = [f"{bucket_data[slug].loc[b, metric]:.4f}" for slug, _ in TIERS]
        lines.append(f"| {b} | " + " | ".join(cells) + " |")
    lines.append("")
    return lines


def main() -> None:
    bucket_data = {slug: load_bucket_metrics(slug) for slug, _ in TIERS}

    lines = ["# Cross-Tier Synthesis\n"]
    lines.append(
        "All 4 tiers, same leave-1-out split, same sampled-negatives eval "
        "protocol (99 negatives, seed 42), same history-depth buckets. "
        "Pulled from each tier's own report — nothing refit here.\n"
    )

    lines += render_metric_table(bucket_data, "hit@10", "hit@10 by bucket")
    lines += render_metric_table(bucket_data, "ndcg@10", "ndcg@10 by bucket")

    lines.append("## Coverage: personalized vs. popularity-fallback\n")
    lines.append("| bucket | " + " | ".join(name for _, name in TIERS) + " |")
    lines.append("|---|" + "---|" * len(TIERS))
    for b in BUCKET_LABELS:
        cells = [f"{bucket_data[slug].loc[b, 'personalized']*100:.0f}%" for slug, _ in TIERS]
        lines.append(f"| {b} | " + " | ".join(cells) + " |")
    lines.append(
        "\nNote: Tier 4's coverage number here reflects MF's personalization "
        "(inherited, ~100%), not content-profile coverage, which is far lower "
        "and reported separately in `reports/eval_hybrid.md`.\n"
    )

    lines.append("## Cost proxies\n")
    lines.append("| tier | training time | inference latency (ms/request) |")
    lines.append("|---|---|---|")
    for slug, name in TIERS:
        cost = parse_cost_proxies(slug)
        lat = f"{cost['latency_ms']:.3f}" if cost["latency_ms"] is not None else "—"
        lines.append(f"| {name} | {format_train_time(cost['train_ms'])} | {lat} |")
    lines.append("")

    for metric_slug, metric_label in [("ndcg10", "ndcg@10"), ("hit10", "hit@10")]:
        lines.append(f"## Significance (paired Wilcoxon, {metric_label}, per bucket)\n")
        lines.append("| comparison | " + " | ".join(BUCKET_LABELS) + " |")
        lines.append("|---|" + "---|" * len(BUCKET_LABELS))
        for a, b in SIGNIFICANCE_PAIRS:
            sig = parse_significance(a, b, metric_slug)
            cells = []
            for bucket in BUCKET_LABELS:
                row = sig.get(bucket)
                if row is None:
                    cells.append("—")
                else:
                    verdict = "significant" if row["significant"] else "not significant"
                    p_str = row["p"] if row["p"].startswith("<") else f"={row['p']}"
                    cells.append(f"p{p_str} ({verdict})")
            lines.append(f"| {a} vs {b} | " + " | ".join(cells) + " |")
        lines.append("")
    lines.append(
        "Note: hit@10 and ndcg@10 don't always agree on which tier wins a bucket "
        "(they don't for item-CF vs. MF in 6-20/21+ — see README `## Results` for "
        "why that's a real finding, not an inconsistency to paper over).\n"
    )

    OUT_PATH.write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"\nWrote {OUT_PATH}")


if __name__ == "__main__":
    main()
