"""Ranking evaluation: Precision/Recall/NDCG@K with sampled negatives,
bucketed by train-history depth, with per-user coverage tracking.

**Known limitation, state this in the writeup rather than hiding it:** no true
negative signal exists in this dataset (only "read," never "shown and
skipped"). Ranking eval here uses the standard leave-one-out-with-sampled-
negatives protocol (He et al.-style): for each test user, rank their held-out
item against N_NEG items drawn uniformly at random from items they never
interacted with (train or test), and check where the held-out item lands.
This measures "how well does the model separate the held-out item from
random unseen items" — an assumption stacked on top of the data, not a
ground-truth negative signal. Findings speak to borrow/checkout-level
personalization value, not click/browse-level.

Because each test user has exactly one held-out positive item (K_TEST=1 in
split.py), Recall@K reduces to Hit-Rate@K (0 or 1), and Precision@K = Recall@K
/ K. NDCG@K additionally credits *where* in the top-K the hit landed.
"""

import time
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd

from .models.base import RankingModel
from .split import BUCKET_BINS, BUCKET_LABELS, bucketize_counts

DEFAULT_K_LIST = (5, 10, 20)
DEFAULT_N_NEG = 99
DEFAULT_SEED = 42


@dataclass
class EvalResult:
    per_user: pd.DataFrame  # one row per test user: rank, hits, personalized
    per_bucket: pd.DataFrame  # aggregated metrics by history-depth bucket
    overall: pd.Series  # aggregated metrics across all test users
    mean_inference_latency_ms: float
    n_users_evaluated: int
    # populated only for user_ids passed via evaluate(..., trace_users=...):
    # user_id -> {"candidates": [ISBN...], "scores": [float...], "rank": int}.
    # Captured for free during the same loop every user goes through, purely
    # for human/notebook inspection -- doesn't change scoring or metrics.
    traces: dict = field(default_factory=dict)


def _ndcg_at_k(rank: int, k: int) -> float:
    if rank >= k:
        return 0.0
    return 1.0 / np.log2(rank + 2)  # rank is 0-indexed; +2 so top hit = log2(2)=1


def _sample_negatives(
    rng: np.random.Generator,
    item_pool: np.ndarray,
    exclude: set,
    n_neg: int,
) -> list:
    negs: list = []
    seen = set()
    # oversample and filter; loop only in the (rare) case the pool is small
    # relative to n_neg or exclusion set is large
    while len(negs) < n_neg:
        draw = rng.choice(item_pool, size=n_neg * 2, replace=True)
        for item in draw:
            if item not in exclude and item not in seen:
                seen.add(item)
                negs.append(item)
                if len(negs) == n_neg:
                    break
    return negs


def evaluate(
    model: RankingModel,
    train: pd.DataFrame,
    test: pd.DataFrame,
    k_list: Sequence[int] = DEFAULT_K_LIST,
    n_neg: int = DEFAULT_N_NEG,
    seed: int = DEFAULT_SEED,
    trace_users: set | None = None,
) -> EvalResult:
    rng = np.random.default_rng(seed)
    item_pool = train["ISBN"].unique()
    train_items_by_user = train.groupby("User-ID")["ISBN"].apply(set).to_dict()
    train_depth_by_user = train.groupby("User-ID").size()

    rows: list[dict] = []
    latencies: list[float] = []
    traces: dict = {}

    # NOTE: "User-ID" / "ISBN" aren't valid Python identifiers (hyphen), so
    # itertuples() would silently rename them to positional fields -- zip
    # over the raw columns instead of relying on attribute access.
    for user_id, pos_item in zip(test["User-ID"], test["ISBN"]):
        seen = train_items_by_user.get(user_id, set()) | {pos_item}
        negs = _sample_negatives(rng, item_pool, seen, n_neg)
        candidates = [pos_item] + negs

        t0 = time.perf_counter()
        scores = model.score_items(user_id, candidates)
        latencies.append((time.perf_counter() - t0) * 1000)

        order = np.argsort(-np.asarray(scores, dtype=float))
        rank = int(np.where(order == 0)[0][0])  # position of pos_item (index 0)

        if trace_users is not None and user_id in trace_users:
            traces[user_id] = {
                "candidates": list(candidates),
                "scores": [float(s) for s in scores],
                "rank": rank,
            }

        rec: dict = {
            "User-ID": user_id,
            "rank": rank,
            "personalized": bool(model.is_personalized(user_id)),
            "train_depth": int(train_depth_by_user.get(user_id, 0)),
        }
        for k in k_list:
            hit = int(rank < k)
            rec[f"hit@{k}"] = hit
            rec[f"precision@{k}"] = hit / k
            rec[f"recall@{k}"] = hit
            rec[f"ndcg@{k}"] = _ndcg_at_k(rank, k)
        rows.append(rec)

    per_user = pd.DataFrame(rows)
    per_user["bucket"] = pd.cut(
        per_user["train_depth"], bins=BUCKET_BINS, labels=BUCKET_LABELS
    )

    metric_cols = [c for c in per_user.columns if "@" in c] + ["personalized"]
    per_bucket = (
        per_user.groupby("bucket", observed=False)[metric_cols]
        .mean()
        .reindex(BUCKET_LABELS)
    )
    per_bucket.insert(0, "n_users", per_user.groupby("bucket", observed=False).size().reindex(BUCKET_LABELS))
    overall = per_user[metric_cols].mean()
    overall["n_users"] = len(per_user)

    return EvalResult(
        per_user=per_user,
        per_bucket=per_bucket.reset_index(),
        overall=overall,
        mean_inference_latency_ms=float(np.mean(latencies)) if latencies else float("nan"),
        n_users_evaluated=len(per_user),
        traces=traces,
    )
