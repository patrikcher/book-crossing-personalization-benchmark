"""Leave-1-out per-user train/test split.

Holds out each user's single most-recent interaction as test. The raw data
has no timestamp column, so "most recent" is approximated by row order
within BX-Book-Ratings.csv, which is the best ordering signal available in
this dataset — flagged explicitly as a limitation, not hidden.

K_TEST is fixed at 1: 68.1% of users already sit in the 1-2 interaction
bucket, so holding out 2 would need min_interactions=3 and would exclude even
more of the user base than the current cut already does. Holding out a fixed
1 keeps every user with >=2 interactions in play. Revisit if per-bucket eval
turns out to need k=2 for some other reason.

Users with fewer than MIN_INTERACTIONS_TO_SPLIT total interactions cannot be
split at all (holding out their only interaction would leave zero training
rows) and are excluded from train/test, but counted separately as "pure cold"
users (56.2% of rating-active users) rather than silently dropped — that
share is itself a finding worth reporting, not noise.
"""

from dataclasses import dataclass

import pandas as pd

MIN_INTERACTIONS_TO_SPLIT = 2  # need >= k_test + 1 train row
K_TEST = 1  # hold out this many most-recent interactions per user as test

BUCKET_BINS = [0, 2, 5, 20, float("inf")]
BUCKET_LABELS = ["1-2", "3-5", "6-20", "21+"]


@dataclass
class SplitResult:
    train: pd.DataFrame
    test: pd.DataFrame
    pure_cold_users: pd.DataFrame  # excluded, <MIN_INTERACTIONS_TO_SPLIT rows
    n_users_total: int
    n_users_split: int
    n_users_pure_cold: int


def bucketize_counts(counts: pd.Series) -> pd.Series:
    return pd.cut(counts, bins=BUCKET_BINS, labels=BUCKET_LABELS)


def leave_k_out_split(
    ratings: pd.DataFrame,
    k_test: int = K_TEST,
    min_interactions: int = MIN_INTERACTIONS_TO_SPLIT,
) -> SplitResult:
    """Split ratings into train/test per user.

    `ratings` is assumed to be in the original file's row order, used as a
    proxy for chronological order (see module docstring).
    """
    df = ratings.copy()
    df["_row_order"] = range(len(df))

    counts = df.groupby("User-ID").size()
    splittable_users = counts[counts >= min_interactions].index
    cold_users = counts[counts < min_interactions].index

    splittable = df[df["User-ID"].isin(splittable_users)]
    pure_cold = df[df["User-ID"].isin(cold_users)].drop(columns="_row_order")

    # rank rows within each user by row order, descending -> rank 0 = most recent
    splittable = splittable.sort_values(["User-ID", "_row_order"])
    splittable["_rank_from_end"] = (
        splittable.groupby("User-ID")["_row_order"].rank(
            method="first", ascending=False
        )
        - 1
    )

    test = splittable[splittable["_rank_from_end"] < k_test].drop(
        columns=["_row_order", "_rank_from_end"]
    )
    train = splittable[splittable["_rank_from_end"] >= k_test].drop(
        columns=["_row_order", "_rank_from_end"]
    )

    # user's train-history-depth bucket, needed by the eval harness for
    # per-bucket metrics — computed here (from train only, not train+test,
    # since that's the depth a real system would have seen at inference time)
    train_counts = train.groupby("User-ID").size()

    return SplitResult(
        train=train.reset_index(drop=True),
        test=test.reset_index(drop=True),
        pure_cold_users=pure_cold.reset_index(drop=True),
        n_users_total=int(counts.shape[0]),
        n_users_split=int(len(splittable_users)),
        n_users_pure_cold=int(len(cold_users)),
    )
