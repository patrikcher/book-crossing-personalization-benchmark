"""Tier 1: popularity baseline.

Recommends the same globally-popular items to everyone, ranked by train-set
interaction count. This is the tier every other tier has to beat — and the
reference point for "coverage": Tiers 2-4 report what fraction of users get
a real personalized ranking vs. silently falling back to this; Tier 1 is
*always* the fallback (is_personalized is always False), by definition.
"""

from typing import Sequence

import pandas as pd


class PopularityModel:
    def __init__(self) -> None:
        self.item_counts: pd.Series | None = None

    def fit(self, train: pd.DataFrame) -> None:
        self.item_counts = train.groupby("ISBN").size()

    def score_items(self, user_id: int, item_ids: Sequence[str]) -> Sequence[float]:
        assert self.item_counts is not None, "call fit() first"
        return [float(self.item_counts.get(iid, 0)) for iid in item_ids]

    def is_personalized(self, user_id: int) -> bool:
        return False
