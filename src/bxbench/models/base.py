"""Model interface every tier (popularity, item-CF, MF, hybrid) implements.

Kept deliberately minimal: eval.py only needs a model to (a) rank a small
candidate set for one user and (b) say whether that ranking came from real
personalized signal or a popularity fallback. That second part is coverage
tracking: CF/MF implementations can silently degrade to popularity for
near-cold users, so this measures how often that actually happens per model,
rather than assuming it away.
"""

from typing import Protocol, Sequence

import pandas as pd


class RankingModel(Protocol):
    def fit(self, train: pd.DataFrame) -> None:
        """Fit on the training interactions (User-ID, ISBN, Book-Rating)."""
        ...

    def score_items(self, user_id: int, item_ids: Sequence[str]) -> Sequence[float]:
        """Return a score per item_id (higher = more recommended) for one user."""
        ...

    def is_personalized(self, user_id: int) -> bool:
        """True if this user's ranking used real per-user signal, False if it
        fell back to a non-personalized default (e.g. global popularity)."""
        ...
