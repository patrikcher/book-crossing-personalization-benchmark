"""Tier 3: matrix factorization (implicit-feedback ALS).

Learns latent user/item vectors directly from the interaction matrix via
implicit's AlternatingLeastSquares, rather than hand-computing similarity
the way item-CF (Tier 2) does. Every interaction counts as 1 (binary,
implicit-feedback framing), not the 0-10 rating value.

ALPHA matters a lot here and is easy to leave at a bad default: implicit's
ALS follows Hu/Koren/Volinsky, where confidence for an observed interaction
is 1 + alpha * r_ui against a baseline confidence of 1 for everything
unobserved. The library's own default is alpha=1.0, which gives observed
interactions only 2x the confidence of unobserved ones -- barely any
separation. The original paper reports good results with alpha in the
15-40 range; DEFAULT_ALPHA below follows that rather than the library
default, after an initial run at alpha=1.0 underperformed even the
popularity baseline in the sparsest buckets.

Coverage: unlike item-CF, ALS assigns every user who appears in train a
real latent vector, even from a single interaction (regularized, so noisy
but not structurally zero) -- there's no natural signal-based fallback
condition the way there was for item-CF's co-occurrence requirement.
is_personalized is True for any user seen during fit; False only for a
user who wasn't in train at all (shouldn't happen given the split
guarantees every included user has >=1 train interaction, but handled
defensively rather than assumed away).
"""

from typing import Sequence

import numpy as np
import pandas as pd
import scipy.sparse as sp
from implicit.als import AlternatingLeastSquares

from .popularity import PopularityModel

DEFAULT_FACTORS = 64
DEFAULT_REGULARIZATION = 0.01
DEFAULT_ALPHA = 40.0
DEFAULT_ITERATIONS = 15
DEFAULT_SEED = 42


class MFModel:
    def __init__(
        self,
        factors: int = DEFAULT_FACTORS,
        regularization: float = DEFAULT_REGULARIZATION,
        alpha: float = DEFAULT_ALPHA,
        iterations: int = DEFAULT_ITERATIONS,
        random_state: int = DEFAULT_SEED,
    ) -> None:
        self.popularity = PopularityModel()
        self.item_index: dict[str, int] = {}
        self.user_index: dict[int, int] = {}
        self.user_factors: np.ndarray | None = None
        self.item_factors: np.ndarray | None = None
        self._als_kwargs = dict(
            factors=factors,
            regularization=regularization,
            alpha=alpha,
            iterations=iterations,
            random_state=random_state,
        )

    def fit(self, train: pd.DataFrame) -> None:
        self.popularity.fit(train)

        items = train["ISBN"].unique()
        users = train["User-ID"].unique()
        self.item_index = {it: i for i, it in enumerate(items)}
        self.user_index = {u: i for i, u in enumerate(users)}

        rows = train["User-ID"].map(self.user_index).to_numpy()
        cols = train["ISBN"].map(self.item_index).to_numpy()
        data = np.ones(len(train), dtype=np.float32)
        user_items = sp.csr_matrix((data, (rows, cols)), shape=(len(users), len(items)))
        user_items.sum_duplicates()

        model = AlternatingLeastSquares(**self._als_kwargs)
        model.fit(user_items, show_progress=False)
        self.user_factors = np.asarray(model.user_factors)
        self.item_factors = np.asarray(model.item_factors)

    def score_items(self, user_id: int, item_ids: Sequence[str]) -> Sequence[float]:
        u = self.user_index.get(user_id)
        if u is None:
            return self.popularity.score_items(user_id, item_ids)

        u_vec = self.user_factors[u]
        item_rows = np.array(
            [self.item_index.get(iid, -1) for iid in item_ids], dtype=np.int64
        )
        scores = np.zeros(len(item_ids), dtype=np.float64)
        known_mask = item_rows >= 0
        if known_mask.any():
            scores[known_mask] = self.item_factors[item_rows[known_mask]] @ u_vec
        # unseen items (item_rows == -1) score 0 -- no learned vector exists,
        # same treatment popularity gives an item it never saw
        return scores.tolist()

    def is_personalized(self, user_id: int) -> bool:
        return user_id in self.user_index
