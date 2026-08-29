"""Tier 2: item-based collaborative filtering.

Score a candidate item for a user as the sum of cosine similarity between
that candidate and every item in the user's train history, where item
vectors are rows of the binary item x user interaction matrix (implicit
feedback: interacted or not, not the 0-10 rating value).

**Fallback rule (signal-based, not a fixed interaction-count threshold):**
a user is "personalized" if at least one item in their train history has
ever been co-interacted-with by
some *other* user who also touched more than one item — i.e. there exists,
structurally, at least one other item in the catalog that could in
principle share similarity with something in this user's history. This
measures actual signal absence rather than assuming a depth cutoff. It
undercounts "technically some signal but mostly noise" cases — worth
stating as a limitation if this ends up mattering for the eval numbers.
Falls back to Tier 1's popularity score when the rule trips.
"""

from typing import Sequence

import numpy as np
import pandas as pd
import scipy.sparse as sp

from .popularity import PopularityModel


class ItemCFModel:
    def __init__(self) -> None:
        self.popularity = PopularityModel()
        self.item_index: dict[str, int] = {}
        self.item_user: sp.csr_matrix | None = None  # rows=items, cols=users, binary
        self.item_norms: np.ndarray | None = None
        self.item_has_signal: np.ndarray | None = None  # bool per item row
        self.train_items_by_user: dict[int, list[int]] = {}  # user -> item row-idx

    def fit(self, train: pd.DataFrame) -> None:
        self.popularity.fit(train)

        items = train["ISBN"].unique()
        users = train["User-ID"].unique()
        self.item_index = {it: i for i, it in enumerate(items)}
        user_index = {u: i for i, u in enumerate(users)}

        rows = train["ISBN"].map(self.item_index).to_numpy()
        cols = train["User-ID"].map(user_index).to_numpy()
        data = np.ones(len(train), dtype=np.float32)
        item_user = sp.csr_matrix((data, (rows, cols)), shape=(len(items), len(users)))
        item_user.data[:] = 1.0
        item_user.sum_duplicates()  # in case of any duplicate (user, item) train rows
        self.item_user = item_user

        norms = np.sqrt(item_user.multiply(item_user).sum(axis=1)).A1
        self.item_norms = np.where(norms == 0, 1.0, norms)  # avoid div-by-zero

        user_degree = train.groupby("User-ID").size()
        user_deg_by_col = np.array([user_degree[u] for u in users], dtype=np.float32)
        has_multi_item_user = (user_deg_by_col > 1).astype(np.float32)
        # item_has_signal[i]: does item i share >=1 user with degree>1 who could
        # link it to some other item? sparse matrix-vector product, not a full
        # item-item matrix -- keeps this cheap regardless of catalog size.
        signal_counts = item_user.dot(has_multi_item_user)
        self.item_has_signal = signal_counts > 0

        self.train_items_by_user = {
            u: [self.item_index[it] for it in isbns]
            for u, isbns in train.groupby("User-ID")["ISBN"]
        }

    def _has_signal(self, hist: list[int]) -> bool:
        return bool(hist) and any(self.item_has_signal[h] for h in hist)

    def score_items(self, user_id: int, item_ids: Sequence[str]) -> Sequence[float]:
        hist = self.train_items_by_user.get(user_id, [])
        if not self._has_signal(hist):
            return self.popularity.score_items(user_id, item_ids)

        cand_idx = [self.item_index.get(iid) for iid in item_ids]
        known_rows = [c for c in cand_idx if c is not None]
        if not known_rows:
            return self.popularity.score_items(user_id, item_ids)

        H = self.item_user[hist]  # (n_hist, n_users)
        h_norms = self.item_norms[hist]
        C = self.item_user[known_rows]  # (n_known, n_users)
        c_norms = self.item_norms[known_rows]

        raw = C.dot(H.T).toarray()  # (n_known, n_hist) co-occurrence counts
        cos = raw / (c_norms[:, None] * h_norms[None, :])
        known_scores = cos.sum(axis=1)  # sum of similarity over history

        scores = np.zeros(len(item_ids), dtype=np.float64)
        j = 0
        for i, c in enumerate(cand_idx):
            if c is not None:
                scores[i] = known_scores[j]
                j += 1
            # else: candidate never seen in train -> no signal -> score 0,
            # same treatment popularity gives an unseen item
        return scores.tolist()

    def is_personalized(self, user_id: int) -> bool:
        return self._has_signal(self.train_items_by_user.get(user_id, []))
