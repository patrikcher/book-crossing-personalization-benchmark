"""Tier 4: hybrid / content-aware.

Blends Tier 3's matrix-factorization score with a content-similarity boost
computed from book description embeddings, for the minority of (user,
candidate) pairs where both a user's reading history and the candidate
book have a description available.

**Coverage is the whole story for this tier and it's worse than the raw
2.6% book-level description coverage suggests.** A user needs at least one
*described* book in their train history to get a content profile at all --
and since description coverage doesn't depend on a book's popularity (each
book has roughly the same ~2.6% chance regardless of how many people read
it), a user's odds of ever hitting a described book scale with how many
books they've read. Measured on this dataset: 2.4% of the 1-2-interaction
bucket has any content profile at all, vs. 64.4% of the 21+ bucket. So the
content signal ends up concentrated almost entirely in the bucket that
already has the most collaborative signal to work with (deep history,
where MF already does well) and is nearly absent from the bucket where it
would matter most (sparse/cold-start users) -- the opposite of what this
tier is nominally supposed to help with. This is measured, not assumed,
and reported explicitly per bucket in the eval output.

Scoring: MF's raw scores are z-score normalized within each candidate set
(MF dot products and cosine similarities live on different, incomparable
scales), then a content term (content_weight * cosine similarity between
the user's profile embedding -- the mean of their described train items'
embeddings -- and the candidate's embedding) is added only where both
exist. Candidates or users without content data fall through to the
normalized MF score alone, not a penalty.
"""

from typing import Sequence

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from .mf import MFModel

DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DEFAULT_CONTENT_WEIGHT = 1.0


class HybridModel:
    def __init__(
        self,
        content_weight: float = DEFAULT_CONTENT_WEIGHT,
        embedding_model_name: str = DEFAULT_EMBEDDING_MODEL,
        mf_kwargs: dict | None = None,
    ) -> None:
        self.mf = MFModel(**(mf_kwargs or {}))
        self.content_weight = content_weight
        self.embedding_model_name = embedding_model_name
        self.item_embeddings: dict[str, np.ndarray] = {}
        self.user_profile: dict[int, np.ndarray] = {}
        # diagnostics -- how often content signal was actually available,
        # measured during eval calls, not assumed from book-level coverage
        self._n_candidates_scored = 0
        self._n_content_boosted = 0

    def fit(self, train: pd.DataFrame, descriptions: pd.DataFrame) -> None:
        self.mf.fit(train)

        train_items = set(train["ISBN"].unique())
        described = descriptions[descriptions["ISBN"].isin(train_items)]
        described = described.drop_duplicates(subset="ISBN")

        model = SentenceTransformer(self.embedding_model_name)
        embeddings = model.encode(
            described["description"].fillna("").tolist(),
            normalize_embeddings=True,  # unit vectors -> dot product == cosine similarity
            show_progress_bar=False,
        )
        self.item_embeddings = dict(zip(described["ISBN"], embeddings))

        # user profile = mean of L2-normalized embeddings of their described
        # train items, re-normalized (mean of unit vectors isn't unit length)
        train_with_desc = train[train["ISBN"].isin(self.item_embeddings)]
        for user_id, group in train_with_desc.groupby("User-ID")["ISBN"]:
            vecs = np.stack([self.item_embeddings[isbn] for isbn in group])
            profile = vecs.mean(axis=0)
            norm = np.linalg.norm(profile)
            if norm > 0:
                self.user_profile[user_id] = profile / norm

    def score_items(self, user_id: int, item_ids: Sequence[str]) -> Sequence[float]:
        mf_scores = np.asarray(self.mf.score_items(user_id, item_ids), dtype=np.float64)
        mu, sigma = mf_scores.mean(), mf_scores.std()
        scores = (mf_scores - mu) / sigma if sigma > 0 else mf_scores.copy()

        profile = self.user_profile.get(user_id)
        self._n_candidates_scored += len(item_ids)
        if profile is not None:
            for i, iid in enumerate(item_ids):
                emb = self.item_embeddings.get(iid)
                if emb is not None:
                    scores[i] += self.content_weight * float(profile @ emb)
                    self._n_content_boosted += 1
        return scores.tolist()

    def is_personalized(self, user_id: int) -> bool:
        return self.mf.is_personalized(user_id)

    def has_content_profile(self, user_id: int) -> bool:
        return user_id in self.user_profile

    def content_boost_rate(self) -> float:
        """Fraction of scored (user, candidate) pairs that actually got a
        content-similarity term added, vs. falling through to MF alone."""
        if self._n_candidates_scored == 0:
            return 0.0
        return self._n_content_boosted / self._n_candidates_scored
