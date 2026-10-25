"""Full-ranking leave-one-out evaluation: Hit@K, NDCG@K and MRR."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from .metrics import DEFAULT_KS, from_ranks


def sample_pairs(pairs: np.ndarray, n: int | None, seed: int = 0) -> np.ndarray:
    if n is None or n >= len(pairs):
        return pairs
    return pairs[np.random.default_rng(seed).choice(len(pairs), n, replace=False)]


def evaluate(
    model,
    query: sp.csr_matrix,
    seen: sp.csr_matrix,
    pairs: np.ndarray,
    n_candidates: int,
    ks: tuple[int, ...] = DEFAULT_KS,
    batch: int = 1000,
) -> dict[str, float]:
    """Rank the held-out item of each (user, item) pair among n_candidates items.

    Scores come from model.score(query[user]): the full training row for user-item
    models, a one-hot of the last item for Markov models. Items in seen[user] are
    excluded. n_candidates = number of base items ranks only within the base domain;
    the full width ranks over the whole catalogue, so injected items can crowd out
    the top-K.
    """
    ranks = np.empty(len(pairs), dtype=np.int64)
    popular = np.asarray(seen.sum(axis=0), dtype=np.float32).ravel()[:n_candidates]
    tiebreak = (
        1e-6 * popular / max(popular.max(), 1.0)
    )  # sparse models tie a lot; fall back to popularity
    for lo in range(0, len(pairs), batch):
        u, t = pairs[lo : lo + batch, 0], pairs[lo : lo + batch, 1]
        s = np.array(model.score(query[u])[:, :n_candidates], dtype=np.float32) + tiebreak
        seen_b = seen[u][:, :n_candidates].tocoo()
        s[seen_b.row, seen_b.col] = -np.inf
        target = s[np.arange(len(u)), t]
        ranks[lo : lo + batch] = (s > target[:, None]).sum(axis=1)
    return from_ranks(ranks + 1, ks)
