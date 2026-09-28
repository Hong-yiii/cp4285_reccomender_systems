"""Full-ranking leave-one-out evaluation: NDCG@K, Recall@K and MRR."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp


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
    ks: tuple[int, ...] = (10, 20),
    batch: int = 1000,
) -> dict[str, float]:
    """Rank the held-out item of each (user, item) pair among n_candidates items.

    Scores come from model.score(query[user]): the full training row for user-item
    models, a one-hot of the last item for Markov models. Items in seen[user] are
    excluded. n_candidates = number of base items ranks only within the base domain;
    the full width ranks over the whole catalogue, so injected items can crowd out
    the top-K.
    """
    return metrics(rank_targets(model, query, seen, pairs, n_candidates, batch), ks)


def metrics(ranks: np.ndarray, ks: tuple[int, ...] = (10, 20)) -> dict[str, float]:
    out = {}
    for k in ks:
        hit = ranks < k
        out[f"ndcg@{k}"] = float(np.where(hit, 1 / np.log2(ranks + 2), 0).mean())
        out[f"recall@{k}"] = float(hit.mean())
    out["mrr"] = float((1 / (ranks + 1)).mean())
    return out


def ndcg_per_user(ranks: np.ndarray, k: int = 10) -> np.ndarray:
    return np.where(ranks < k, 1 / np.log2(ranks + 2), 0.0)


def rank_targets(model, query, seen, pairs, n_candidates, batch: int = 1000) -> np.ndarray:
    """0-based rank of each held-out item (see evaluate)."""
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
    return ranks


def paired_change(base: np.ndarray, other: np.ndarray, n_boot: int = 1000, seed: int = 0):
    """Relative change in mean per-user score, other vs base, over the same users.

    Returns (point estimate, 95% bootstrap CI low, high, one-sided p that the true
    change is >= 0). Users are resampled jointly, so the pairing is kept.
    """
    rng = np.random.default_rng(seed)
    point = other.mean() / base.mean() - 1
    idx = rng.integers(0, len(base), size=(n_boot, len(base)))
    boot = other[idx].mean(axis=1) / base[idx].mean(axis=1) - 1
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return float(point), float(lo), float(hi), float((boot >= 0).mean())
