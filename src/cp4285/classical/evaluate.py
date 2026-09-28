"""Full-ranking leave-last-out evaluation: NDCG@K, Hit@K, Recall@K and MRR, per user."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp


def sample_pairs(pairs: np.ndarray, n: int | None, seed: int = 0) -> np.ndarray:
    """Sample n users and keep all of their (user, item) target rows, users in sampled order.

    pairs must be sorted by user. With one target per user this equals sampling n rows.
    """
    users = np.unique(pairs[:, 0])
    if n is None or n >= len(users):
        return pairs
    pick = users[np.random.default_rng(seed).choice(len(users), n, replace=False)]
    rows = pairs[np.isin(pairs[:, 0], pick)]
    by_user = np.argsort(pick)
    position = by_user[np.searchsorted(pick[by_user], rows[:, 0])]
    return rows[np.argsort(position, kind="stable")]


def evaluate(
    model,
    query: sp.csr_matrix,
    seen: sp.csr_matrix,
    pairs: np.ndarray,
    n_candidates: int,
    ks: tuple[int, ...] = (10, 20),
    batch: int = 1000,
) -> dict[str, float]:
    """Rank each user's held-out items among n_candidates items.

    Scores come from model.score(query[user]): the full training row for user-item
    models, a one-hot of the last item for Markov models. Items in seen[user] are
    excluded. n_candidates = number of base items ranks only within the base domain;
    the full width ranks over the whole catalogue, so injected items can crowd out
    the top-K. A user's rows in pairs must be adjacent; each row is one target.
    """
    return metrics(rank_targets(model, query, seen, pairs, n_candidates, batch), pairs[:, 0], ks)


def metrics(
    ranks: np.ndarray, user: np.ndarray, ks: tuple[int, ...] = (10, 20)
) -> dict[str, float]:
    """Means over users. Hit@K and Recall@K are equal when every user has one target."""
    out = {}
    for k in ks:
        scores = per_user(ranks, user, k)
        for name in ("ndcg", "hit", "recall"):
            out[f"{name}@{k}"] = float(scores[name].mean())
    out["mrr"] = float(per_user(ranks, user, max(ks))["rr"].mean())
    return out


def per_user(ranks: np.ndarray, user: np.ndarray, k: int = 10) -> dict[str, np.ndarray]:
    """Per-user scores from the 0-based rank of every target row.

    ndcg    DCG of the targets in the top K over the ideal DCG of min(targets, K) hits
    hit     1 if any target is in the top K
    recall  share of the user's targets in the top K
    rr      1 / (rank of the best-ranked target), not cut at K
    """
    starts = np.flatnonzero(np.r_[True, user[1:] != user[:-1]])
    if len(starts) != len(np.unique(user)):
        raise ValueError("each user's target rows must be adjacent")
    n_targets = np.diff(np.r_[starts, len(user)])
    hit = ranks < k
    dcg = np.add.reduceat(np.where(hit, 1 / np.log2(ranks + 2), 0.0), starts)
    ideal = np.cumsum(1 / np.log2(np.arange(k) + 2))[np.minimum(n_targets, k) - 1]
    hits = np.add.reduceat(hit.astype(np.float64), starts)
    return {
        "ndcg": dcg / ideal,
        "hit": (hits > 0).astype(np.float64),
        "recall": hits / n_targets,
        "rr": 1 / (np.minimum.reduceat(ranks, starts) + 1),
    }


def ndcg_per_user(ranks: np.ndarray, user: np.ndarray, k: int = 10) -> np.ndarray:
    return per_user(ranks, user, k)["ndcg"]


def rank_targets(model, query, seen, pairs, n_candidates, batch: int = 1000) -> np.ndarray:
    """0-based rank of each held-out row (see evaluate). Each user is scored once."""
    ranks = np.empty(len(pairs), dtype=np.int64)
    popular = np.asarray(seen.sum(axis=0), dtype=np.float32).ravel()[:n_candidates]
    tiebreak = (
        1e-6 * popular / max(popular.max(), 1.0)
    )  # sparse models tie a lot; fall back to popularity
    starts = np.flatnonzero(np.r_[True, pairs[1:, 0] != pairs[:-1, 0]])
    bounds = np.r_[starts, len(pairs)]
    for lo in range(0, len(starts), batch):
        hi = min(lo + batch, len(starts))
        u = pairs[starts[lo:hi], 0]
        s = np.array(model.score(query[u])[:, :n_candidates], dtype=np.float32) + tiebreak
        seen_b = seen[u][:, :n_candidates].tocoo()
        s[seen_b.row, seen_b.col] = -np.inf
        r0, r1 = bounds[lo], bounds[hi]
        row_user = np.repeat(np.arange(hi - lo), np.diff(bounds[lo : hi + 1]))
        for c in range(r0, r1, batch):  # at most `batch` rows of s compared at once
            rows = row_user[c - r0 : min(c + batch, r1) - r0]
            target = s[rows, pairs[c : c + len(rows), 1]]
            ranks[c : c + len(rows)] = (s[rows] > target[:, None]).sum(axis=1)
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
