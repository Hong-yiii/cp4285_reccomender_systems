"""Shared single-target ranking metrics: Hit@K, NDCG@K and MRR.

Every evaluator reduces its scores to one 1-based rank per example and reports the
same keys. With one held-out target per example Recall@K equals Hit@K, so the
``recall@K`` key is kept only as an alias for older run files.
"""

from __future__ import annotations

import numpy as np

# Hit@10 is the primary metric (team decision of 28 September, issue #8); NDCG@10 breaks ties.
DEFAULT_KS = (5, 10, 20)
PRIMARY_K = 10


def from_ranks(ranks, ks=DEFAULT_KS) -> dict[str, float]:
    """Mean metrics for 1-based ranks of the held-out target."""
    ranks = np.asarray(ranks, dtype=np.float64)
    if not len(ranks):
        raise ValueError("Evaluation needs at least one example")
    out = {}
    for k in sorted(set(ks)):
        hit = ranks <= k
        out[f"hit@{k}"] = float(hit.mean())
        out[f"recall@{k}"] = out[f"hit@{k}"]
        out[f"ndcg@{k}"] = float(np.where(hit, 1 / np.log2(ranks + 1), 0).mean())
        out[f"mrr@{k}"] = float(np.where(hit, 1 / ranks, 0).mean())
    out["mrr"] = float((1 / ranks).mean())
    return out


def selection_key(metrics, k=PRIMARY_K):
    """Validation ordering: Hit@K first, NDCG@K as the tie-break."""
    return metrics[f"hit@{k}"], metrics[f"ndcg@{k}"]
