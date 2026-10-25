"""Fixed-catalogue cutoff-pilot evaluation; distinct from the classical study."""

import torch

from ..models.sasrec_learner import SASRecLearner, tensors
from .metrics import DEFAULT_KS, from_ranks


def target_ranks(scores, targets):
    # Stable tie rule: lower item index wins; the same rule applies to every checkpoint.
    target_scores = scores.gather(1, (targets - 1).unsqueeze(1))
    ids = torch.arange(1, scores.shape[1] + 1, device=scores.device)[None, :]
    rank = (
        1
        + (scores > target_scores).sum(1)
        + ((scores == target_scores) & (ids < targets[:, None])).sum(1)
    )
    return rank


def ranking_metrics(scores, targets, k):
    """Batch sums at one cutoff; kept for callers that accumulate by hand."""
    rank = target_ranks(scores, targets)
    hit = rank <= k
    return {
        "ndcg": float((hit.float() / torch.log2(rank.float() + 1)).sum()),
        "hit": float(hit.float().sum()),
        "count": len(targets),
    }


def evaluate(learner: SASRecLearner, examples, batch_size, k, device):
    ranks = []
    for start in range(0, len(examples), batch_size):
        h, t = tensors(examples[start : start + batch_size], device, learner.model.max_length)
        # No seen-item masking: the target is a future review event, possibly a repeated item.
        ranks.append(target_ranks(learner.score(h), t).cpu())
    if not ranks:
        raise ValueError("Evaluation needs at least one example")
    ranks = torch.cat(ranks).numpy()
    return {**from_ranks(ranks, (*DEFAULT_KS, k)), "examples": len(ranks)}
