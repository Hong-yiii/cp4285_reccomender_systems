"""Fixed-catalogue cutoff-pilot evaluation; distinct from the classical study."""

import torch

from ..models.sasrec_learner import SASRecLearner, tensors


def ranking_metrics(scores, targets, k):
    # Stable tie rule: lower item index wins; the same rule applies to every checkpoint.
    target_scores = scores.gather(1, (targets - 1).unsqueeze(1))
    ids = torch.arange(1, scores.shape[1] + 1, device=scores.device)[None, :]
    rank = (
        1
        + (scores > target_scores).sum(1)
        + ((scores == target_scores) & (ids < targets[:, None])).sum(1)
    )
    hit = rank <= k
    return {
        "ndcg": float((hit.float() / torch.log2(rank.float() + 1)).sum()),
        "recall": float(hit.float().sum()),
        "count": len(targets),
    }


def evaluate(learner: SASRecLearner, examples, batch_size, k, device):
    total = {"ndcg": 0.0, "recall": 0.0, "count": 0}
    for start in range(0, len(examples), batch_size):
        h, t = tensors(examples[start : start + batch_size], device, learner.model.max_length)
        scores = learner.score(h)
        # No seen-item masking: the target is a future review event, possibly a repeated item.
        part = ranking_metrics(scores, t, k)
        for key in total:
            total[key] += part[key]
    if not total["count"]:
        raise ValueError("Evaluation needs at least one example")
    return {
        f"ndcg@{k}": total["ndcg"] / total["count"],
        f"recall@{k}": total["recall"] / total["count"],
        "examples": total["count"],
    }
