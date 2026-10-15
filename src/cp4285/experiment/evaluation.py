"""Shared evaluation example. Change benchmark policy here, never inside a model."""

import math

from .data import Evaluation
from .model import Recommender


def evaluate(model: Recommender, task: Evaluation, *, k: int) -> dict[str, float]:
    """Single-target macro averages; lexical original-item-ID ties, no seen masking.

    Candidate order is an interface detail, not a tie-breaking rule. A repeated
    item remains eligible. Recall@k equals Hit@k for this single-target contract.
    """
    if type(k) is not int or k <= 0:
        raise ValueError("k must be a positive integer")
    if (
        not task.examples
        or not task.candidates
        or len(set(task.candidates)) != len(task.candidates)
    ):
        raise ValueError("Evaluation needs examples and nonempty unique candidates")
    ndcg = hit = 0.0
    for example in task.examples:
        if example.target.domain != task.domain or example.target.item not in task.candidates:
            raise ValueError("Invalid evaluation target/domain")
        scores = tuple(model.score(example.query, task.candidates))
        if len(scores) != len(task.candidates) or any(not math.isfinite(s) for s in scores):
            raise ValueError("score() must return one finite value per candidate")
        target_score = scores[task.candidates.index(example.target.item)]
        rank = 1 + sum(
            score > target_score or (score == target_score and item < example.target.item)
            for item, score in zip(task.candidates, scores, strict=True)
        )
        if rank <= k:
            hit += 1
            ndcg += 1 / math.log2(rank + 1)
    count = len(task.examples)
    return {"ndcg": ndcg / count, "hit": hit / count, "recall": hit / count}
