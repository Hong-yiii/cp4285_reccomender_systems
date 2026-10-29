"""SASRec with a CaFe-style category channel, for sparse item sequences.

Our design after Li et al., "Coarse-to-Fine Sparse Sequential Recommendation" (CaFe,
SIGIR 2022); not a port of their code. A second SASRec encoder reads each history's
category sequence. Its output is added to the item encoder's to score items and, alone,
predicts the next category with a softmax over categories. Training adds that category
cross-entropy to the original sampled item loss. Items rank by
log sigmoid(item logit) + log P(item's category), so a category the history makes
likely lifts all its items, including rarely trained ones.
"""

import torch
from torch import nn
from torch.nn import functional as F

from .sasrec import SASRec
from .sasrec_learner import SASRecLearner

IMPLEMENTATION = "category-channel-sasrec-v1"


class CategorySASRec(nn.Module):
    def __init__(self, item_categories, max_length, **sasrec_args):
        """item_categories[i] is item i's 1-based category; index 0 (padding) maps to 0."""
        super().__init__()
        categories = torch.as_tensor(item_categories, dtype=torch.long)
        if categories.ndim != 1 or len(categories) < 2 or categories[0] != 0:
            raise ValueError("item_categories needs index 0 for padding and one entry per item")
        if (categories[1:] < 1).any():
            raise ValueError("Every item needs a positive category ID")
        self.encoder = SASRec(len(categories) - 1, max_length, **sasrec_args)
        self.intent = SASRec(int(categories.max()), max_length, **sasrec_args)
        self.register_buffer("category_of", categories)
        self.max_length = max_length
        self.items = self.encoder.items  # SASRecLearner reads the catalogue size here

    def represent(self, history):
        return self.encoder(history), self.intent(self.category_of[history])

    def category_logits(self, intents):
        return intents @ self.intent.items.weight[1:].T

    def scores(self, history):
        items, intents = self.represent(history)
        logits = (items[:, -1] + intents[:, -1]) @ self.encoder.items.weight[1:].T
        prior = F.log_softmax(self.category_logits(intents[:, -1]), dim=-1)
        return F.logsigmoid(logits) + prior[:, self.category_of[1:] - 1]

    def sampled_loss(self, history, positive, negative):
        """SASRec's sampled item BCE on the summed encoders plus next-category cross-entropy."""
        if positive.shape != history.shape or negative.shape != history.shape:
            raise ValueError("History, positive and negative tensors must have identical shapes")
        active = positive.ne(0)
        if not active.any() or (active & (history.eq(0) | negative.eq(0))).any():
            raise ValueError("Loss needs real history/negative items at every supervised position")
        items, intents = self.represent(history)
        vectors = (items + intents)[active]
        pos = (vectors * self.encoder.items(positive[active])).sum(-1)
        neg = (vectors * self.encoder.items(negative[active])).sum(-1)
        item_loss = (F.softplus(-pos) + F.softplus(neg)).mean()
        category_loss = F.cross_entropy(
            self.category_logits(intents[active]), self.category_of[positive[active]] - 1
        )
        tables = (m for e in (self.encoder, self.intent) for m in (e.items, e.positions))
        l2 = 0.5 * self.encoder.l2_emb * sum(m.weight.square().sum() for m in tables)
        return item_loss + category_loss + l2


class CategorySASRecLearner(SASRecLearner):
    """SASRecLearner's update and state policy, unchanged, around CategorySASRec."""

    def describe(self) -> dict:
        return {
            **super().describe(),
            "model": "SASRec + category channel",
            "implementation": IMPLEMENTATION,
            "categories": int(self.model.category_of.max()),
            "ranking": "log sigmoid(item logit) + log P(category)",
        }
