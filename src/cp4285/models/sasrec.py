"""PyTorch port of kang205/SASRec model.py and modules.py (Apache-2.0).

Modified for this project: TensorFlow operations translated to PyTorch, stable
logistic loss, full-catalogue scoring and explicit input validation. Architecture
follows upstream e3738967fddab206d6eeb4fda433e7a7034dd8b1, not torch's generic encoder.
Original authors: Wang-Cheng Kang and Julian McAuley; attention utilities adapted
upstream from Kyubyong Park's Transformer (June 2017). See NOTICE and licenses/.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F

UPSTREAM_COMMIT = "e3738967fddab206d6eeb4fda433e7a7034dd8b1"
IMPLEMENTATION = "kang205-sasrec-pytorch-v1"


class SASRecBlock(nn.Module):
    def __init__(self, hidden, heads, dropout):
        super().__init__()
        self.heads = heads
        self.attention_norm = nn.LayerNorm(hidden, eps=1e-8)
        self.query = nn.Linear(hidden, hidden)
        self.key = nn.Linear(hidden, hidden)
        self.value = nn.Linear(hidden, hidden)
        self.forward_norm = nn.LayerNorm(hidden, eps=1e-8)
        # Upstream's two width-1 convolutions are pointwise hidden -> hidden -> hidden.
        self.forward_in = nn.Linear(hidden, hidden)
        self.forward_out = nn.Linear(hidden, hidden)
        self.dropout = nn.Dropout(dropout)

    def forward(self, sequence):
        batch, length, hidden = sequence.shape
        queries = self.attention_norm(sequence)
        q, k, v = [
            projected.reshape(batch, length, self.heads, hidden // self.heads).transpose(1, 2)
            for projected in (self.query(queries), self.key(sequence), self.value(sequence))
        ]
        logits = q @ k.transpose(-2, -1) / math.sqrt(hidden // self.heads)
        keys_present = sequence.abs().sum(-1).ne(0)
        causal = torch.ones(length, length, dtype=torch.bool, device=sequence.device).tril()
        allowed = keys_present[:, None, None, :] & causal
        # Upstream uses a finite sentinel: fully masked padding rows must not yield NaNs.
        weights = logits.masked_fill(~allowed, -(2**32) + 1).softmax(dim=-1)
        weights = weights * queries.abs().sum(-1).ne(0)[:, None, :, None]
        attended = (self.dropout(weights) @ v).transpose(1, 2).reshape(batch, length, hidden)
        # Both upstream residuals add the NORMALIZED inputs; there is no output projection.
        attended = attended + queries
        normalized = self.forward_norm(attended)
        output = self.dropout(self.forward_in(normalized).relu())
        return normalized + self.dropout(self.forward_out(output))


class SASRec(nn.Module):
    def __init__(
        self, item_count, max_length, hidden=50, heads=1, layers=2, dropout=0.5, l2_emb=0.0
    ):
        super().__init__()
        if min(item_count, max_length, hidden, heads, layers) < 1 or hidden % heads:
            raise ValueError("Sizes must be positive and hidden must be divisible by heads")
        if not math.isfinite(l2_emb) or l2_emb < 0:
            raise ValueError("l2_emb must be finite and nonnegative")
        self.max_length = max_length
        self.hidden = hidden
        self.l2_emb = l2_emb
        self.items = nn.Embedding(item_count + 1, hidden, padding_idx=0)
        self.positions = nn.Embedding(max_length, hidden)
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(SASRecBlock(hidden, heads, dropout) for _ in range(layers))
        self.norm = nn.LayerNorm(hidden, eps=1e-8)
        for module in self.modules():
            if isinstance(module, (nn.Embedding, nn.Linear)):
                nn.init.xavier_uniform_(module.weight)
            if isinstance(module, nn.Linear):
                nn.init.zeros_(module.bias)
        with torch.no_grad():
            self.items.weight[0].zero_()

    def forward(self, history):
        """Represent every position in a fixed-width, left-padded history."""
        if history.ndim != 2 or history.shape[1] != self.max_length:
            raise ValueError("History must be [batch, max_length], left-padded to the fixed width")
        valid = history.ne(0)
        if not valid[:, -1].all() or (valid[:, :-1] & ~valid[:, 1:]).any():
            raise ValueError("Each history must be nonempty and left-padded without internal gaps")
        positions = torch.arange(self.max_length, device=history.device)
        sequence = self.dropout(
            self.items(history) * math.sqrt(self.hidden) + self.positions(positions)
        )
        sequence = sequence * valid.unsqueeze(-1)
        for block in self.blocks:
            sequence = block(sequence) * valid.unsqueeze(-1)
        return self.norm(sequence)

    def scores(self, history):
        return self(history)[:, -1, :] @ self.items.weight[1:].T

    def sampled_loss(self, history, positive, negative):
        """Original per-position sampled BCE, averaged over non-padding targets."""
        if positive.shape != history.shape or negative.shape != history.shape:
            raise ValueError("History, positive and negative tensors must have identical shapes")
        active = positive.ne(0)
        if not active.any() or (active & (history.eq(0) | negative.eq(0))).any():
            raise ValueError("Loss needs real history/negative items at every supervised position")
        vectors = self(history)[active]
        pos = (vectors * self.items(positive[active])).sum(-1)
        neg = (vectors * self.items(negative[active])).sum(-1)
        # Equivalent to upstream's log-sigmoid BCE, without its numerical epsilon clipping.
        loss = (F.softplus(-pos) + F.softplus(neg)).mean()
        return loss + 0.5 * self.l2_emb * (
            self.items.weight.square().sum() + self.positions.weight.square().sum()
        )
