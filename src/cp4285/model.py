"""Small SASRec-style baseline, not an exact reproduction of the original implementation."""

import math

import torch
from torch import nn


class SequentialRecommender(nn.Module):
    def __init__(self, item_count, max_length, hidden=32, heads=1, layers=2, dropout=0.2):
        super().__init__()
        if hidden % heads:
            raise ValueError("hidden must be divisible by heads")
        self.items = nn.Embedding(item_count + 1, hidden, padding_idx=0)
        self.positions = nn.Embedding(max_length, hidden)
        block = nn.TransformerEncoderLayer(
            hidden, heads, hidden * 4, dropout, activation="relu", batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(block, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(hidden)
        self.dropout = nn.Dropout(dropout)
        self.hidden = hidden

    def forward(self, history):
        length = history.shape[1]
        positions = torch.arange(length, device=history.device)
        values = self.dropout(
            self.items(history) * math.sqrt(self.hidden) + self.positions(positions)
        )
        mask = torch.ones(length, length, device=history.device, dtype=torch.bool).triu(1)
        encoded = self.encoder(values, mask=mask, src_key_padding_mask=history.eq(0))
        indices = history.ne(0).sum(1) - 1  # right padding; every example has a nonempty prefix
        return self.norm(encoded[torch.arange(len(history), device=history.device), indices])

    def scores(self, history):
        return self(history) @ self.items.weight[1:].T
