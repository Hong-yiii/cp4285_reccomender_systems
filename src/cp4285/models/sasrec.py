"""Small CPU-only SASRec-style learner; not a reproduction of sampled-BCE SASRec."""

import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import cast

import torch
from torch import nn
from torch.nn import functional as F

from cp4285.experiment import Example, Query, Recommender


class _Encoder(nn.Module):
    def __init__(self, sizes: dict[str, int], width: int, heads: int, layers: int, horizon: int):
        super().__init__()
        self.items = nn.ModuleDict({d: nn.Embedding(n + 1, width) for d, n in sizes.items()})
        self.positions = nn.Embedding(horizon, width)
        layer = nn.TransformerEncoderLayer(
            width,
            heads,
            dim_feedforward=4 * width,
            dropout=0.0,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(width)

    def item_table(self, domain: str) -> nn.Embedding:
        return cast(nn.Embedding, self.items[domain])

    def encode(self, domain: str, ids: torch.Tensor, noise: torch.Tensor | None = None):
        x = self.item_table(domain)(ids) + self.positions(torch.arange(ids.shape[1]))
        if noise is not None:
            x = x * noise
        mask = torch.ones((ids.shape[1], ids.shape[1]), dtype=torch.bool).triu(1)
        return self.norm(self.encoder(x, mask=mask))


class SASRecRecommender(Recommender):
    """Shared causal encoder/positions, separate tied input/output tables per domain.

    One Adam update per supplied target, in order; full domain softmax CE at the
    last history position. No padding targets, negative sampling, replay or hidden
    epochs. Input dropout uses an owned CPU generator; attention/FFN dropout is zero.
    This small-pilot policy trades throughput for transparent exposure accounting.
    """

    def __init__(
        self,
        vocabularies: dict[str, tuple[str, ...]],
        *,
        width: int = 16,
        heads: int = 2,
        layers: int = 1,
        max_history: int = 10,
        learning_rate: float = 0.003,
        dropout: float = 0.1,
        seed: int = 7,
    ) -> None:
        if any(type(n) is not int or n <= 0 for n in (width, heads, layers, max_history)):
            raise ValueError("Positive integer architecture dimensions required")
        if width % heads or not 0 <= dropout < 1 or not 0 < learning_rate < 1:
            raise ValueError("Invalid heads, dropout or learning rate")
        if not vocabularies or any(
            not d
            or "." in d
            or not items
            or len(items) != len(set(items))
            or any(not isinstance(item, str) or not item for item in items)
            for d, items in vocabularies.items()
        ):
            raise ValueError("Nonempty unique training vocabularies required")
        self.vocabularies = {d: tuple(items) for d, items in vocabularies.items()}
        self.indices = {
            d: {item: i + 1 for i, item in enumerate(items)}
            for d, items in self.vocabularies.items()
        }
        self.max_history, self.width = max_history, width
        self.learning_rate, self.dropout = learning_rate, dropout
        self.config = {
            "width": width,
            "heads": heads,
            "layers": layers,
            "max_history": max_history,
            "learning_rate": learning_rate,
            "dropout": dropout,
            "seed": seed,
        }
        # Scoped initialization restores the caller's CPU RNG; updates use only self.generator.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.network = _Encoder(
                {d: len(v) for d, v in self.vocabularies.items()}, width, heads, layers, max_history
            )
        self.generator = torch.Generator(device="cpu").manual_seed(seed + 1)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=learning_rate)
        self.continuation_started = False
        self.targets_seen = 0
        self.last_loss: float | None = None

    def _ids(self, query: Query) -> torch.Tensor:
        if query.domain not in self.indices or not query.history:
            raise ValueError("Known domain and nonempty history required")
        try:
            ids = [self.indices[query.domain][item] for item in query.history[-self.max_history :]]
        except KeyError as error:
            raise ValueError("Unknown history item; vocabulary must come from training") from error
        return torch.tensor([ids], dtype=torch.long)

    def train(self, examples: tuple[Example, ...]) -> None:
        # Validate the complete call before any gradient updates.
        encoded = []
        for example in examples:
            ids = self._ids(example.query)
            try:
                target = self.indices[example.target.domain][example.target.item] - 1
            except KeyError as error:
                raise ValueError("Unknown training target") from error
            encoded.append((example.target.domain, ids, target))
        losses = []
        for domain, ids, target in encoded:
            self.optimizer.zero_grad(set_to_none=True)
            p = self.dropout
            noise = None
            if p:
                shape = (1, ids.shape[1], self.width)
                noise = (torch.rand(shape, generator=self.generator) >= p).float() / (1 - p)
            hidden = self.network.encode(domain, ids, noise)[:, -1]
            logits = hidden @ self.network.item_table(domain).weight[1:].T
            loss = F.cross_entropy(logits, torch.tensor([target]))
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1.0)
            self.optimizer.step()
            losses.append(float(loss.detach()))
            self.targets_seen += 1
        self.last_loss = sum(losses) / len(losses) if losses else None

    def score(self, query: Query, candidates: tuple[str, ...]) -> tuple[float, ...]:
        ids = self._ids(query)
        try:
            candidate_ids = [self.indices[query.domain][item] for item in candidates]
        except KeyError as error:
            raise ValueError("Unknown candidate") from error
        with torch.no_grad():
            hidden = self.network.encode(query.domain, ids)[:, -1]
            scores = hidden @ self.network.item_table(query.domain).weight[candidate_ids].T
        return tuple(scores.flatten().tolist())

    def begin_continuation(self) -> None:
        if self.continuation_started:
            raise ValueError("Continuation optimizer may only be reset once")
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self.learning_rate)
        self.continuation_started = True

    def describe(self) -> dict[str, object]:
        count = sum(p.numel() for p in self.network.parameters())
        return {
            "family": "SASRec-style shared causal Transformer (not a reproduction)",
            "config": dict(self.config),
            "seed": self.config["seed"],
            "randomness": "owned CPU generator for input dropout; scoped seeded initialization",
            "update_policy": "one Adam step/target; full domain CE; reset Adam once at continuation",
            "parameter_count": count,
            "trainable_parameter_count": count,
            "catalogue_sizes": {d: len(v) for d, v in self.vocabularies.items()},
            "capacity_note": "includes both domain tables, shared positions, encoder and layer norm",
            "torch": torch.__version__,
            "device": "cpu",
        }

    def save(self, path: str | Path) -> None:
        """Create-only full learner state; load only locally trusted checkpoint files."""
        path = Path(path)
        if os.path.lexists(path):
            raise FileExistsError(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format": 1,
            "vocabularies": self.vocabularies,
            "config": self.config,
            "network": self.network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "rng": self.generator.get_state(),
            "targets_seen": self.targets_seen,
            "last_loss": self.last_loss,
            "continuation_started": self.continuation_started,
        }
        with NamedTemporaryFile(dir=path.parent, suffix=".tmp") as stream:
            torch.save(payload, stream)
            stream.flush()
            os.link(stream.name, path)

    @classmethod
    def load(cls, path: str | Path) -> "SASRecRecommender":
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload["format"] != 1:
            raise ValueError("Unknown neural checkpoint format")
        model = cls(payload["vocabularies"], **payload["config"])
        model.network.load_state_dict(payload["network"])
        model.optimizer.load_state_dict(payload["optimizer"])
        model.generator.set_state(payload["rng"])
        model.targets_seen = payload["targets_seen"]
        model.last_loss = payload["last_loss"]
        model.continuation_started = payload["continuation_started"]
        return model
