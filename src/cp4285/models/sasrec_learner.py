"""SASRec training state, separate from nn.Module.train()'s mode switch.

Sampling/tensor helpers adapt kang205/SASRec sampler.py (Apache-2.0; see NOTICE).
Initial user selection stays in experiment/pilot.py. One train call is one Adam step.
"""

import math
import random
from collections.abc import Sequence

import torch

from ..experiment.model import Recommender, candidate_indices
from .sasrec import IMPLEMENTATION, UPSTREAM_COMMIT, SASRec


def tensors(examples, device, max_length):
    history = torch.zeros((len(examples), max_length), dtype=torch.long, device=device)
    for i, row in enumerate(examples):
        prefix = row["history"][-max_length:]
        if not prefix or any(item <= 0 for item in prefix):
            raise ValueError("Each example needs a nonempty history of positive item IDs")
        history[i, -len(prefix) :] = torch.tensor(prefix, device=device)
    targets = torch.tensor([x["target"] for x in examples], device=device)
    return history, targets


def training_tensors(examples, count, device, max_length, rng, *, all_positions=False):
    history, targets = tensors(examples, device, max_length)
    positive = torch.zeros_like(history)
    positive[:, -1] = targets
    if all_positions:
        positive[:, :-1] = history[:, 1:]
        positive.masked_fill_(history.eq(0), 0)
    negative = torch.zeros_like(history)
    for i, row in enumerate(examples):
        excluded = row.get("seen", set(row["history"]) | {row["target"]})
        if len(excluded) >= count:
            raise ValueError("No eligible negative item for an example; adjust cohort/catalogue")
        size = min(len(row["history"]), max_length) if all_positions else 1
        samples = []
        for _ in range(size):
            candidate = rng.randrange(1, count + 1)
            while candidate in excluded:
                candidate = rng.randrange(1, count + 1)
            samples.append(candidate)
        negative[i, -size:] = torch.tensor(samples, device=device)
    return history, positive, negative


def update(model, optimizer, examples, count, device, rng, *, all_positions=False):
    model.train()
    history, positive, negative = training_tensors(
        examples, count, device, model.max_length, rng, all_positions=all_positions
    )
    loss = model.sampled_loss(history, positive, negative)
    if not torch.isfinite(loss):
        raise ValueError("Nonfinite training loss")
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"), error_if_nonfinite=True)
    optimizer.step()
    return float(loss.detach())


class SASRecLearner(Recommender[list[dict], torch.Tensor, torch.Tensor]):
    """Own weights, Adam, negative-sampling RNG and CPU/device dropout RNG.

    Initialize/seed the network before wrapping it. Initial batches must contain
    each sampled user's full initial `seen` set. Continuation supervises only the
    final target. Query tensors use 1-based item IDs and zero padding; candidate
    columns use 0-based positions in the fixed item catalogue.
    """

    def __init__(
        self,
        model: SASRec,
        learning_rate: float,
        continuation_learning_rate: float,
        rng: random.Random,
    ):
        if any(
            not math.isfinite(lr) or lr <= 0 for lr in (learning_rate, continuation_learning_rate)
        ):
            raise ValueError("Learning rates must be positive and finite")
        self.model = model
        self.device = next(model.parameters()).device
        if self.device.type not in {"cpu", "cuda", "mps"}:
            raise ValueError("Learner supports CPU, CUDA or MPS")
        self.learning_rate = learning_rate
        self.continuation_learning_rate = continuation_learning_rate
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.98))
        self.rng = rng
        self.continuing = False
        self.torch_rng_state = torch.get_rng_state()
        self.device_rng_state = (
            None
            if self.device.type == "cpu"
            else getattr(torch, self.device.type).get_rng_state(self.device)
        )

    def train(self, batch: list[dict]) -> float:
        count = self.model.items.num_embeddings - 1
        if not batch:
            raise ValueError("Training needs a nonempty batch")
        for row in batch:
            ids = [*row["history"], row["target"], *row.get("seen", ())]
            if not row["history"] or any(type(i) is not int or not 1 <= i <= count for i in ids):
                raise ValueError("Training needs nonempty histories and in-catalogue item IDs")
            if not self.continuing and "seen" not in row:
                raise ValueError("Initial batches need each user's full initial seen-item set")
            excluded = row.get("seen", set(row["history"]) | {row["target"]})
            if not set(row["history"]).union({row["target"]}) <= set(excluded):
                raise ValueError("Negative exclusions must include the supplied history and target")
            if len(excluded) >= count:
                raise ValueError(
                    "No eligible negative item for an example; adjust cohort/catalogue"
                )
        # ponytail: isolate RNG for sequential forks, not concurrent threads sharing torch globals.
        devices = [] if self.device.type == "cpu" else [self.device.index or 0]
        backend = "cuda" if self.device.type == "cpu" else self.device.type
        with torch.random.fork_rng(devices=devices, device_type=backend):
            torch.set_rng_state(self.torch_rng_state)
            if self.device_rng_state is not None:
                getattr(torch, backend).set_rng_state(self.device_rng_state, self.device)
            loss = update(
                self.model,
                self.optimizer,
                batch,
                count,
                self.device,
                self.rng,
                all_positions=not self.continuing,
            )
            self.torch_rng_state = torch.get_rng_state()
            if self.device_rng_state is not None:
                self.device_rng_state = getattr(torch, backend).get_rng_state(self.device)
        return loss

    def begin_continuation(self) -> None:
        if self.continuing:
            raise ValueError("Continuation has already begun; refusing a second optimizer reset")
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.continuation_learning_rate, betas=(0.9, 0.98)
        )
        self.continuing = True

    @torch.no_grad()
    def score(self, queries: torch.Tensor, candidates: Sequence[int] | None = None) -> torch.Tensor:
        columns = candidate_indices(candidates, self.model.items.num_embeddings - 1)
        modes = [(module, module.training) for module in self.model.modules()]
        try:
            self.model.eval()
            scores = self.model.scores(queries)
            if not torch.isfinite(scores).all():
                raise ValueError("Nonfinite evaluation scores")
            return scores if columns is None else scores[:, columns]
        finally:
            for module, training in modes:
                module.training = training

    def describe(self) -> dict:
        return {
            "model": "SASRec",
            "implementation": IMPLEMENTATION,
            "upstream_commit": UPSTREAM_COMMIT,
            "parameters": sum(p.numel() for p in self.model.parameters()),
            "initial_unit": "one Adam step over sampled users; all nonpadding next-item positions",
            "continuation_unit": "one Adam step over fresh final targets only",
            "optimizer_policy": "reset Adam once at transition, then retain moments",
            "rng_policy": "owned Python and CPU/device torch RNG; sequential in-memory forks",
            "phase": "continuation" if self.continuing else "initial",
            "candidate_indices": "zero-based columns; query item IDs are one-based with zero padding",
            "learning_rate": self.learning_rate,
            "continuation_learning_rate": self.continuation_learning_rate,
        }
