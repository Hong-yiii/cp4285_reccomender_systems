"""Small CPU/CUDA SASRec-style learner; not a sampled-BCE SASRec reproduction."""

import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import cast

import torch
from torch import nn
from torch.nn import functional as F

from cp4285.experiment import Example, Query, Recommender

_CUDA_HELP = (
    "NVIDIA CUDA GPU required when device='cuda'; no fallback for explicit CUDA. "
    "Check nvidia-smi and GPU visibility (CUDA_VISIBLE_DEVICES), install a compatible NVIDIA "
    "driver, then run `uv sync --locked --extra neural-cuda` with Python 3.12 and restart the kernel. "
    "See docs/RTX2070-HANDOFF.md."
)


def require_cuda() -> torch.device:
    """Resolve the current visible CUDA GPU, never a CPU/MPS fallback."""
    if torch.version.cuda is None or not torch.cuda.is_available():
        raise RuntimeError(_CUDA_HELP)
    return torch.device("cuda", torch.cuda.current_device())


def resolve_device(device: str = "cpu") -> torch.device:
    """Choose only a declared backend; auto uses CPU if CUDA is unavailable."""
    if device not in {"cpu", "cuda", "auto"}:
        raise ValueError("device must be 'cpu', 'cuda' or 'auto'")
    if device == "cuda" or (device == "auto" and torch.cuda.is_available()):
        return require_cuda()
    return torch.device("cpu")


def device_preflight(device: str = "cpu") -> dict[str, object]:
    """Run a real tensor/backward check on the selected backend, without RNG use."""
    selected = resolve_device(device)
    if selected.type == "cuda":
        return cuda_preflight()
    x = torch.ones((2, 2), device=selected, requires_grad=True)
    (x @ x).sum().backward()
    if x.grad is None or not bool(torch.isfinite(x.grad).all()):
        raise RuntimeError("Nonfinite CPU preflight gradient")
    return {
        "device": str(selected),
        "torch": str(torch.__version__),
        "cuda_runtime": torch.version.cuda,
        "torch_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "logical_cpus": os.cpu_count(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "reproducibility": "owned CPU RNG; same-runtime checks; no cross-device bitwise guarantee",
    }


def cuda_preflight() -> dict[str, object]:
    """Exercise CUDA allocation, matrix multiply and backward before any data/artifacts."""
    device = require_cuda()
    try:
        x = torch.ones((2, 2), device=device, requires_grad=True)
        (x @ x).sum().backward()
        if x.grad is None or not bool(torch.isfinite(x.grad).all()):
            raise RuntimeError("Nonfinite CUDA preflight gradient")
        torch.cuda.synchronize(device)
    except (RuntimeError, AssertionError) as error:
        raise RuntimeError(f"CUDA tensor/backward preflight failed. {_CUDA_HELP}") from error
    properties = torch.cuda.get_device_properties(device)
    return {
        "device": str(device),
        "gpu": properties.name,
        "compute_capability": list(torch.cuda.get_device_capability(device)),
        "total_memory_bytes": properties.total_memory,
        "torch": str(torch.__version__),
        "cuda_runtime": torch.version.cuda,
        "compiled_architectures": torch.cuda.get_arch_list(),
        "cudnn": torch.backends.cudnn.version(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "reproducibility": "owned CUDA RNG; same-runtime checks; no cross-device bitwise guarantee",
    }


class _Encoder(nn.Module):
    def __init__(
        self,
        sizes: dict[str, int],
        width: int,
        heads: int,
        layers: int,
        horizon: int,
        device: torch.device,
    ):
        super().__init__()
        self.items = nn.ModuleDict(
            {d: nn.Embedding(n + 1, width, device=device) for d, n in sizes.items()}
        )
        self.positions = nn.Embedding(horizon, width, device=device)
        layer = nn.TransformerEncoderLayer(
            width,
            heads,
            dim_feedforward=4 * width,
            dropout=0.0,
            batch_first=True,
            norm_first=True,
            activation="gelu",
            device=device,
        )
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(width, device=device)

    def item_table(self, domain: str) -> nn.Embedding:
        return cast(nn.Embedding, self.items[domain])

    def encode(self, domain: str, ids: torch.Tensor, noise: torch.Tensor | None = None):
        x = self.item_table(domain)(ids) + self.positions(
            torch.arange(ids.shape[1], device=ids.device)
        )
        if noise is not None:
            x = x * noise
        mask = torch.ones((ids.shape[1], ids.shape[1]), dtype=torch.bool, device=ids.device).triu(1)
        return self.norm(self.encoder(x, mask=mask))


class SASRecRecommender(Recommender):
    """Shared causal encoder/positions, separate tied input/output tables per domain.

    One Adam update per supplied target, in order; full domain softmax CE at the
    last history position. No padding targets, negative sampling, replay or hidden
    epochs. Input dropout uses an owned device generator; attention/FFN dropout is zero.
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
        device: str = "cpu",
    ) -> None:
        self.device = resolve_device(device)
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
        # Seed only the selected backend; fork_rng restores process RNG after construction.
        devices = [self.device.index] if self.device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            if self.device.type == "cuda":
                with torch.cuda.device(self.device):
                    torch.cuda.manual_seed(seed)
            else:
                torch.default_generator.manual_seed(seed)
            self.network = _Encoder(
                {d: len(v) for d, v in self.vocabularies.items()},
                width,
                heads,
                layers,
                max_history,
                self.device,
            )
        self.generator = torch.Generator(device=self.device).manual_seed(seed + 1)
        self.optimizer = torch.optim.Adam(
            self.network.parameters(), lr=learning_rate, capturable=self.device.type == "cuda"
        )
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
        return torch.tensor([ids], dtype=torch.long, device=self.device)

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
                noise = (
                    torch.rand(shape, generator=self.generator, device=self.device) >= p
                ).float() / (1 - p)
            hidden = self.network.encode(domain, ids, noise)[:, -1]
            logits = hidden @ self.network.item_table(domain).weight[1:].T
            loss = F.cross_entropy(logits, torch.tensor([target], device=self.device))
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
            indices = torch.tensor(candidate_ids, dtype=torch.long, device=self.device)
            scores = hidden @ self.network.item_table(query.domain).weight[indices].T
        return tuple(scores.flatten().tolist())

    def begin_continuation(self) -> None:
        if self.continuation_started:
            raise ValueError("Continuation optimizer may only be reset once")
        self.optimizer = torch.optim.Adam(
            self.network.parameters(), lr=self.learning_rate, capturable=self.device.type == "cuda"
        )
        self.continuation_started = True

    def describe(self) -> dict[str, object]:
        count = sum(p.numel() for p in self.network.parameters())
        return {
            "family": "SASRec-style shared causal Transformer (not a reproduction)",
            "config": dict(self.config),
            "seed": self.config["seed"],
            "randomness": "owned device generator for input dropout; scoped backend initialization",
            "update_policy": "one Adam step/target; full domain CE; reset Adam once at continuation",
            "parameter_count": count,
            "trainable_parameter_count": count,
            "catalogue_sizes": {d: len(v) for d, v in self.vocabularies.items()},
            "capacity_note": "includes both domain tables, shared positions, encoder and layer norm",
            "torch": str(torch.__version__),
            "device": str(self.device),
            "runtime": {
                "gpu": torch.cuda.get_device_name(self.device)
                if self.device.type == "cuda"
                else None,
                "compute_capability": list(torch.cuda.get_device_capability(self.device))
                if self.device.type == "cuda"
                else None,
                "cuda_runtime": torch.version.cuda,
                "compiled_architectures": torch.cuda.get_arch_list()
                if self.device.type == "cuda"
                else [],
                "torch_threads": torch.get_num_threads(),
                "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
                "reproducibility": "not enforced; no cross-device bitwise guarantee",
            },
        }

    def save(self, path: str | Path) -> None:
        """Create-only full learner state; load only locally trusted checkpoint files."""
        path = Path(path)
        if os.path.lexists(path):
            raise FileExistsError(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format": 3,
            "device_type": self.device.type,
            "runtime": self.describe(),
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
    def load(cls, path: str | Path, *, device: str = "cpu") -> "SASRecRecommender":
        selected = resolve_device(device)
        payload = torch.load(path, map_location=selected, weights_only=True)
        if payload["format"] != 3:
            raise ValueError("Unknown neural checkpoint format")
        if payload["device_type"] != selected.type:
            raise ValueError(
                "Checkpoint RNG backend differs; full-state loading requires same backend"
            )
        model = cls(payload["vocabularies"], device=selected.type, **payload["config"])
        model.network.load_state_dict(payload["network"])
        model.optimizer.load_state_dict(payload["optimizer"])
        # Generator state API requires a CPU ByteTensor; this is metadata, not CPU learning.
        model.generator.set_state(payload["rng"].cpu())
        model.targets_seen = payload["targets_seen"]
        model.last_loss = payload["last_loss"]
        model.continuation_started = payload["continuation_started"]
        return model
