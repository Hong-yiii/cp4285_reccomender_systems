"""A-only lifecycle with validation selection and fixed full-catalogue evaluation.

User sampling adapts kang205/SASRec sampler.py (Apache-2.0; see NOTICE).
The learner owns updates/state; this module owns sampling, selection and measurement.
"""

import copy
import json
import math
import random
from pathlib import Path

import numpy as np
import torch

from ..models.sasrec import IMPLEMENTATION, UPSTREAM_COMMIT, SASRec
from ..models.sasrec_learner import SASRecLearner, tensors
from .neural_evaluation import evaluate
from .utils import save_json, sha256


def initial_sequences(examples):
    """One final training sequence per user; exclude ALL their initial items from negatives."""
    latest, seen = {}, {}
    for row in examples:  # Prepared examples are chronological; no validation/retention rows here.
        user = row["user"]
        latest[user] = row
        seen.setdefault(user, set()).update([*row["history"], row["target"]])
    return [{**row, "seen": seen[user]} for user, row in latest.items()]


def run(prepared_path, cfg, output, synthetic=False):
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite run: {output}")
    data = json.loads(Path(prepared_path).read_text())
    training = cfg["training"]
    for key in ("epochs", "batch_size", "cycles", "updates_per_cycle", "k"):
        if training[key] < 1:
            raise ValueError(f"{key} must be positive")
    needed = training["batch_size"] * training["updates_per_cycle"] * training["cycles"]
    if len(data["continuation"]) < needed:
        raise ValueError(
            f"Need {needed} fresh continuation examples, have {len(data['continuation'])}; reduce budget explicitly"
        )
    output.mkdir(parents=True)
    seed = training["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(2)
    device = torch.device(training["device"])
    model_args = {
        "item_count": len(data["item_ids"]),
        "max_length": data["metadata"]["config"]["max_length"],
        **cfg["model"],
    }
    learner = SASRecLearner(
        SASRec(**model_args).to(device),
        training["learning_rate"],
        training["continuation_learning_rate"],
        random.Random(seed),
    )
    batch = training["batch_size"]
    k = training["k"]
    initial = initial_sequences(data["train"])
    if not initial:
        raise ValueError("Initial training needs at least one user sequence")
    validation_trace = []
    best = -math.inf
    best_state = None
    for epoch in range(training["epochs"]):
        losses = []
        # Upstream samples users uniformly with replacement. Allow one batch for tiny smoke data.
        for _ in range(max(1, len(initial) // batch)):
            examples = learner.rng.choices(initial, k=batch)
            losses.append(learner.train(examples))
        metrics = evaluate(learner, data["validation"], batch, k, device)
        validation_trace.append({"epoch": epoch + 1, "loss": float(np.mean(losses)), **metrics})
        if metrics[f"ndcg@{k}"] > best:
            best = metrics[f"ndcg@{k}"]
            best_state = copy.deepcopy(learner.model.state_dict())
    if best_state is None:
        raise ValueError("Validation did not select a finite initial checkpoint")
    learner.model.load_state_dict(best_state)
    before = evaluate(learner, data["retention"], batch, k, device)
    provenance = {"implementation": IMPLEMENTATION, "upstream_commit": UPSTREAM_COMMIT}
    torch.save({"model": best_state, "model_args": model_args, **provenance}, output / "initial.pt")
    restored = learner.fork()
    restored.model.load_state_dict(
        torch.load(output / "initial.pt", map_location=device, weights_only=True)["model"]
    )
    probe, _ = tensors(data["retention"][:batch], device, learner.model.max_length)
    reload_max_difference = float((restored.score(probe) - learner.score(probe)).abs().max())
    after = evaluate(restored, data["retention"], batch, k, device)
    if reload_max_difference > 1e-6 or before != after:
        raise AssertionError("Checkpoint reload did not reproduce fixed evaluation")
    frozen = restored.fork()
    restored.begin_continuation()
    curves = [{"cycle": 0, "updates": 0, "frozen": before, "a_only": after}]
    cursor = 0
    for cycle in range(1, training["cycles"] + 1):
        for _ in range(training["updates_per_cycle"]):
            examples = data["continuation"][cursor : cursor + batch]
            restored.train(examples)
            cursor += batch
        frozen_metrics = evaluate(frozen, data["retention"], batch, k, device)
        if frozen_metrics != before:
            raise AssertionError("Frozen control changed")
        curves.append(
            {
                "cycle": cycle,
                "updates": cycle * training["updates_per_cycle"],
                "frozen": frozen_metrics,
                "a_only": evaluate(restored, data["retention"], batch, k, device),
            }
        )
        torch.save(
            {
                **provenance,
                "model": restored.model.state_dict(),
                "optimizer": restored.optimizer.state_dict(),
                "model_args": model_args,
                "cycle": cycle,
                "stream_cursor": cursor,
                "python_rng_state": restored.rng.getstate(),
                "torch_rng_state": restored.torch_rng_state,
                "device_rng_state": restored.device_rng_state,
                "learner": restored.describe(),
            },
            output / f"cycle-{cycle}.pt",
        )
    result = {
        "scope": "synthetic software smoke test"
        if synthetic
        else "single-seed bounded A-only pilot",
        "synthetic": synthetic,
        "prepared_sha256": sha256(prepared_path),
        "config": cfg,
        "data": data["metadata"],
        "torch_version": str(torch.__version__),
        "device": str(device),
        "checkpoint_reload_max_difference": reload_max_difference,
        "validation": validation_trace,
        "retention": curves,
        "evaluation": "full initial-training catalogue; no seen-item exclusion; fixed prefixes",
        "protocol": "neural-cutoff-a-only",
        "learner": restored.describe(),
        **provenance,
        "training_protocol": {
            "initial": "uniform user sampling; all nonpadding next-item positions in final window",
            "initial_users": len(initial),
            "initial_batches_per_epoch": max(1, len(initial) // batch),
            "continuation": "fresh target only; historical positions masked to avoid implicit replay",
            "adam_betas": [0.9, 0.98],
        },
        "unimplemented": [
            "B interfaces and mixed updates",
            "classical comparison under the neural pilot protocol",
            "multi-seed study",
        ],
    }
    save_json(output / "metrics.json", result)
    return result
