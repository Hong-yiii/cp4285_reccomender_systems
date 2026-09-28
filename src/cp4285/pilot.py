"""A-only lifecycle check with validation selection and fixed full-catalogue evaluation.

Sequence sampling adapts kang205/SASRec sampler.py (Kang and McAuley, Apache-2.0;
see NOTICE). Modified for prepared data, PyTorch and fresh-event continuation.
"""

import copy
import json
import math
import random
from pathlib import Path

import numpy as np
import torch

from .common.utils import save_json, sha256
from .model import IMPLEMENTATION, UPSTREAM_COMMIT, SASRec


def tensors(examples, device, max_length):
    history = torch.zeros((len(examples), max_length), dtype=torch.long, device=device)
    for i, row in enumerate(examples):
        prefix = row["history"][-max_length:]
        if not prefix or any(item <= 0 for item in prefix):
            raise ValueError("Each example needs a nonempty history of positive item IDs")
        history[i, -len(prefix) :] = torch.tensor(prefix, device=device)
    targets = torch.tensor([x["target"] for x in examples], device=device)
    return history, targets


def initial_sequences(examples):
    """One final training sequence per user; exclude ALL their initial items from negatives."""
    latest, seen = {}, {}
    for row in examples:  # Prepared examples are chronological; no validation/retention rows here.
        user = row["user"]
        latest[user] = row
        seen.setdefault(user, set()).update([*row["history"], row["target"]])
    return [{**row, "seen": seen[user]} for user, row in latest.items()]


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


@torch.no_grad()
def evaluate(model, examples, batch_size, k, device):
    model.eval()
    total = {"ndcg": 0.0, "recall": 0.0, "count": 0}
    for start in range(0, len(examples), batch_size):
        h, t = tensors(examples[start : start + batch_size], device, model.max_length)
        scores = model.scores(h)
        if not torch.isfinite(scores).all():
            raise ValueError("Nonfinite evaluation scores")
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
    # Check gradients without clipping: upstream Adam does not use gradient clipping.
    torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"), error_if_nonfinite=True)
    optimizer.step()
    return float(loss.detach())


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
    rng = random.Random(seed)
    model_args = {
        "item_count": len(data["item_ids"]),
        "max_length": data["metadata"]["config"]["max_length"],
        **cfg["model"],
    }
    model = SASRec(**model_args).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=training["learning_rate"], betas=(0.9, 0.98)
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
            examples = rng.choices(initial, k=batch)
            losses.append(
                update(
                    model,
                    optimizer,
                    examples,
                    len(data["item_ids"]),
                    device,
                    rng,
                    all_positions=True,
                )
            )
        metrics = evaluate(model, data["validation"], batch, k, device)
        validation_trace.append({"epoch": epoch + 1, "loss": float(np.mean(losses)), **metrics})
        if metrics[f"ndcg@{k}"] > best:
            best = metrics[f"ndcg@{k}"]
            best_state = copy.deepcopy(model.state_dict())
    if best_state is None:
        raise ValueError("Validation did not select a finite initial checkpoint")
    model.load_state_dict(best_state)
    before = evaluate(model, data["retention"], batch, k, device)
    provenance = {"implementation": IMPLEMENTATION, "upstream_commit": UPSTREAM_COMMIT}
    torch.save({"model": best_state, "model_args": model_args, **provenance}, output / "initial.pt")
    restored = SASRec(**model_args).to(device)
    restored.load_state_dict(
        torch.load(output / "initial.pt", map_location=device, weights_only=True)["model"]
    )
    restored.eval()
    model.eval()
    probe, _ = tensors(data["retention"][:batch], device, model.max_length)
    with torch.no_grad():
        reload_max_difference = float((restored.scores(probe) - model.scores(probe)).abs().max())
    after = evaluate(restored, data["retention"], batch, k, device)
    if reload_max_difference > 1e-6 or before != after:
        raise AssertionError("Checkpoint reload did not reproduce fixed evaluation")
    frozen = copy.deepcopy(restored)
    # Reset once at phase transition, retain the same optimizer across update cycles.
    optimizer = torch.optim.Adam(
        restored.parameters(), lr=training["continuation_learning_rate"], betas=(0.9, 0.98)
    )
    curves = [{"cycle": 0, "updates": 0, "frozen": before, "a_only": after}]
    cursor = 0
    for cycle in range(1, training["cycles"] + 1):
        for _ in range(training["updates_per_cycle"]):
            examples = data["continuation"][cursor : cursor + batch]
            update(restored, optimizer, examples, len(data["item_ids"]), device, rng)
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
                "model": restored.state_dict(),
                "optimizer": optimizer.state_dict(),
                "model_args": model_args,
                "cycle": cycle,
                "stream_cursor": cursor,
                "python_rng_state": rng.getstate(),
                "torch_rng_state": torch.get_rng_state(),
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
