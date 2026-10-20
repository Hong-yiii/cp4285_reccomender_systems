"""Size of PESO's KL proximity penalty and its gradient, measured from saved checkpoints.

Recomputes upstream's `compute_kl_divergence_loss` ("latest" variant) as the trainer
applies it: lambda times the mean over LoRA modules of KL(softmax(A) || softmax(A_prev))
plus the same for B, where A_prev is the previous block's saved snapshot. Each block's
final adapter gives the largest value reached while training that block, since the
adapter starts the block equal to the snapshot.

The softmax runs over every entry of a flattened matrix (thousands to tens of thousands),
so for small changes KL ~ ||delta - mean(delta)||^2 / (2n): the n in the denominator
shrinks the penalty. Reported in float32 and in bfloat16, the dtype the model trains in.

    python penalty_size.py <PESO run ckpt dir> <new report.json>
"""

import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file

WEIGHT = 2.0  # continual_loss_weight of the PESO run (tag "kldiv_latest_cl_2.0")


def kl(current, previous, dtype):
    """Upstream's KL for one matrix, step for step, in the given dtype."""
    c = F.softmax(current.to(dtype).flatten(), dim=0) + 1e-8
    p = F.softmax(previous.to(dtype).flatten(), dim=0) + 1e-8
    c, p = c / c.sum(), p / p.sum()
    return torch.clamp(torch.sum(c * torch.log(c / p)), min=0.0)


def small_change(current, previous):
    """Second-order approximation ||delta - mean(delta)||^2 / (2n) for near-uniform softmax."""
    delta = (current.float() - previous.float()).flatten()
    return float(((delta - delta.mean()) ** 2).sum() / (2 * delta.numel()))


def block_report(state, block):
    modules = sorted(k[: -len(".A")] for k in state if k.endswith(".A"))
    previous = block - 1  # the "latest" snapshot while training this block
    pairs = [
        (state[f"{m}.{x}"], state[f"{m}.{x}_matrices.{previous}"]) for m in modules for x in "AB"
    ]
    report = {"modules": len(modules), "dtype": str(state[modules[0] + ".A"].dtype)}
    for dtype in (torch.float32, torch.bfloat16):
        weights = [cur.float().clone().requires_grad_(dtype == torch.float32) for cur, _ in pairs]
        total = sum(kl(w, prev, dtype) for w, (_, prev) in zip(weights, pairs))
        penalty = WEIGHT * total / len(modules)
        report[f"penalty_{str(dtype).split('.')[1]}"] = float(penalty)
        if dtype == torch.float32:
            grads = torch.autograd.grad(penalty, weights)
            report["gradient_norm_float32"] = float(sum((g**2).sum() for g in grads).sqrt())
    approx = sum(small_change(cur, prev) for cur, prev in pairs)
    report["penalty_small_change_approx"] = WEIGHT * approx / len(modules)
    change = sum(((cur.float() - prev.float()) ** 2).sum() for cur, prev in pairs).sqrt()
    size = sum((prev.float() ** 2).sum() for _, prev in pairs).sqrt()
    report["adapter_change_relative"] = float(change / size)
    report["entries_per_matrix"] = sorted({cur.numel() for cur, _ in pairs})
    return report


def main(ckpt, output, blocks=(1, 2, 3, 4)):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    report = {"source": str(ckpt), "weight": WEIGHT, "blocks": {}}
    for block in blocks:
        state = load_file(Path(ckpt) / f"block_{block}" / "model.safetensors", device=device)
        report["blocks"][f"block{block}"] = block_report(state, block)
        print(json.dumps({f"block{block}": report["blocks"][f"block{block}"]}), flush=True)
    with Path(output).open("x") as f:
        json.dump(report, f, indent=2)
    return report


if __name__ == "__main__":
    main(*sys.argv[1:3])
