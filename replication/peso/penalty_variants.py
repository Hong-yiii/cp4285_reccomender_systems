"""Run upstream PESO training with one change: a penalty variant (O8, O2) or replay.

Upstream's penalty (continual_lora.ContinualLoRALinear.compute_kl_divergence_loss, the
paper's Eq. 12) takes the KL between softmaxes over every entry of a flattened LoRA
matrix. For small changes that is about ||delta - mean(delta)||^2 / (2n), so it shrinks
with the matrix size n and stays near 1e-6 (penalty_size.py, finding 10).

    --penalty_variant size        each matrix's KL times n, computed in float32 (O8)
    --penalty_variant none        upstream's penalty unchanged (for replay alone)
    --item_penalty_weight MU      also add MU * ||W - W_start||^2 / 2 over the trained
                                  item layers (embed_tokens, lm_head copies), where
                                  W_start is their state when the block starts (O2)
    --replay_fraction F           in blocks 1-4, add training examples from earlier
                                  blocks: F times the current block's count, split
                                  evenly over the earlier blocks, sampled with a seed
                                  fixed by --seed and the block (direction R1)

Upstream penalizes only the LoRA matrices; the item layers (about 98% of trainable
parameters) train unpenalized, and their frozen originals hold the base model, not the
previous block. The item term therefore snapshots the trained copies itself, on each
trainer's first loss: upstream builds one trainer per block, after loading the previous
block's state. Like upstream's KL, the term is part of the loss in training and in the
evaluation loss that picks the block's best checkpoint.

Replay draws from each earlier block's own training examples, built by upstream's
_process_train_data for that block; validation and test stay the current block's.

Everything else is upstream's, unmodified: this script swaps the patched methods in memory
and then runs Fine-tuning/continual_train.py as __main__ with the remaining arguments.
Run it from the pinned Fine-tuning directory as cp4285_penalty_train.py, with
penalty_size.py beside it as cp4285_penalty_size.py. The paper's summed form needs no
patch: upstream averages over the 112 modules, so pass --continual_loss_weight 112 * lambda.
"""

import os
import random
import runpy
import sys

import continual_data_v2
import continual_lora
import torch
import transformers
from cp4285_penalty_size import kl

ITEM_LAYERS = ".modules_to_save.default."


def size_scaled_kl(self):
    """Upstream's KL to the latest snapshot, per matrix times its entry count, in float32."""
    if len(self.A_matrices) == 0:
        return torch.tensor(0.0, device=self.A.device, dtype=torch.float32)
    snapshots = range(len(self.A_matrices))
    if "latest" in self.option:
        snapshots = [len(self.A_matrices) - 1]
    total = torch.tensor(0.0, device=self.A.device, dtype=torch.float32)
    for i in snapshots:
        for current, previous in ((self.A, self.A_matrices[i]), (self.B, self.B_matrices[i])):
            total = total + kl(current, previous.detach(), torch.float32) * current.numel()
    return total if "latest" in self.option else total / len(self.A_matrices)


def item_penalty(compute_loss, weight, log_every=100):
    """Wrap one trainer's compute_loss with weight * ||W - W_start||^2 / 2 on the item layers."""
    start, calls = {}, [0]

    def wrapped(model, inputs, return_outputs=False, num_items_in_batch=None):
        result = compute_loss(model, inputs, return_outputs, num_items_in_batch)
        # Training passes the DeepSpeed engine (names start "module."), evaluation the model.
        params = [
            (n.removeprefix("module."), p) for n, p in model.named_parameters() if ITEM_LAYERS in n
        ]
        if not params:
            raise RuntimeError("No trained item layers found for the item penalty")
        if not start:  # the first loss of this block comes before any optimizer step
            start.update((n, p.detach().clone()) for n, p in params)
        # Exact in float32: both operands are bfloat16 values.
        penalty = weight * sum(((p.float() - start[n].float()) ** 2).sum() for n, p in params) / 2
        calls[0] += 1
        if calls[0] % log_every == 1 and os.environ.get("RANK", "0") == "0":
            print(f"[cp4285 item penalty] call {calls[0]}: {float(penalty):.6g}", flush=True)
        if return_outputs:
            loss, outputs = result
            return loss + penalty, outputs
        return result + penalty

    return wrapped


def add_item_penalty(weight):
    """Give every trainer built from now on the item penalty, at the given weight."""
    init = transformers.Trainer.__init__

    def patched(self, *args, **kwargs):
        init(self, *args, **kwargs)
        self.compute_loss = item_penalty(self.compute_loss, weight)  # bound; instance wins

    transformers.Trainer.__init__ = patched


def with_replay(process, fraction, seed):
    """Wrap _process_train_data: append a balanced sample of earlier blocks' examples."""

    def wrapped(self):
        current = process(self)
        block = self.current_block
        if block == 0 or self.mode != "train":
            return current
        per_block = round(fraction * len(current) / block)
        rng = random.Random(seed * 1000 + block)
        replay = []
        for earlier in range(block):
            self.current_block = earlier
            try:
                examples = process(self)
            finally:
                self.current_block = block
            replay += rng.sample(examples, min(per_block, len(examples)))
        if os.environ.get("RANK", "0") == "0":
            print(
                f"[cp4285 replay] block {block}: {len(current)} + {len(replay)} replayed",
                flush=True,
            )
        return current + replay

    return wrapped


def option(argv, name):
    """Remove --name VALUE from argv and return VALUE, or None if absent."""
    if name not in argv:
        return None
    at = argv.index(name)
    value = argv[at + 1]
    del argv[at : at + 2]
    return value


def main():
    argv = sys.argv[1:]
    variant = option(argv, "--penalty_variant")
    item_weight = option(argv, "--item_penalty_weight")
    replay = option(argv, "--replay_fraction")
    if variant not in ("size", "none"):
        raise SystemExit(f"--penalty_variant size or none is required, got {variant}")
    if variant == "size":
        continual_lora.ContinualLoRALinear.compute_kl_divergence_loss = size_scaled_kl
    if item_weight is not None:
        add_item_penalty(float(item_weight))
    if replay is not None:
        seed = int(argv[argv.index("--seed") + 1])
        dataset = continual_data_v2.ContinualSeqRecDataset_v2
        dataset._process_train_data = with_replay(dataset._process_train_data, float(replay), seed)
    sys.argv = ["continual_train.py", *argv]
    runpy.run_path("continual_train.py", run_name="__main__")


if __name__ == "__main__":
    main()
