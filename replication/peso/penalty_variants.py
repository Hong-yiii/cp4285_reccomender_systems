"""Run upstream PESO training with a different proximal-penalty scale (direction O8).

Upstream's penalty (continual_lora.ContinualLoRALinear.compute_kl_divergence_loss, the
paper's Eq. 12) takes the KL between softmaxes over every entry of a flattened LoRA
matrix. For small changes that is about ||delta - mean(delta)||^2 / (2n), so it shrinks
with the matrix size n and stays near 1e-6 (penalty_size.py, finding 10).

    --penalty_variant size   each matrix's KL times n, computed in float32

Everything else is upstream's, unmodified: this script swaps that one method in memory
and then runs Fine-tuning/continual_train.py as __main__ with the remaining arguments.
Run it from the pinned Fine-tuning directory as cp4285_penalty_train.py, with
penalty_size.py beside it as cp4285_penalty_size.py. The paper's summed form needs no
patch: upstream averages over the 112 modules, so pass --continual_loss_weight 112 * lambda.
"""

import runpy
import sys

import continual_lora
import torch
from cp4285_penalty_size import kl


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


def main():
    argv = sys.argv[1:]
    if "--penalty_variant" not in argv:
        raise SystemExit("--penalty_variant size is required")
    at = argv.index("--penalty_variant")
    variant = argv[at + 1]
    del argv[at : at + 2]
    if variant != "size":
        raise SystemExit(f"Unknown penalty variant: {variant}")
    continual_lora.ContinualLoRALinear.compute_kl_divergence_loss = size_scaled_kl
    sys.argv = ["continual_train.py", *argv]
    runpy.run_path("continual_train.py", run_name="__main__")


if __name__ == "__main__":
    main()
