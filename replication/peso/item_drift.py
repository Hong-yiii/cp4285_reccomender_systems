"""How far the trained item layers move per block, and the item-penalty weight that matches.

Direction O2 adds weight * ||W - W_start||^2 / 2 over the trained item layers
(embed_tokens and lm_head copies; penalty_variants.py). Its gradient at the end of block
b is weight * (W_b - W_{b-1}), so it pulls as hard as the task does when

    weight = task gradient norm on the item layers / ||W_b - W_{b-1}||.

The task gradient comes from penalty_calibrate.py's report (single LoRA, where block b
starts); the drift from a run's saved checkpoints. Reading only the item tensors, this
needs no GPU.

    python item_drift.py <run ckpt dir> <penalty_calibration.json> <new report.json>
"""

import json
import sys
from pathlib import Path

import torch
from safetensors import safe_open

ITEM_LAYERS = ".modules_to_save.default."


def item_layers(ckpt, block):
    """The trained item-layer tensors of one block's checkpoint, in float32."""
    with safe_open(Path(ckpt) / f"block_{block}" / "model.safetensors", "pt") as f:
        return {k: f.get_tensor(k).float() for k in f.keys() if ITEM_LAYERS in k}  # noqa: SIM118


def main(ckpt, calibration, output, blocks=(1, 2, 3, 4)):
    task = json.loads(Path(calibration).read_text())["blocks"]
    report = {"source": str(ckpt), "calibration": str(calibration), "blocks": {}}
    previous = item_layers(ckpt, blocks[0] - 1)
    for block in blocks:
        current = item_layers(ckpt, block)
        if current.keys() != previous.keys() or not current:
            raise ValueError(f"Item layers differ between blocks {block - 1} and {block}")
        drift = {k: float((current[k] - previous[k]).norm()) for k in current}
        size = {k: float(previous[k].norm()) for k in current}
        total = sum(d**2 for d in drift.values()) ** 0.5
        grad = task[f"block{block}"]["task"]["grad_norm_item_layers"]
        report["blocks"][f"block{block}"] = {
            "drift_norm": drift, "drift_relative": {k: drift[k] / size[k] for k in drift},
            "rows_changed": {
                k: int(((current[k] - previous[k]).abs().amax(dim=-1) > 0).sum()) for k in current
            },
            "task_grad_norm_item_layers": grad, "matched_weight": grad / total,
        }  # fmt: skip
        print(json.dumps({f"block{block}": report["blocks"][f"block{block}"]}), flush=True)
        previous = current
    weights = sorted(b["matched_weight"] for b in report["blocks"].values())
    report["matched_weight_range"] = [weights[0], weights[-1]]
    with Path(output).open("x") as f:
        json.dump(report, f, indent=2)
    return report


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main(*sys.argv[1:4])
