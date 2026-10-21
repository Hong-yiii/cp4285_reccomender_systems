"""How hard would each PESO penalty variant pull, relative to the task gradient on the adapters?

Run like jlens.py, from the pinned upstream Fine-tuning directory, with jlens.py,
jlens_items.py, peso_eval.py and penalty_size.py beside it as cp4285_*.py.

**Task gradient.** The next-item cross-entropy over full-vocabulary logits on the target's
four semantic-ID codes, teacher-forced, as upstream training scores response tokens. For
block b it is measured on single LoRA's state after block b-1 (where block b's training
starts), with block b's test prompts standing in for its training data. It is taken in
steps of 32 examples (the training effective batch), for the LoRA A/B matrices and, for
context, the trained item layers.

**Penalty gradients,** at the drift PESO's adapters actually reached by the end of block b:
- **released:** lambda times the mean over modules of KL(softmax(A) || softmax(A_prev)) + B;
- **sum:** the paper's Eq. 12, which sums over matrices: the released value times 112;
- **size:** each matrix's KL times its entry count n, in float32. For small changes this is
  ||delta - mean(delta)||^2 / 2: a proximal L2 term whose size no longer shrinks with n.

    torchrun ... cp4285_penalty_calibrate.py <upstream args for the single-LoRA run> \
        --peso_ckpt <PESO run dir> --out <new report.json>
"""

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
import torch.distributed as dist
import torch.nn.functional as F
from cp4285_jlens import prompts
from cp4285_jlens_items import LEVELS, code_ids, encode_forced
from cp4285_penalty_size import WEIGHT, kl
from cp4285_peso_eval import load_state
from safetensors.torch import load_file
from transformers import AutoTokenizer
from utils import parse_dataset_args, parse_global_args, parse_test_args, parse_train_args, set_seed


def task_gradient(model, tokenizer, items, device, micro=8, step=32):
    """Mean over steps of the gradient norm on LoRA A/B and on the item layers, and the loss."""
    lora = [p for n, p in model.named_parameters() if n.endswith((".A", ".B"))]
    items_layers = [p for n, p in model.named_parameters() if ".modules_to_save.default." in n]
    for p in model.parameters():
        p.requires_grad_(False)
    for p in lora + items_layers:
        p.requires_grad_(True)
    norms, losses = {"lora": [], "item_layers": []}, []
    for start in range(0, len(items) - step + 1, step):
        model.zero_grad(set_to_none=True)
        total = 0.0
        for m in range(start, start + step, micro):
            chunk = items[m : m + micro]
            batch, positions = encode_forced(tokenizer, chunk, device)
            logits = model(**batch, position_ids=positions, use_cache=False).logits
            logits = logits[:, -LEVELS:, :].float()
            targets = torch.tensor([code_ids(tokenizer, t) for _, _, t in chunk], device=device)
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
            (loss * micro / step).backward()  # gradient accumulation, as the trainer scales it
            total += float(loss) * micro / step
        losses.append(total)
        for name, params in (("lora", lora), ("item_layers", items_layers)):
            norms[name].append(float(sum(p.grad.float().pow(2).sum() for p in params).sqrt()))
    mean = lambda v: sum(v) / len(v)
    return {"loss": mean(losses), **{f"grad_norm_{k}": mean(v) for k, v in norms.items()}}


def penalty_gradients(state, block):
    """Gradient norm of each penalty variant at PESO's end-of-block adapters."""
    modules = sorted(k[: -len(".A")] for k in state if k.endswith(".A"))
    pairs = [
        (state[f"{m}.{x}"], state[f"{m}.{x}_matrices.{block - 1}"]) for m in modules for x in "AB"
    ]
    out = {}
    for variant in ("released", "sum", "size"):
        weights = [cur.float().clone().requires_grad_(True) for cur, _ in pairs]
        total = sum(
            kl(w, prev, torch.float32) * (w.numel() if variant == "size" else 1)
            for w, (_, prev) in zip(weights, pairs)
        )
        penalty = WEIGHT * total / len(modules) * (len(modules) if variant == "sum" else 1)
        grads = torch.autograd.grad(penalty, weights)
        out[variant] = {
            "penalty": float(penalty),
            "grad_norm": float(sum((g**2).sum() for g in grads).sqrt()),
        }
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for add in (parse_global_args, parse_dataset_args, parse_test_args, parse_train_args):
        parser = add(parser)
    parser.add_argument("--num_blocks", type=int, default=5)
    parser.add_argument("--blocks", default="1,2,3,4")
    parser.add_argument("--examples", type=int, default=64, help="Prompts per block (steps of 32)")
    parser.add_argument("--peso_ckpt", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    for key in ("lora_modules_to_save", "lora_target_modules"):
        setattr(args, key, getattr(args, key).split(",") if getattr(args, key) else None)
    set_seed(args.seed)
    dist.init_process_group(backend="nccl", world_size=1, rank=0)
    torch.cuda.set_device(0)
    device = torch.device("cuda", 0)
    tokenizer = AutoTokenizer.from_pretrained(args.ckpt_path)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    report = {"weight": WEIGHT, "blocks": {}}
    for block in map(int, args.blocks.split(",")):
        model = load_state(args, tokenizer, 0, block - 1)
        model.train()  # gradients flow; dropout as in training
        items, _ = prompts(args, block, args.examples, 4285)
        task = task_gradient(model, tokenizer, items, device)
        del model
        torch.cuda.empty_cache()
        peso = load_file(Path(args.peso_ckpt) / f"block_{block}" / "model.safetensors")
        pen = penalty_gradients(peso, block)
        ratios = {v: p["grad_norm"] / task["grad_norm_lora"] for v, p in pen.items()}
        report["blocks"][f"block{block}"] = {"task": task, "penalty": pen, "ratio_to_task": ratios}
        print(json.dumps({f"block{block}": report["blocks"][f"block{block}"]}), flush=True)
    with Path(args.out).open("x") as f:
        json.dump(report, f, indent=2)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
