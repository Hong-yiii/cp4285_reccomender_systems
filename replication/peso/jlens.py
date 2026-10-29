"""Jacobian-lens probe of internal versus output forgetting in PESO block states.

Our implementation of the J-lens (Gurnee et al., 2026, "Verbalizable Representations
Form a Global Workspace in Language Models", transformer-circuits.pub/2026/workspace),
adapted to a generative recommender. Run from the pinned upstream Fine-tuning directory,
like peso_eval.py, so the upstream loader restores each saved block state.

Adaptation, stated rather than hidden:
- The paper averages d h_L(t') / d h_l(t) over position pairs t <= t'. We use the
  decision point only (t = t' = the last prompt token, where the first item code is
  produced), averaged over prompts.
- Items are 4-token semantic IDs; the lens reads the first code token only.
- Rows are r_v = (W_U[v] * gamma) J_l, with gamma the final RMSNorm weight. The norm's
  remaining effect is a positive rescaling, which does not change token rankings.
- The logit lens (J = I) is reported alongside as the baseline.

Outputs per (state, layer): the J-lens rows for first-code tokens, readout ranks of each
test user's target first code (J-lens, logit lens, final output), and, across states, the
drift of r_v split into the item-head part (dW_U J) and the internal-pathway part (W_U dJ).
"""

import argparse
import json
import os
import random
import time
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
import torch.distributed as dist
from continual_data_v2 import ContinualSeqRecDataset_v2
from cp4285_peso_eval import load_state
from transformers import AutoTokenizer
from utils import parse_dataset_args, parse_global_args, parse_test_args, parse_train_args, set_seed


def prompts(args, block, count, seed):
    """Test prompts of one block: (dataset index, prompt text, target item string)."""
    data = ContinualSeqRecDataset_v2(
        args, mode="test", current_block=block, num_blocks=args.num_blocks
    )
    data.set_prompt(0)
    keep = sorted(random.Random(seed + block).sample(range(len(data)), min(count, len(data))))
    return [(i, data[i]["input_ids"], data[i]["labels"]) for i in keep], data


def encode(tokenizer, texts, device):
    tokenizer.padding_side = "left"  # the decision point is the last column for every row
    batch = tokenizer(texts, return_tensors="pt", padding=True).to(device)
    positions = (batch["attention_mask"].cumsum(-1) - 1).clamp(min=0)
    return batch, positions


def forward(model, batch, positions, layers, track=False):
    """Residual stream after each requested layer and after the last layer (pre-norm).

    With track=True the token embeddings become a differentiable input, so the frozen
    model still builds the graph needed for d h_L / d h_l.
    """
    store, hooks = {}, []
    blocks = model.model.layers
    for layer in {*layers, len(blocks) - 1}:

        def hook(module, inputs, output, layer=layer):
            store[layer] = output[0] if isinstance(output, tuple) else output

        hooks.append(blocks[layer].register_forward_hook(hook))
    try:
        inputs = {"input_ids": batch["input_ids"]}
        if track:
            embeds = model.model.embed_tokens(batch["input_ids"]).detach().requires_grad_(True)
            inputs = {"inputs_embeds": embeds}
        model(
            **inputs,
            attention_mask=batch["attention_mask"],
            position_ids=positions,
            use_cache=False,
        )
    finally:
        for h in hooks:
            h.remove()
    return store


def unembedding(model):
    head = model.lm_head
    weight = (
        head.modules_to_save["default"].weight if hasattr(head, "modules_to_save") else head.weight
    )
    return weight.float() * model.model.norm.weight.float()  # fold the final RMSNorm gain in


def jlens_rows(model, tokenizer, texts, token_ids, layers, device, batch_size, cotangents=None):
    """Mean over prompts of c_v^T d h_L(T) / d h_l(T) for each token v and layer l.

    The cotangents c_v default to the model's own unembedding rows u_v (the J-lens);
    passing u_v(state) - u_v(reference) gives the item-head part of the drift instead.
    """
    U = unembedding(model)[token_ids]  # [V, d]
    C = U if cotangents is None else cotangents.to(U)
    last = len(model.model.layers) - 1
    rows = {layer: torch.zeros_like(U) for layer in layers}
    for start in range(0, len(texts), batch_size):
        batch, positions = encode(tokenizer, texts[start : start + batch_size], device)
        with torch.enable_grad():
            store = forward(model, batch, positions, layers, track=True)
            h_final = store[last][:, -1, :].float()  # [B, d] at the decision point
            for v in range(len(token_ids)):
                scalar = (h_final @ C[v]).sum()
                grads = torch.autograd.grad(
                    scalar, [store[layer] for layer in layers], retain_graph=True
                )
                for layer, g in zip(layers, grads):
                    rows[layer][v] += g[:, -1, :].float().sum(0)
        del store
    return {layer: r / len(texts) for layer, r in rows.items()}, U


def rank(scores, v):
    """1-based rank of token v: one plus the number of strictly higher scores."""
    return int((scores > scores[v]).sum()) + 1


@torch.no_grad()
def readouts(model, tokenizer, items, rows, U, token_ids, layers, device, batch_size):
    """Rank of each target's first code among first-code tokens, per layer and lens."""
    index = {t: i for i, t in enumerate(token_ids)}
    out = []
    last = len(model.model.layers) - 1
    for start in range(0, len(items), batch_size):
        chunk = items[start : start + batch_size]
        batch, positions = encode(tokenizer, [text for _, text, _ in chunk], device)
        store = forward(model, batch, positions, layers)
        final = store[last][:, -1, :].float() @ U.T
        for row, (idx, _, target) in enumerate(chunk):
            target_id = tokenizer(target, add_special_tokens=False)["input_ids"][0]
            v = index[target_id]

            record = {"index": idx, "final": rank(final[row], v)}
            for layer in layers:
                h = store[layer][row, -1, :].float()
                record[f"jlens_l{layer}"] = rank(rows[layer] @ h, v)
                record[f"logit_l{layer}"] = rank(U @ h, v)
            out.append(record)
    return out


def first_code_tokens(tokenizer, data):
    """Token ids of every item's first semantic-ID code, sorted."""
    return sorted(
        {
            tokenizer(code[0], add_special_tokens=False)["input_ids"][0]
            for code in data.indices.values()
        }
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for add in (parse_global_args, parse_dataset_args, parse_test_args, parse_train_args):
        parser = add(parser)
    parser.add_argument("--num_blocks", type=int, default=5)
    parser.add_argument("--states", required=True)
    parser.add_argument("--layers", default="4,8,12")
    parser.add_argument("--lens_prompts", type=int, default=200, help="Prompts averaged into J")
    parser.add_argument("--readout_users", type=int, default=500, help="Users per test block")
    parser.add_argument("--readout_blocks", default="0,1,2,3,4")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument(
        "--head_drift_from",
        help="Reference lens.pt: also compute (U_state - U_ref) J_state, the item-head drift",
    )
    args = parser.parse_args()
    for key in ("lora_modules_to_save", "lora_target_modules"):
        setattr(args, key, getattr(args, key).split(",") if getattr(args, key) else None)
    set_seed(args.seed)
    dist.init_process_group(backend="nccl", world_size=1, rank=0)
    torch.cuda.set_device(0)
    device = torch.device("cuda", 0)
    tokenizer = AutoTokenizer.from_pretrained(args.ckpt_path)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    layers = [int(x) for x in args.layers.split(",")]
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    # Lens prompts: a fixed mix of blocks 1-4 test prompts, identical for every state.
    lens_items, data = [], None
    for block in range(1, args.num_blocks):
        chosen, data = prompts(args, block, args.lens_prompts // (args.num_blocks - 1), 4285)
        lens_items += chosen
    token_ids = first_code_tokens(tokenizer, data)
    reference = torch.load(args.head_drift_from) if args.head_drift_from else None
    for state in map(int, args.states.split(",")):
        name = out / f"state{state}"
        if reference is not None:
            if (name / "head_drift.pt").exists():
                continue
            model = load_state(args, tokenizer, 0, state)
            for p in model.parameters():
                p.requires_grad_(False)
            delta = unembedding(model)[token_ids] - reference["U"].to(device)
            drift, _ = jlens_rows(
                model, tokenizer, [t for _, t, _ in lens_items], token_ids, layers, device,
                args.batch, cotangents=delta,
            )  # fmt: skip
            tmp = name / "head_drift.pt.tmp"
            torch.save({"rows": {k: v.cpu() for k, v in drift.items()}}, tmp)
            tmp.rename(name / "head_drift.pt")
            print(json.dumps({"state": state, "head_drift": "saved"}), flush=True)
            del model
            torch.cuda.empty_cache()
            continue
        if (name / "readouts.jsonl").exists():
            continue  # resumable: a finished state is never redone
        start = time.time()
        model = load_state(args, tokenizer, 0, state)
        for p in model.parameters():
            p.requires_grad_(False)
        rows, U = jlens_rows(
            model, tokenizer, [t for _, t, _ in lens_items], token_ids, layers, device, args.batch
        )
        records = []
        for block in map(int, args.readout_blocks.split(",")):
            items, _ = prompts(args, block, args.readout_users, 4285)
            for r in readouts(
                model, tokenizer, items, rows, U, token_ids, layers, device, args.batch
            ):
                records.append({"state": state, "block": block, **r})
        name.mkdir(exist_ok=True)
        torch.save({"rows": {k: v.cpu() for k, v in rows.items()}, "U": U.cpu(), "token_ids": token_ids,
                    "layers": layers}, name / "lens.pt")  # fmt: skip
        tmp = name / "readouts.jsonl.tmp"
        tmp.write_text("".join(json.dumps(r) + "\n" for r in records))
        tmp.rename(name / "readouts.jsonl")
        print(json.dumps({"state": state, "seconds": round(time.time() - start, 1),
                          "readouts": len(records)}), flush=True)  # fmt: skip
        del model
        torch.cuda.empty_cache()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
