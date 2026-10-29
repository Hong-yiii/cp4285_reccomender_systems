"""Item-level Jacobian lens: every semantic-ID code of the target item, at every layer.

Extends jlens.py, which reads only the first code at three layers. Run like jlens.py,
from the pinned upstream Fine-tuning directory, with jlens.py beside it as
cp4285_jlens.py.

- **Decision points.** The target's codes are fed teacher-forced after the prompt, so
  level c is read where code c is generated (prompt + codes[:c]). Causal attention
  leaves level 0 identical to jlens.py's decision point, which checks this script.
- **Candidates.** Upstream's constrained decoding allows exactly the level-c codes at
  that position (continual_data_v2.get_prefix_allowed_tokens_fn), so each level ranks
  the target's code among its level's codes.
- **Lens rows.** Per level, (W_U γ)[v] ∂h_L/∂h_ℓ at the level's decision point,
  averaged over jlens.py's fixed lens prompts, for every layer before the last.
- **Drift split.** For states after 0, the rows are computed a second time with state 0's
  unembedding as cotangent. That splits drift exactly:
  - item-head part: rows_i(U_i) − rows_i(U_0);
  - inner-layer part: rows_i(U_0) − rows_0(U_0).

Readouts per test user: each level's target-code rank under the final output, the
J-lens and the logit lens at every layer, plus the output's teacher-forced
log-likelihood of the whole item under constrained decoding.
"""

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
import torch.distributed as dist
from cp4285_jlens import forward, prompts, rank, unembedding
from cp4285_peso_eval import load_state
from transformers import AutoTokenizer
from utils import parse_dataset_args, parse_global_args, parse_test_args, parse_train_args, set_seed

LEVELS = 4  # semantic-ID tokens per item


def level_tokens(tokenizer, data):
    """Sorted token ids of every item's code at each level."""
    out = []
    for c in range(LEVELS):
        ids = {
            tuple(tokenizer(code[c], add_special_tokens=False)["input_ids"])
            for code in data.indices.values()
        }
        if any(len(i) != 1 for i in ids):
            raise ValueError(f"Level-{c} codes are not single tokens")
        out.append(sorted(i[0] for i in ids))
    return out


def code_ids(tokenizer, target):
    ids = tokenizer(target, add_special_tokens=False)["input_ids"]
    if len(ids) < LEVELS:
        raise ValueError(f"Target {target!r} has fewer than {LEVELS} tokens")
    return ids[:LEVELS]


def encode_forced(tokenizer, items, device):
    """Prompt plus the target's first LEVELS-1 codes, left-padded: level c sits at c - LEVELS."""
    rows = [tokenizer(text)["input_ids"] + code_ids(tokenizer, target)[: LEVELS - 1]
            for _, text, target in items]  # fmt: skip
    width = max(map(len, rows))
    pad = tokenizer.pad_token_id
    ids = torch.tensor([[pad] * (width - len(r)) + r for r in rows], device=device)
    mask = torch.tensor([[0] * (width - len(r)) + [1] * len(r) for r in rows], device=device)
    return {"input_ids": ids, "attention_mask": mask}, (mask.cumsum(-1) - 1).clamp(min=0)


def lens_rows(model, tokenizer, items, cotangent_sets, layers, device, batch_size):
    """Mean over prompts of c_v^T d h_L(t_c) / d h_l(t_c), per cotangent set, level, layer."""
    last = len(model.model.layers) - 1
    rows = {
        name: [{layer: torch.zeros_like(cot[c]) for layer in layers} for c in range(LEVELS)]
        for name, cot in cotangent_sets.items()
    }
    for start in range(0, len(items), batch_size):
        batch, positions = encode_forced(tokenizer, items[start : start + batch_size], device)
        with torch.enable_grad():
            store = forward(model, batch, positions, layers, track=True)
            inputs = [store[layer] for layer in layers]
            for c in range(LEVELS):
                col = c - LEVELS
                h_final = store[last][:, col, :].float()
                for name, cot in cotangent_sets.items():
                    for v in range(len(cot[c])):
                        scalar = (h_final @ cot[c][v]).sum()
                        grads = torch.autograd.grad(scalar, inputs, retain_graph=True)
                        for layer, g in zip(layers, grads):
                            rows[name][c][layer][v] += g[:, col, :].float().sum(0)
        del store
    return {
        name: [{layer: r / len(items) for layer, r in by_layer.items()} for by_layer in levels]
        for name, levels in rows.items()
    }


@torch.no_grad()
def readouts(model, tokenizer, items, rows, U, tokens, layers, device, batch_size):
    """Per user: each level's target-code rank per source, and the output's item log-likelihood."""
    last = len(model.model.layers) - 1
    eps = model.model.norm.variance_epsilon
    index = [{t: i for i, t in enumerate(level)} for level in tokens]
    out = []
    for start in range(0, len(items), batch_size):
        chunk = items[start : start + batch_size]
        batch, positions = encode_forced(tokenizer, chunk, device)
        store = forward(model, batch, positions, layers)
        for row, (idx, _, target) in enumerate(chunk):
            codes = code_ids(tokenizer, target)
            record, loglik = {"index": idx}, 0.0
            for c in range(LEVELS):
                col, v = c - LEVELS, index[c][codes[c]]
                h_final = store[last][row, col, :].float()
                final = U[c] @ h_final
                record[f"final_c{c}"] = rank(final, v)
                # Constrained decoding: softmax over the level's codes of lm_head(norm(h_L)).
                scale = torch.rsqrt(h_final.pow(2).mean() + eps)
                loglik += float(torch.log_softmax(final * scale, dim=0)[v])
                for layer in layers:
                    h = store[layer][row, col, :].float()
                    record[f"jlens_l{layer}_c{c}"] = rank(rows[c][layer] @ h, v)
                    record[f"logit_l{layer}_c{c}"] = rank(U[c] @ h, v)
            record["loglik_final"] = loglik
            out.append(record)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for add in (parse_global_args, parse_dataset_args, parse_test_args, parse_train_args):
        parser = add(parser)
    parser.add_argument("--num_blocks", type=int, default=5)
    parser.add_argument("--states", required=True)
    parser.add_argument("--layers", default="all", help="Comma list, or all layers but the last")
    parser.add_argument("--lens_prompts", type=int, default=200, help="Prompts averaged into J")
    parser.add_argument("--readout_users", type=int, default=500, help="Users per test block")
    parser.add_argument("--readout_blocks", default="0,1,2,3,4")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--reference", help="State-0 lens.pt whose U per level splits the drift")
    args = parser.parse_args()
    for key in ("lora_modules_to_save", "lora_target_modules"):
        setattr(args, key, getattr(args, key).split(",") if getattr(args, key) else None)
    set_seed(args.seed)
    dist.init_process_group(backend="nccl", world_size=1, rank=0)
    torch.cuda.set_device(0)
    device = torch.device("cuda", 0)
    tokenizer = AutoTokenizer.from_pretrained(args.ckpt_path)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    # The same fixed lens prompts as jlens.py: blocks 1-4 test prompts, seed 4285.
    lens_items, data = [], None
    for block in range(1, args.num_blocks):
        chosen, data = prompts(args, block, args.lens_prompts // (args.num_blocks - 1), 4285)
        lens_items += chosen
    tokens = level_tokens(tokenizer, data)
    reference = torch.load(args.reference)["U"] if args.reference else None
    for state in map(int, args.states.split(",")):
        name = out / f"state{state}"
        if (name / "readouts.jsonl").exists():
            if state == 0 and reference is None:
                reference = torch.load(name / "lens.pt")["U"]
            continue  # resumable: a finished state is never redone
        start = time.time()
        model = load_state(args, tokenizer, 0, state)
        for p in model.parameters():
            p.requires_grad_(False)
        n_layers = len(model.model.layers)
        layers = (list(range(n_layers - 1)) if args.layers == "all"
                  else [int(x) for x in args.layers.split(",")])  # fmt: skip
        full = unembedding(model)
        U = [full[level] for level in tokens]
        cotangents = {"own": U}
        if state != 0:
            if reference is None:
                raise ValueError("States after 0 need --reference or state 0 first")
            cotangents["ref"] = [u.to(device) for u in reference]
        rows = lens_rows(model, tokenizer, lens_items, cotangents, layers, device, args.batch)
        records = []
        for block in map(int, args.readout_blocks.split(",")):
            items, _ = prompts(args, block, args.readout_users, 4285)
            for r in readouts(model, tokenizer, items, rows["own"], U, tokens, layers, device,
                              args.batch):  # fmt: skip
                records.append({"state": state, "block": block, **r})
        name.mkdir(exist_ok=True)
        save = {
            "rows": [{k: v.cpu() for k, v in level.items()} for level in rows["own"]],
            "U": [u.cpu() for u in U], "tokens": tokens, "layers": layers,
        }  # fmt: skip
        if "ref" in rows:
            save["rows_ref"] = [{k: v.cpu() for k, v in level.items()} for level in rows["ref"]]
        tmp = name / "lens.pt.tmp"
        torch.save(save, tmp)
        tmp.rename(name / "lens.pt")
        tmp = name / "readouts.jsonl.tmp"
        tmp.write_text("".join(json.dumps(r) + "\n" for r in records))
        tmp.rename(name / "readouts.jsonl")
        if state == 0 and reference is None:
            reference = save["U"]
        print(json.dumps({"state": state, "seconds": round(time.time() - start, 1),
                          "readouts": len(records)}), flush=True)  # fmt: skip
        del model
        torch.cuda.empty_cache()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
