"""Evaluate saved PESO block states on any block's test users and dump per-user top-10s.

Our code, launched by modal_app.py from the pinned upstream Fine-tuning directory so
it can import upstream modules. It reuses their model loader, test dataset, collator,
generation settings and metric functions unchanged. Only the pairing of a model state
with a different block's test users, and the per-example dump, are new. A state
evaluated on its own block must reproduce upstream continual_test.py exactly.
"""

import argparse
import json
import os
import random
import time
from pathlib import Path
from tempfile import NamedTemporaryFile

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
import torch.distributed as dist
from collator import TestCollator
from continual_data_v2 import ContinualSeqRecDataset_v2
from continual_test import load_continual_model
from evaluate import get_metrics_results, get_topk_results
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoTokenizer
from utils import parse_dataset_args, parse_global_args, parse_test_args, parse_train_args, set_seed


def publish(path, text):
    """Create-only and atomic: a preempted write never leaves a partial artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("w", dir=path.parent, suffix=".tmp") as f:
        f.write(text)
        f.flush()
        os.link(f.name, path)  # refuses an existing destination


def load_state(args, tokenizer, local_rank, state):
    """Restore the model as it stood after training block `state`."""
    final = "pretrain" not in args.shift_flag
    if not final and state != 0:
        raise ValueError("A pretrain run has only the block-0 state")
    path = Path(args.ckpt_path) / (f"block_{args.num_blocks - 1}" if final else "block_0")
    args.use_final_model = final
    model = load_continual_model(args, tokenizer, {"": local_rank}, local_rank, str(path), state)
    for module in model.modules():
        if hasattr(module, "set_saved_matrices"):
            module.set_saved_matrices(state, use_final_model=final)
    return model.to(torch.device("cuda", local_rank)).eval()


def ranked(decoded, scores, beams):
    """Upstream get_topk_results ordering, keeping the item strings for the dump."""
    items = [text.split("Response:")[-1].strip().replace(" ", "") for text in decoded]
    out = []
    for b in range(len(items) // beams):
        pairs = sorted(
            zip(items[b * beams : (b + 1) * beams], scores[b * beams : (b + 1) * beams].tolist()),
            key=lambda pair: pair[1],
            reverse=True,
        )
        out.append(pairs)
    return out


def evaluate(args, model, tokenizer, block, rank, world):
    data = ContinualSeqRecDataset_v2(
        args, mode="test", current_block=block, num_blocks=args.num_blocks
    )
    users = [
        u
        for u, d in data.remapped_block_user_sequences[block].items()
        if len(d["current_block"]) >= 3
    ]
    if len(users) != len(data):
        raise AssertionError("Test examples no longer align with eligible block users")
    data.set_prompt(0)
    keep = list(range(len(data)))
    if 0 < args.max_users < len(data):
        # Fixed per block, identical for every state and method, so flips stay paired.
        keep = sorted(random.Random(args.subsample_seed + block).sample(keep, args.max_users))
    subset = Subset(data, keep)
    sampler = DistributedSampler(subset, num_replicas=world, rank=rank, drop_last=True)
    order = [keep[i] for i in sampler]  # dataset indices, in the loader's deterministic order
    loader = DataLoader(
        subset,
        batch_size=args.test_batch_size,
        collate_fn=TestCollator(args, tokenizer),
        sampler=sampler,
        num_workers=2,
        pin_memory=True,
    )
    allowed = data.get_prefix_allowed_tokens_fn(tokenizer)
    device = torch.device("cuda", rank)
    rows = []
    with torch.no_grad():
        for step, (inputs, targets) in enumerate(loader):
            inputs = inputs.to(device)
            beams = args.num_beams
            while True:
                try:
                    output = model.generate(
                        input_ids=inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                        max_new_tokens=10,
                        prefix_allowed_tokens_fn=allowed,
                        num_beams=beams,
                        num_return_sequences=beams,
                        output_scores=True,
                        return_dict_in_generate=True,
                        early_stopping=True,
                    )
                    break
                except torch.cuda.OutOfMemoryError:
                    beams -= 1  # upstream's fallback; recorded per example below
            decoded = tokenizer.batch_decode(output["sequences"], skip_special_tokens=True)
            scores = output["sequences_scores"]
            hits = get_topk_results(decoded, scores.clone(), targets, beams)
            indices = order[
                step * args.test_batch_size : step * args.test_batch_size + len(targets)
            ]
            local = []
            for idx, target, pairs, hit in zip(
                indices, targets, ranked(decoded, scores, beams), hits
            ):
                if [int(item == target) for item, _ in pairs] != hit:
                    raise AssertionError("Dump ordering diverged from upstream get_topk_results")
                local.append(
                    {
                        "index": idx,
                        "user": users[idx],
                        "target": target,
                        "ranked": [item for item, _ in pairs],
                        "scores": [score for _, score in pairs],
                        "hits": hit,
                        "beams": beams,
                    }
                )
            gathered = [None] * world
            dist.all_gather_object(gathered, local)
            if rank == 0:
                rows.extend(row for part in gathered for row in part)
            dist.barrier()
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for add in (parse_global_args, parse_dataset_args, parse_test_args, parse_train_args):
        parser = add(parser)
    parser.add_argument("--num_blocks", type=int, default=5)
    parser.add_argument("--states", required=True, help="Comma-separated saved block states")
    parser.add_argument("--test_blocks", required=True, help="Comma-separated test blocks")
    parser.add_argument("--dump_dir", required=True)
    parser.add_argument("--max_users", type=int, default=0, help="Seeded per-block subsample")
    parser.add_argument("--subsample_seed", type=int, default=4285)
    args = parser.parse_args()
    for key in ("lora_modules_to_save", "lora_target_modules"):
        setattr(args, key, getattr(args, key).split(",") if getattr(args, key) else None)

    set_seed(args.seed)
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("LOCAL_RANK") or 0)
    torch.cuda.set_device(rank)
    dist.init_process_group(backend="nccl", world_size=world, rank=rank)
    tokenizer = AutoTokenizer.from_pretrained(args.ckpt_path)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    if args.test_batch_size > 1:
        # Upstream saves a right-padding tokenizer and generates one user at a time;
        # batched generation needs left padding. Check against batch 1 before use.
        tokenizer.padding_side = "left"
    metrics = args.metrics.split(",")
    dump = Path(args.dump_dir)
    blocks = [int(b) for b in args.test_blocks.split(",")]
    for state in map(int, args.states.split(",")):
        # Resume after preemption: a published dump is complete, so it is never redone.
        pending = [b for b in blocks if not (dump / f"state{state}_block{b}.jsonl").exists()]
        model = load_state(args, tokenizer, rank, state) if pending else None
        for block in blocks:
            name = f"state{state}_block{block}"
            start = time.time()
            if block in pending:
                rows = evaluate(args, model, tokenizer, block, rank, world)
                if rank == 0:
                    publish(dump / f"{name}.jsonl", "".join(json.dumps(r) + "\n" for r in rows))
            if rank == 0 and not (dump / f"{name}.metrics.json").exists():
                rows = [
                    json.loads(line) for line in (dump / f"{name}.jsonl").read_text().splitlines()
                ]
                summary = get_metrics_results([row["hits"] for row in rows], metrics)
                record = {
                    "state": state,
                    "block": block,
                    "examples": len(rows),
                    "num_beams": args.num_beams,
                    "test_batch_size": args.test_batch_size,
                    "max_users": args.max_users,
                    "subsample_seed": args.subsample_seed,
                    "seconds": time.time() - start,
                    "metrics": {m: value / len(rows) for m, value in summary.items()},
                }
                publish(dump / f"{name}.metrics.json", json.dumps(record, indent=2))
                print(json.dumps(record), flush=True)
            dist.barrier()
        del model
        torch.cuda.empty_cache()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
