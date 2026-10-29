"""SASRec, with or without a category channel, on PESO's chronological Instruments blocks.

Keeps PESO's users, block order and targets (upstream continual_data_v2.py): in each
block, users with at least three new events train on all but their last two events,
validate on the second-to-last and test on the last. SASRec needs a history, so a target
with none is skipped and counted. Block 0 trains from scratch with the pilot's regime
(uniform user sampling, every position of each user's final window). Blocks 1-4 continue
on fresh targets only, after one Adam reset. Each block keeps its validation-best epoch,
selected by Hit@K with NDCG@K as tie-break.

Every saved state is scored on every block's test users: the full catalogue, no seen-item
masking, lower item ID wins ties, metrics from experiment.metrics. The catalogue is all
PESO items from the start, as in PESO's semantic-ID index, so items first seen after
block 0 begin untrained. Targets are also split by how often the item was a training
target up to the scored state, and a popularity baseline ranks by each block's training
targets. With --dumps, every cell also writes one row per test user (PESO item IDs, the
exact rank and the top 20 by score) inside the run folder, which stays out of the repo.
"""

import json
import math
import random
import statistics
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from ..models.category_sasrec import CategorySASRec, CategorySASRecLearner
from ..models.sasrec import SASRec
from ..models.sasrec_learner import SASRecLearner, tensors
from .metrics import from_ranks, selection_key
from .neural_evaluation import target_ranks
from .pilot import initial_sequences
from .utils import save_json

BLOCKS = range(5)
KS = (1, 5, 10, 20)
BUCKETS = {"unseen": (0, 0), "1-4": (1, 4), "5+": (5, math.inf)}


def add_arguments(parser):
    parser.add_argument("--data", required=True, help="Folder with PESO's *.all.block_{0..4}.json")
    parser.add_argument(
        "--categories", required=True, help="category_index.py index: item -> [<cat_k>]"
    )
    parser.add_argument("--arm", choices=("sasrec", "category"), required=True)
    parser.add_argument("--output", required=True, help="New run folder")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-length", type=int, default=20, help="PESO's history window")
    parser.add_argument("--epochs", type=int, default=100, help="Block 0 epoch cap")
    parser.add_argument("--block-epochs", type=int, default=20, help="Epoch cap, blocks 1-4")
    parser.add_argument("--patience", type=int, default=5, help="Epochs without improvement")
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--continuation-lr", type=float, default=1e-4)
    parser.add_argument("--k", type=int, default=10, help="Selection cutoff")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument(
        "--dumps", action="store_true", help="Also write per-user ranks and top-20 lists"
    )


def load(data_dir, categories_path):
    """Blocks, PESO item ID -> 1-based ID, and the 1-based category of each 1-based ID."""
    data_dir = Path(data_dir)
    blocks = [json.loads(next(data_dir.glob(f"*.all.block_{b}.json")).read_text()) for b in BLOCKS]
    index = json.loads(Path(categories_path).read_text())
    items = sorted(index, key=int)
    position = {item: k + 1 for k, item in enumerate(items)}
    category = [0] + [int(index[i][0].removeprefix("<cat_").removesuffix(">")) + 1 for i in items]
    return blocks, position, category


def split(users, position):
    """One block's training, validation and test examples, as upstream builds them."""
    out = {"train": [], "valid": [], "test": [], "skipped_empty_history": 0}
    for user, data in users.items():
        if len(data["current_block"]) < 3:
            continue
        seq = [position[i] for i in data["previous_block"] + data["current_block"]]
        for t in range(len(data["previous_block"]), len(seq) - 2):
            if t:
                out["train"].append({"user": user, "history": seq[:t], "target": seq[t]})
            else:
                out["skipped_empty_history"] += 1
        out["valid"].append({"user": user, "history": seq[:-2], "target": seq[-2]})
        out["test"].append({"user": user, "history": seq[:-1], "target": seq[-1]})
    return out


def ranks(learner, rows, batch=512):
    out = []
    for start in range(0, len(rows), batch):
        part = rows[start : start + batch]
        history, targets = tensors(part, learner.device, learner.model.max_length)
        out.append(target_ranks(learner.score(history), targets).cpu())
    return torch.cat(out).numpy()


def write_dump(path, learner, rows, rank, items, k=20, batch=512):
    """Per-user rows: PESO item IDs, the exact target rank and the top k by score."""
    lines = []
    for start in range(0, len(rows), batch):
        part = rows[start : start + batch]
        history, _ = tensors(part, learner.device, learner.model.max_length)
        scores = learner.score(history)
        top = scores.topk(min(k, scores.shape[1]), dim=1).indices.cpu().tolist()
        for offset, (row, best) in enumerate(zip(part, top)):
            lines.append(json.dumps({
                "user": row["user"], "target": items[row["target"] - 1],
                "rank": int(rank[start + offset]), "ranked": [items[c] for c in best],
            }))  # fmt: skip
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as f:
        f.write("\n".join(lines) + "\n")


def popular_ranks(counts, rows, catalogue, batch=512):
    scores = torch.zeros(1, catalogue)
    for item, n in counts.items():
        scores[0, item - 1] = n
    out = []
    for start in range(0, len(rows), batch):
        targets = torch.tensor([r["target"] for r in rows[start : start + batch]])
        out.append(target_ranks(scores.expand(len(targets), -1), targets))
    return torch.cat(out).numpy()


def cell(rank, rows, exposure):
    """Metrics for all targets and for each training-exposure bucket."""
    out = {"all": {"users": len(rank), **from_ranks(rank, KS)}}
    seen = np.array([exposure[r["target"]] for r in rows])
    for name, (low, high) in BUCKETS.items():
        chosen = rank[(seen >= low) & (seen <= high)]
        out[name] = {"users": len(chosen), **(from_ranks(chosen, KS) if len(chosen) else {})}
    return out


def fit(learner, epoch_batches, valid, epochs, patience, k):
    """Train epoch by epoch on epoch_batches(learner); return the validation-best fork."""
    best, best_key, trace, stale = learner.fork(), (-math.inf, -math.inf), [], 0
    for epoch in range(1, epochs + 1):
        losses = [learner.train(batch) for batch in epoch_batches(learner)]
        metrics = from_ranks(ranks(learner, valid), KS)
        trace.append({"epoch": epoch, "loss": float(np.mean(losses)), **metrics})
        if selection_key(metrics, k) > best_key:
            best, best_key, stale = learner.fork(), selection_key(metrics, k), 0
        else:
            stale += 1
            if stale >= patience:
                break
    return best, trace


def run(args):
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite run: {output}")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    blocks, position, category = load(args.data, args.categories)
    data = [split(users, position) for users in blocks]
    catalogue = len(position)
    sasrec = {"hidden": 50, "heads": 1, "layers": 2, "dropout": 0.5, "l2_emb": 0.0}
    if args.arm == "category":
        model = CategorySASRec(category, args.max_length, **sasrec)
        learner = CategorySASRecLearner(
            model, args.lr, args.continuation_lr, random.Random(args.seed)
        )
    else:
        model = SASRec(catalogue, args.max_length, **sasrec)
        learner = SASRecLearner(model, args.lr, args.continuation_lr, random.Random(args.seed))
    initial = initial_sequences(data[0]["train"])
    rounds = max(1, len(initial) // args.batch)
    learner, trace0 = fit(
        learner,
        lambda current: (current.rng.choices(initial, k=args.batch) for _ in range(rounds)),
        data[0]["valid"], args.epochs, args.patience, args.k,
    )  # fmt: skip
    states, traces = [learner.fork()], [trace0]
    learner.begin_continuation()
    for b in range(1, 5):

        def batches(current, rows=data[b]["train"]):  # fresh targets, reshuffled each epoch
            order = rows[:]
            current.rng.shuffle(order)
            return (order[i : i + args.batch] for i in range(0, len(order), args.batch))

        learner, trace = fit(
            learner, batches, data[b]["valid"], args.block_epochs, args.patience, args.k
        )
        states.append(learner.fork())
        traces.append(trace)
    targets = [Counter(r["target"] for r in d["train"]) for d in data]
    items = sorted(position, key=int)  # 1-based ID - 1 -> PESO item ID
    matrix, popular = {}, {}
    for j in BLOCKS:
        test = data[j]["test"]
        popular[f"block{j}"] = cell(popular_ranks(targets[j], test, catalogue), test, targets[j])
        for i, state in enumerate(states):
            exposure = sum(targets[: i + 1], Counter())
            rank = ranks(state, test)
            matrix[f"state{i}_block{j}"] = cell(rank, test, exposure)
            if args.dumps:
                write_dump(output / "dumps" / f"state{i}_block{j}.jsonl", state, test, rank, items)
    result = {
        "scope": "PESO Instruments blocks, ID-based SASRec arm; not comparable with LLM scores",
        "arm": args.arm,
        "config": {k: v for k, v in vars(args).items() if k != "func"},
        "sasrec": sasrec,
        "data": {
            "items": catalogue,
            "categories": max(category),
            "blocks": [
                {k: (len(v) if isinstance(v, list) else v) for k, v in d.items()} for d in data
            ],
        },
        "learner": states[-1].describe(),
        "parameters": sum(p.numel() for p in states[-1].model.parameters()),
        "validation": traces,
        "matrix": matrix,
        "popular": popular,
        "evaluation": "full catalogue; no seen-item masking; lower item ID wins ties",
        "torch_version": str(torch.__version__),
    }
    save_json(output / "metrics.json", result)
    return result


SUMMARY = ("hit@1", "hit@5", "hit@10", "ndcg@10", "mrr@10", "mrr")
AVERAGES = {"blocks 1-3": (1, 2, 3), "blocks 1-4": (1, 2, 3, 4)}


def spread(values):
    return {
        "mean": statistics.mean(values),
        "sd": statistics.stdev(values) if len(values) > 1 else None,
        "seeds": values,
    }


def own(result, block, part="all"):
    return result["matrix"][f"state{block}_block{block}"][part]


def summarize(paths):
    """Own-block averages per arm, category minus SASRec paired by seed, backward transfer."""
    runs = [json.loads(Path(p).read_text()) for p in paths]
    arms = {}
    for r in runs:
        arms.setdefault(r["arm"], {})[r["config"]["seed"]] = r
    out = {"runs": [str(p) for p in paths], "averages": {}, "backward_transfer": {}}
    for label, blocks in AVERAGES.items():
        rows = {}
        for arm, seeds in arms.items():
            for part in ("all", *BUCKETS):
                cells = {}
                for m in SUMMARY:  # a bucket can be empty in some blocks; average the others
                    values = [
                        statistics.mean(own(r, b, part)[m] for b in present)
                        for r in seeds.values()
                        if (present := [b for b in blocks if m in own(r, b, part)])
                    ]
                    if values:
                        cells[m] = spread(values)
                rows.setdefault(arm, {})[part] = cells
        first = next(iter(arms.values()))
        rows["popular"] = {
            "all": {
                m: spread([statistics.mean(r["popular"][f"block{b}"]["all"][m] for b in blocks)
                           for r in first.values()])
                for m in SUMMARY
            }
        }  # fmt: skip
        if {"sasrec", "category"} <= arms.keys():
            common = sorted(arms["sasrec"].keys() & arms["category"].keys())
            rows["category minus sasrec (paired)"] = {
                "all": {
                    m: spread([
                        statistics.mean(own(arms["category"][s], b)[m] for b in blocks)
                        - statistics.mean(own(arms["sasrec"][s], b)[m] for b in blocks)
                        for s in common
                    ])
                    for m in SUMMARY
                }
            }  # fmt: skip
        out["averages"][label] = rows
    for arm, seeds in arms.items():
        out["backward_transfer"][arm] = {
            m: spread([
                statistics.mean(
                    r["matrix"][f"state4_block{j}"]["all"][m] - own(r, j)[m] for j in range(4)
                )
                for r in seeds.values()
            ])
            for m in SUMMARY
        }  # fmt: skip
    return out


def markdown(summary):
    lines = []
    for label, rows in summary["averages"].items():
        lines += [f"{label} (own state):", "", "| Arm | Targets | " + " | ".join(SUMMARY) + " |"]
        lines.append("| --- | --- |" + " --- |" * len(SUMMARY))
        for arm, parts in rows.items():
            for part, cells in parts.items():
                values = [
                    f"{c['mean']:.4f}" + (f" ± {c['sd']:.4f}" if c["sd"] is not None else "")
                    for c in cells.values()
                ]
                lines.append(f"| {arm} | {part} | " + " | ".join(values) + " |")
        lines.append("")
    lines += ["Backward transfer (state 4 minus own state, blocks 0-3):", ""]
    lines.append("| Arm | " + " | ".join(SUMMARY) + " |")
    lines.append("| --- |" + " --- |" * len(SUMMARY))
    for arm, cells in summary["backward_transfer"].items():
        values = [f"{c['mean']:+.4f}" for c in cells.values()]
        lines.append(f"| {arm} | " + " | ".join(values) + " |")
    return "\n".join(lines)
