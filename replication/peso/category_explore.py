"""Experiment 2: does the category model learn moves between categories, or mostly repeat?

Each test target is a repeat when its category is among the user's last 20 events (the
model's whole input) and an exploration otherwise. Within each group the model's own-block
cells (Pretrain = single LoRA's state 0; single LoRA and PESO = state b on block b, blocks
1-3) are scored beside three rules that rank every category for the same users:
popularity and history (as category_report.py ranks them) and a first-order Markov chain,
the count of last-category -> next-category moves among the block's training targets.

For exploration targets a novel rank also drops every category already in the history, so
repeating earns nothing. The model's lead over the history rule is split by group,
lead = p_rep * lead_rep + p_exp * lead_exp, with 95% bootstrap intervals over users per seed.

    python category_explore.py <category dataset dir> <new report.json> <run dir> [<run dir> ...]

Each run dir holds one seed's matrix-beam20/{lora,peso} dumps. Aggregates only.
"""

import json
import math
import random
import statistics
import sys
from collections import Counter
from pathlib import Path

from category_report import HISTORY, test_users

from cp4285.experiment.metrics import from_ranks

BLOCKS = (1, 2, 3)
KEEP = ("hit@1", "hit@5", "mrr@10")
RULES = ("popular", "history", "markov")


def training_moves(users, code):
    """Category counts and last -> next move counts of a block's training targets."""
    counts, moves = Counter(), Counter()
    for data in users.values():
        if len(data["current_block"]) < 3:
            continue
        seq = [code[i] for i in data["previous_block"] + data["current_block"][:-2]]
        for t in range(len(data["previous_block"]), len(seq)):
            counts[seq[t]] += 1
            if t:
                moves[seq[t - 1], seq[t]] += 1
    return counts, moves


def rule_orders(history, categories, counts, moves):
    """Each rule's full category order for one user."""
    popular = sorted(categories, key=lambda c: (-counts.get(c, 0), c))
    seen = Counter(history)
    last = {c: i for i, c in enumerate(history)}
    own = sorted(seen, key=lambda c: (-seen[c], -last[c]))
    after = history[-1]
    markov = sorted(categories, key=lambda c: (-moves.get((after, c), 0), -counts.get(c, 0), c))
    return {"popular": popular, "history": own + [c for c in popular if c not in seen],
            "markov": markov}  # fmt: skip


def rank(order, target, exclude=()):
    kept = [c for c in order if c not in exclude]
    return kept.index(target) + 1 if target in kept else math.inf


def rows_for(run, method, state_of, blocks, users, code, categories, training):
    """Per-user ranks for one method's own-block cells."""
    out = []
    for b in BLOCKS:
        path = Path(run) / "matrix-beam20" / method / f"state{state_of(b)}_block{b}.jsonl"
        counts, moves = training[b]
        for line in path.read_text().splitlines():
            row = json.loads(line)
            item, history_items = users[b][row["user"]]
            history = history_items[-HISTORY:]
            target = code[item]
            orders = rule_orders(history, categories, counts, moves)
            seen = set(history)
            out.append({
                "group": "repeat" if target in seen else "exploration",
                "model": rank(row["ranked"], target),
                "model_novel": rank(row["ranked"], target, seen),
                **{r: rank(orders[r], target) for r in RULES},
                **{f"{r}_novel": rank(orders[r], target, seen) for r in RULES},
            })  # fmt: skip
    return out


def scores(ranks):
    m = from_ranks(ranks, (1, 5, 10))
    return {"users": len(ranks), **{k: m[k] for k in KEEP}}


def bootstrap(diffs, draws=1000, seed=0):
    rng = random.Random(seed)
    means = sorted(statistics.mean(rng.choices(diffs, k=len(diffs))) for _ in range(draws))
    return [means[int(0.025 * draws)], means[int(0.975 * draws) - 1]]


def summarize(rows):
    """Group scores, novel ranks and the split of the Hit@1 lead over the history rule."""
    out = {}
    for group in ("all", "repeat", "exploration"):
        chosen = [r for r in rows if group == "all" or r["group"] == group]
        cell = {"share": len(chosen) / len(rows)}
        for name in ("model", *RULES):
            cell[name] = scores([r[name] for r in chosen])
        if group == "exploration":
            cell["novel"] = {
                n: scores([r[f"{n}_novel"] for r in chosen]) for n in ("model", *RULES)
            }
        diffs = [(r["model"] == 1) - (r["history"] == 1) for r in chosen]
        cell["hit@1_lead_over_history"] = {"mean": statistics.mean(diffs), "ci95": bootstrap(diffs)}
        out[group] = cell
    shares = {g: out[g]["share"] for g in ("repeat", "exploration")}
    leads = {g: out[g]["hit@1_lead_over_history"]["mean"] for g in shares}
    out["lead_split"] = {g: shares[g] * leads[g] for g in shares}
    return out


def transitions(dataset_users, code, top=10):
    """Most common exploration moves among test targets, last category -> next."""
    moves = Counter()
    for b in BLOCKS:
        for item, history_items in dataset_users[b].values():
            history = history_items[-HISTORY:]
            if code[item] not in set(history):
                moves[history[-1], code[item]] += 1
    return [{"from": a, "to": c, "targets": n} for (a, c), n in moves.most_common(top)]


def main(data_dir, output, *runs):
    data_dir, output = Path(data_dir), Path(output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    dataset = data_dir.name
    index = json.loads(next(data_dir.glob(f"{dataset}.index.*.json")).read_text())
    code = {k: "".join(v) for k, v in index.items()}
    categories = sorted(set(code.values()))
    blocks = [
        json.loads((data_dir / f"{dataset}.all.block_{b}.json").read_text()) for b in range(5)
    ]
    users = {b: test_users(blocks, b, code) for b in BLOCKS}
    training = {b: training_moves(blocks[b], code) for b in BLOCKS}
    methods = {"pretrain": ("lora", lambda b: 0), "lora": ("lora", lambda b: b),
               "peso": ("peso", lambda b: b)}  # fmt: skip
    report = {"dataset": dataset, "blocks": list(BLOCKS), "seeds": {}}
    for run in map(Path, runs):
        report["seeds"][run.name] = {
            name: summarize(
                rows_for(run, source, state_of, BLOCKS, users, code, categories, training)
            )
            for name, (source, state_of) in methods.items()
        }
    report["top_exploration_moves"] = transitions(users, code)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as f:
        json.dump(report, f, indent=2)
    return report


if __name__ == "__main__":
    result = main(*sys.argv[1:])
    for seed, methods in result["seeds"].items():
        for name, cells in methods.items():
            e = cells["exploration"]
            print(seed, name, "exploration share", round(e["share"], 3),
                  "hit@1 model", round(e["model"]["hit@1"], 4), "history", round(e["history"]["hit@1"], 4),
                  "markov", round(e["markov"]["hit@1"], 4), "| lead split", {k: round(v, 4) for k, v in cells["lead_split"].items()})  # fmt: skip
