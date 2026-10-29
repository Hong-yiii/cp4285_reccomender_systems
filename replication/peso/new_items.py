"""Experiment 3: can the models recommend products they never trained on?

Each product-level test target is grouped by what the scored state saw of it in training,
through block i for state i, on upstream's split (a block's users with at least three
events train on all but their last two):

- cold: in no training example, as target or history;
- input-only: in some training history, never a training target;
- rare: a training target one to four times;
- common: a training target five or more times.

Every dump cell is scored per group: Hit@20 (PESO returns its top 20) with Wilson 95%
intervals and raw hit counts, plus Hit@10 and NDCG@10 through cp4285.experiment.metrics.
Popularity ranks block j's training targets, lower item ID first on ties. Own-state cells
are pooled over blocks 1-3 (users concatenated). With a full matrix, retention follows
blocks 1-3's own-state hits, grouped as the block's own state saw them, to the last state.

For cold targets the mechanism check takes PESO's four semantic-ID codes and measures the
longest prefix the target shares with any item in the user's 20-event history, with the
model's top guess and with any of its top 20. A logistic regression of a cold Hit@20 on the
history prefix follows.
SASRec never sees the codes, so its rows are a code-blind control. Aggregates only.

    python new_items.py <Instruments_cl_ori dir> <new report.json> \
        [--llm <dump dir with lora/ and peso/> ...] [--sasrec <cp4285 blocks run dir> ...]

LLM dumps come from peso_eval.py; SASRec dumps from `cp4285 blocks --dumps`. Needs a
checkout with cp4285 on the path.
"""

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path

import numpy as np

from cp4285.experiment.metrics import from_ranks

BLOCKS = range(5)
OWN = (1, 2, 3)
HISTORY = 20
K = 20
GROUPS = ("cold", "input-only", "rare", "common")
CODE = re.compile(r"<[a-z]_\d+>")


def exposure(blocks):
    """Per state i: training-target counts and training-history items through block i."""
    targets, inputs, out = Counter(), set(), []
    for users in blocks:
        for data in users.values():
            if len(data["current_block"]) < 3:
                continue
            seq = data["previous_block"] + data["current_block"]
            targets.update(data["current_block"][:-2])
            inputs.update(seq[: len(seq) - 3])
        out.append((targets.copy(), set(inputs)))
    return out


def group(item, seen):
    targets, inputs = seen
    n = targets[item]
    return "common" if n >= 5 else "rare" if n else "input-only" if item in inputs else "cold"


def test_users(users):
    """user -> (target item, history items), as upstream's test split builds them."""
    out = {}
    for user, data in users.items():
        if len(data["current_block"]) >= 3:
            seq = data["previous_block"] + data["current_block"]
            out[user] = (seq[-1], seq[:-1][-HISTORY:])
    return out


def read_dump(path, users, code):
    """user -> (rank or inf, top-20 codes) for a peso_eval.py or `cp4285 blocks` dump."""
    out = {}
    for line in Path(path).read_text().splitlines():
        row = json.loads(line)
        item = users[row["user"]][0]
        if "rank" in row:  # SASRec: PESO item IDs and the exact full-catalogue rank
            if str(row["target"]) != item:
                raise ValueError(f"{path}: target mismatch for user {row['user']}")
            out[row["user"]] = (row["rank"], [code[str(i)] for i in row["ranked"][:K]])
        else:  # PESO: semantic-ID strings, best first
            if row["target"] != code[item]:
                raise ValueError(f"{path}: target mismatch for user {row['user']}")
            top = row["ranked"]
            rank = top.index(row["target"]) + 1 if row["target"] in top else math.inf
            out[row["user"]] = (rank, top[:K])
    return out


def wilson(hits, n, z=1.959964):
    if not n:
        return None
    p, z2 = hits / n, z * z
    centre = (p + z2 / (2 * n)) / (1 + z2 / n)
    half = z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / (1 + z2 / n)
    return [centre - half, centre + half]


def scores(ranks):
    if not ranks:
        return {"users": 0}
    m = from_ranks(ranks, (10, K))
    hits = sum(r <= K for r in ranks)
    return {"users": len(ranks), "hits@20": hits, "hit@20": m["hit@20"],
            "hit@20_ci95": wilson(hits, len(ranks)), "hit@10": m["hit@10"],
            "ndcg@10": m["ndcg@10"]}  # fmt: skip


def by_group(ranks, groups):
    return {g: scores([r for u, r in ranks.items() if groups[u] == g]) for g in GROUPS}


def prefix(a, b):
    """Shared leading codes of two semantic IDs."""
    n = 0
    for x, y in zip(CODE.findall(a), CODE.findall(b or "")):
        if x != y:
            break
        n += 1
    return n


def logistic(x, y, steps=50):
    """Intercept and slope of a one-feature logistic regression, with Wald 95% intervals."""
    X = np.column_stack([np.ones(len(x)), np.asarray(x, float)])
    y, beta = np.asarray(y, float), np.zeros(2)
    for _ in range(steps):
        p = 1 / (1 + np.exp(-X @ beta))
        hessian = X.T @ (X * (p * (1 - p))[:, None])
        step = np.linalg.solve(hessian, X.T @ (y - p))
        beta += step
        if np.abs(step).max() < 1e-10:
            break
    se = np.sqrt(np.diag(np.linalg.inv(hessian)))
    return {name: {"estimate": float(b), "ci95": [float(b - 1.96 * s), float(b + 1.96 * s)]}
            for name, b, s in zip(("intercept", "slope"), beta, se)}  # fmt: skip


def mechanism(rows):
    """Cold targets: Hit@20 by shared prefix with the history, the top guess and the top 20.

    TIGER (Rajput et al., arXiv 2305.05065, section 4.3) reaches unseen items by adding those
    that share the first three codes of a retrieved ID; list_prefix >= 3 marks the cold
    targets that step could reach from these top-20 lists.
    """
    out = {}
    for key in ("history_prefix", "guess_prefix", "list_prefix"):
        out[f"by_{key}"] = {
            str(n): scores([r["rank"] for r in rows if r[key] == n]) for n in range(5)
        }
    x, y = [r["history_prefix"] for r in rows], [r["rank"] <= K for r in rows]
    out["logit_hit@20_on_history_prefix"] = logistic(x, y) if 0 < sum(y) < len(y) else None
    return out


def model(cells, users, code, seen, states_of):
    """Per-cell group scores, pooled own-state blocks, retention and the cold mechanism."""
    out = {"cells": {}}
    pooled, cold = {g: [] for g in GROUPS}, []
    first_code = {g: [] for g in GROUPS}  # does the top guess share the target's first code?
    for (i, j), ranks in sorted(cells.items()):
        groups = {u: group(users[j][u][0], seen[i]) for u in ranks}
        out["cells"][f"state{i}_block{j}"] = by_group({u: r for u, (r, _) in ranks.items()}, groups)
        if j in OWN and states_of(j) == i:
            for u, (rank, top) in ranks.items():
                target, history = users[j][u]
                pooled[groups[u]].append(rank)
                guess = top[0] if top else None
                first_code[groups[u]].append(prefix(code[target], guess) >= 1)
                if groups[u] == "cold":
                    cold.append({"rank": rank, "guess_prefix": prefix(code[target], guess),
                                 "list_prefix": max((prefix(code[target], c) for c in top),
                                                    default=0),
                                 "history_prefix": max((prefix(code[target], code[h])
                                                        for h in history), default=0)})  # fmt: skip
    out["own_blocks_1_3"] = {g: scores(pooled[g]) for g in GROUPS}
    out["top_guess_shares_first_code"] = {
        g: sum(v) / len(v) if v else None for g, v in first_code.items()
    }
    out["cold_mechanism"] = mechanism(cold) if cold else None
    last = max(i for i, _ in cells)
    retention = {g: {"own_hits": 0, "kept_at_last_state": 0} for g in GROUPS}
    for j in OWN:  # blocks every model has its own state for
        if j >= last or (j, j) not in cells or (last, j) not in cells:
            continue
        for u, (rank, _) in cells[j, j].items():
            if rank <= K:
                cell = retention[group(users[j][u][0], seen[j])]
                cell["own_hits"] += 1
                cell["kept_at_last_state"] += cells[last, j][u][0] <= K
    if any(c["own_hits"] for c in retention.values()):
        for c in retention.values():
            c["kept_share"] = c["kept_at_last_state"] / c["own_hits"] if c["own_hits"] else None
        out[f"retention_to_state{last}"] = retention
    return out


def dump_cells(folder, users, code):
    cells = {}
    for path in sorted(Path(folder).glob("state*_block*.jsonl")):
        i, j = (int(x) for x in path.stem.removeprefix("state").split("_block"))
        cells[i, j] = read_dump(path, users[j], code)
    return cells


def popular(blocks, users, seen):
    """Block j's training-target counts, lower item ID first on ties."""
    out, pooled = {}, {g: [] for g in GROUPS}
    for j in BLOCKS:
        counts = Counter(
            i for d in blocks[j].values() if len(d["current_block"]) >= 3
            for i in d["current_block"][:-2]
        )  # fmt: skip
        order = {
            item: k + 1 for k, item in enumerate(sorted(counts, key=lambda i: (-counts[i], int(i))))
        }
        ranks = {u: order.get(target, math.inf) for u, (target, _) in users[j].items()}
        groups = {u: group(users[j][u][0], seen[j]) for u in ranks}
        out[f"block{j}"] = by_group(ranks, groups)
        if j in OWN:
            for u, r in ranks.items():
                pooled[groups[u]].append(r)
    return out, {g: scores(pooled[g]) for g in GROUPS}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", help="Instruments_cl_ori: blocks and the semantic-ID index")
    parser.add_argument("output")
    parser.add_argument("--llm", nargs="*", default=[], help="dump dirs holding lora/ and peso/")
    parser.add_argument("--sasrec", nargs="*", default=[], help="`cp4285 blocks --dumps` runs")
    args = parser.parse_args()
    data, output = Path(args.data), Path(args.output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    blocks = [json.loads(next(data.glob(f"*.all.block_{b}.json")).read_text()) for b in BLOCKS]
    index = json.loads(next(data.glob("*.index.*.json")).read_text())
    code = {k: "".join(v) for k, v in index.items()}
    users = [test_users(b) for b in blocks]
    seen = exposure(blocks)
    report = {
        "groups_own_state": {
            f"block{j}": dict(Counter(group(t, seen[j]) for t, _ in users[j].values()))
            for j in BLOCKS
        },
        "models": {},
    }
    report["popular"], report["popular_own_blocks_1_3"] = popular(blocks, users, seen)
    for folder in map(Path, args.llm):
        label = f"{folder.parent.name}/{folder.name}"
        lora, peso = (dump_cells(folder / m, users, code) for m in ("lora", "peso"))
        pretrain = {k: v for k, v in lora.items() if k[0] == 0}
        report["models"][f"{label}/pretrain"] = model(pretrain, users, code, seen, lambda b: 0)
        report["models"][f"{label}/lora"] = model(lora, users, code, seen, lambda b: b)
        report["models"][f"{label}/peso"] = model(peso, users, code, seen, lambda b: b)
    for run in map(Path, args.sasrec):
        cells = dump_cells(run / "dumps", users, code)
        metrics = run / "metrics.json"
        arm = json.loads(metrics.read_text())["arm"] if metrics.exists() else "sasrec"
        report["models"][f"{run.name}/{arm}"] = model(cells, users, code, seen, lambda b: b)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as f:
        json.dump(report, f, indent=2)
    return report


if __name__ == "__main__":
    out = main()
    print(json.dumps(out["groups_own_state"]))
    for name, m in out["models"].items():
        own = m["own_blocks_1_3"]
        print(name, {g: (own[g]["users"], own[g].get("hits@20")) for g in GROUPS})
