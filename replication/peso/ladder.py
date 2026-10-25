"""Experiment 1: does continual learning matter more as targets get finer?

Each rung of the ladder labels the same events at one depth: PR 14 subcategories at
levels 2, 3 and 4 (category_index.py datasets, scored by category_report.py) and the
products themselves (peso_eval.py matrix dumps). Per rung and seed, from the transfer
matrix R[i][j] (state i on block j's test users; state 0, shared by both methods, is
Pretrain):

- update gain: single LoRA's own state against Pretrain on blocks 1-3, absolute and as
  the mean relative change;
- PESO - single LoRA on the same own-state cells;
- drift after k later updates: the mean over blocks j of R[j+k][j] / R[j][j] - 1, for
  every k the matrix allows (Finding 5), and backward transfer, the mean over blocks 0-3
  of R[4][j] - R[j][j].

From the data alone: the effective number of labels (exp of the entropy of a block's test
targets) and the largest label's share, each averaged over blocks 1-4, and the
Jensen-Shannon divergence (base 2) between consecutive blocks' training-target mixes.
Across rungs and seeds, a least-squares line of each relative quantity on log10 effective
labels tests whether its slope is zero; the drift after one update is also set against
the divergence to the next block.

    python ladder.py <new report.json> \
        --level 2 <category dataset dir> <category_report.json> [...] \
        --level 3 ... --products <Instruments_cl_ori dir> <dump dir with lora/, peso/> [...]

Aggregates only. Needs a checkout with cp4285 on the path.
"""

import argparse
import json
import math
import statistics
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.stats import linregress, pearsonr

from cp4285.experiment.metrics import from_ranks

BLOCKS = range(5)
OWN = (1, 2, 3)
HISTORY = 20  # PESO's history window
METRICS = ("hit@1", "ndcg@10", "mrr@10")


def labels(data_dir):
    """Blocks and item -> label: one category token, or the product's semantic ID."""
    data_dir = Path(data_dir)
    blocks = [json.loads(next(data_dir.glob(f"*.all.block_{b}.json")).read_text()) for b in BLOCKS]
    index = json.loads(next(data_dir.glob("*.index.*.json")).read_text())
    return blocks, {item: "".join(v) for item, v in index.items()}


def mixes(blocks, label):
    """Per block: training-target and test-target label counts, as upstream splits them."""
    train, test = [], []
    for users in blocks:
        kept = [d["current_block"] for d in users.values() if len(d["current_block"]) >= 3]
        train.append(Counter(label[i] for c in kept for i in c[:-2]))
        test.append(Counter(label[c[-1]] for c in kept))
    return train, test


def js(p, q):
    keys = set(p) | set(q)
    a = np.array([p.get(k, 0) for k in keys], float)
    b = np.array([q.get(k, 0) for k in keys], float)
    a, b = a / a.sum(), b / b.sum()
    m = (a + b) / 2

    def kl(x):
        nz = x > 0
        return float((x[nz] * np.log2(x[nz] / m[nz])).sum())

    return (kl(a) + kl(b)) / 2


def repeat_share(blocks, label):
    """Share of blocks 1-4's test targets whose label is among the 20 events before them."""
    repeats = total = 0
    for users in blocks[1:]:
        for d in users.values():
            if len(d["current_block"]) >= 3:
                seq = d["previous_block"] + d["current_block"]
                repeats += label[seq[-1]] in {label[i] for i in seq[:-1][-HISTORY:]}
                total += 1
    return repeats / total


def data_view(blocks, label):
    """Label count, per-block means over blocks 1-4 of exp(entropy) and the top share, the
    repeat share and the block-to-block divergence of training targets."""
    train, test = mixes(blocks, label)
    effective, largest = [], []
    for counts in test[1:]:
        total = sum(counts.values())
        effective.append(math.exp(-sum(n / total * math.log(n / total) for n in counts.values())))
        largest.append(max(counts.values()) / total)
    return {"labels": len(set(label.values())), "effective_labels": statistics.mean(effective),
            "largest_share": statistics.mean(largest), "repeat_share": repeat_share(blocks, label),
            "js_to_previous_block": {f"block{b}": js(train[b], train[b - 1]) for b in range(1, 5)}}  # fmt: skip


def from_report(path):
    """{method: {(i, j): metrics}} from one category_report.py report."""
    methods = json.loads(Path(path).read_text())["methods"]
    out = {}
    for method, cells in methods.items():
        for name, cell in cells.items():
            i, j = (int(x) for x in name.removeprefix("state").split("_block"))
            out.setdefault(method, {})[i, j] = cell["all"]
    return out


def from_dumps(folder):
    """{method: {(i, j): metrics}} scored from peso_eval.py dumps through from_ranks."""
    out = {}
    for method in ("lora", "peso"):
        for path in sorted((Path(folder) / method).glob("state*_block*.jsonl")):
            i, j = (int(x) for x in path.stem.removeprefix("state").split("_block"))
            ranks = []
            for line in path.read_text().splitlines():
                row = json.loads(line)
                top = row["ranked"]
                ranks.append(top.index(row["target"]) + 1 if row["target"] in top else math.inf)
            out.setdefault(method, {})[i, j] = from_ranks(ranks, (1, 10))
    return out


def seed_view(matrix):
    """Update gain, PESO - single LoRA, drift after k updates and backward transfer."""
    lora, peso = matrix["lora"], {**matrix.get("peso", {})}
    peso.update({k: v for k, v in lora.items() if k[0] == 0})  # PESO starts from state 0
    out = {}
    for m in METRICS:
        cell = {}
        if all((b, b) in lora and (0, b) in lora for b in OWN):
            cell["update_gain"] = statistics.mean(lora[b, b][m] - lora[0, b][m] for b in OWN)
            cell["update_gain_relative"] = statistics.mean(
                lora[b, b][m] / lora[0, b][m] - 1 for b in OWN
            )
        if all((b, b) in peso for b in OWN):
            cell["peso_minus_lora"] = statistics.mean(peso[b, b][m] - lora[b, b][m] for b in OWN)
        for name, states in (("lora", lora), ("peso", peso)):
            drift = {}
            for k in range(1, 5):
                pairs = [(j + k, j) for j in BLOCKS if (j + k, j) in states
                         and (j, j) in states and states[j, j][m] > 0]  # fmt: skip
                if pairs:
                    drift[str(k)] = statistics.mean(
                        states[i, j][m] / states[j, j][m] - 1 for i, j in pairs
                    )
            cell[f"{name}_drift_after_k"] = drift
            if all((4, j) in states and (j, j) in states for j in range(4)):
                cell[f"{name}_backward_transfer"] = statistics.mean(
                    states[4, j][m] - states[j, j][m] for j in range(4)
                )
        out[m] = cell
    return out


def slope(points, x_key, y_key):
    rows = [(p[x_key], p[y_key]) for p in points if p.get(y_key) is not None]
    if len({x for x, _ in rows}) < 2:
        return None
    fit = linregress(*zip(*rows))
    return {"points": len(rows), "slope": fit.slope, "slope_se": fit.stderr,
            "p_value": fit.pvalue, "intercept": fit.intercept}  # fmt: skip


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output")
    parser.add_argument("--level", nargs="+", action="append", default=[],
                        metavar="LEVEL DATA REPORT", help="a level, its dataset, its reports")  # fmt: skip
    parser.add_argument("--products", nargs="+", metavar="DATA DUMPS",
                        help="Instruments_cl_ori and matrix dump dirs, one per seed")  # fmt: skip
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    rungs = [(f"level {lv}", data, [from_report(r) for r in reports], reports)
             for lv, data, *reports in args.level]  # fmt: skip
    if args.products:
        data, *dumps = args.products
        rungs.append(("products", data, [from_dumps(d) for d in dumps], dumps))
    report, points = {"rungs": {}}, []
    for name, data, matrices, sources in rungs:
        view = data_view(*labels(data))
        seeds = [seed_view(m) for m in matrices]
        report["rungs"][name] = {"data": view, "sources": [str(s) for s in sources],
                                 "seeds": seeds}  # fmt: skip
        for seed, source in zip(seeds, sources):
            for m in METRICS:
                c = seed[m]
                points.append({
                    "rung": name, "metric": m, "source": str(source),
                    "log10_effective_labels": math.log10(view["effective_labels"]),
                    "update_gain_relative": c.get("update_gain_relative"),
                    "drift_after_1": c["lora_drift_after_k"].get("1"),
                    "drift_after_4": c["lora_drift_after_k"].get("4"),
                })  # fmt: skip
    report["slopes_on_log10_effective_labels"] = {
        m: {y: slope([p for p in points if p["metric"] == m], "log10_effective_labels", y)
            for y in ("update_gain_relative", "drift_after_1", "drift_after_4")}
        for m in METRICS
    }  # fmt: skip
    # Drift of block j after the next update against the shift to block j + 1's mix.
    pairs = {m: [] for m in METRICS}
    for name, data, matrices, _ in rungs:
        shift = report["rungs"][name]["data"]["js_to_previous_block"]
        for matrix in matrices:
            lora = matrix["lora"]
            for j in range(4):
                if (j, j) in lora and (j + 1, j) in lora and min(lora[j, j][m] for m in METRICS):
                    for m in METRICS:
                        drop = lora[j + 1, j][m] / lora[j, j][m] - 1
                        pairs[m].append((shift[f"block{j + 1}"], drop))
    report["drift_after_1_vs_js"] = {
        m: dict(zip(("pearson_r", "p_value"), map(float, pearsonr(*zip(*v)))), points=len(v))
        if len(v) > 2 else None
        for m, v in pairs.items()
    }  # fmt: skip
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as f:
        json.dump(report, f, indent=2)
    return report


if __name__ == "__main__":
    out = main()
    for name, rung in out["rungs"].items():
        d = rung["data"]
        print(name, "labels", d["labels"], "effective", round(d["effective_labels"], 1),
              "JS", {k: round(v, 4) for k, v in d["js_to_previous_block"].items()})  # fmt: skip
        for seed in rung["seeds"]:
            c = seed["ndcg@10"]
            print("   ndcg@10 gain", round(c.get("update_gain_relative", float("nan")), 4),
                  "drift", {k: round(v, 4) for k, v in c["lora_drift_after_k"].items()})  # fmt: skip
