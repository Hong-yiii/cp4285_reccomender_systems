"""Summarize the item-level J-lens probe (jlens_items.py): aggregates only, no user IDs.

    python jlens_items_report.py <jlens_items dir> <first-code jlens dir> <new report.json>

Sections:
- **readout_quality:** MRR and top-10 share of the target's code at every level, layer and
  source (final output, J-lens, logit lens), against each level's chance MRR.
- **forgetting:** from each block's own state to the last (blocks 1-3), how each level's
  output rank and the item log-likelihood change.
- **internal_vs_output:** item-level change, as the sum over levels of log rank. It gives
  Spearman(output, lens), and the share of users whose output got worse while the lens
  did not, against a permutation baseline that breaks the user pairing.
- **drift_split:** total, item-head and inner-layer parts of each level's lens drift from
  state 0, relative to the state-0 lens.
- **level0_check:** agreement of level 0 with jlens.py's first-code readouts.

PESO's state 0 is single LoRA's (the shared block-0 checkpoint).
"""

import json
import math
import random
import statistics
import sys
from pathlib import Path

import torch

LEVELS = 4


def spearman(x, y):
    from scipy.stats import spearmanr

    rho = spearmanr(x, y).statistic
    return None if math.isnan(rho) else float(rho)


def load(root, method):
    rows = {}
    for state in range(5):
        path = root / method / f"state{state}" / "readouts.jsonl"
        if not path.exists() and method == "peso" and state == 0:
            path = root / "lora" / "state0" / "readouts.jsonl"
        if path.exists():
            for line in path.read_text().splitlines():
                r = json.loads(line)
                rows[state, r["block"], r["index"]] = r
    return rows


def sources(rows):
    keys = next(iter(rows.values()))
    layers = sorted({int(k.split("_")[1][1:]) for k in keys if k.startswith("jlens_l")})
    return ["final", *[f"jlens_l{x}" for x in layers], *[f"logit_l{x}" for x in layers]], layers


def chance_mrr(n):
    return sum(1 / k for k in range(1, n + 1)) / n


def readout_quality(rows, names, sizes, states):
    out = {}
    for state in states:
        recs = [r for (s, _, _), r in rows.items() if s == state]
        if not recs:
            continue
        out[f"state{state}"] = {
            f"c{c}": {
                "chance_mrr": chance_mrr(sizes[c]),
                **{
                    name: {
                        "mrr": statistics.mean(1 / r[f"{name}_c{c}"] for r in recs),
                        "top10": statistics.mean(r[f"{name}_c{c}"] <= 10 for r in recs),
                    }
                    for name in names
                },
            }
            for c in range(LEVELS)
        }
    return out


def item_score(r, name):
    """Sum over levels of log rank: 0 when every level ranks the target's code first."""
    return sum(math.log(r[f"{name}_c{c}"]) for c in range(LEVELS))


def pairs(rows, last=4):
    """(own-state record, last-state record) for blocks 1..last-1."""
    for (state, block, index), r in rows.items():
        if state == block and 1 <= block < last and (last, block, index) in rows:
            yield block, r, rows[last, block, index]


def forgetting(rows):
    pp = list(pairs(rows))
    out = {"users": len(pp)}
    for c in range(LEVELS):
        d = [b[f"final_c{c}"] - a[f"final_c{c}"] for _, a, b in pp]
        out[f"c{c}"] = {
            "mean_rank_change": statistics.mean(d),
            "share_worse": statistics.mean(x > 0 for x in d),
        }
    d = [b["loglik_final"] - a["loglik_final"] for _, a, b in pp]
    out["item_loglik"] = {
        "mean_change": statistics.mean(d),
        "share_worse": statistics.mean(x < 0 for x in d),
    }
    return out


def internal_vs_output(rows, layers, permutations=1000, seed=4285):
    pp = list(pairs(rows))
    d_out = [item_score(b, "final") - item_score(a, "final") for _, a, b in pp]
    blocks = [block for block, _, _ in pp]
    rng = random.Random(seed)
    by_block = {b: [k for k, x in enumerate(blocks) if x == b] for b in sorted(set(blocks))}
    out = {}
    for layer in layers:
        name = f"jlens_l{layer}"
        d_in = [item_score(b, name) - item_score(a, name) for _, a, b in pp]

        def share(internal):
            hits = [o > 0 and i <= 0 for o, i in zip(d_out, internal)]
            return statistics.mean(hits)

        null = []
        for _ in range(permutations):  # shuffle the lens change within each block
            shuffled = list(d_in)
            for idx in by_block.values():
                values = [d_in[k] for k in idx]
                rng.shuffle(values)
                for k, v in zip(idx, values):
                    shuffled[k] = v
            null.append(share(shuffled))
        out[f"l{layer}"] = {
            "spearman_output_vs_lens": spearman(d_out, d_in),
            "share_output_worse_lens_not": share(d_in),
            "null_mean": statistics.mean(null),
            "null_sd": statistics.stdev(null),
        }
    return {"users": len(pp), "share_output_worse": statistics.mean(x > 0 for x in d_out), **out}


def drift_split(root, method):
    ref = torch.load(root / "lora" / "state0" / "lens.pt")["rows"]
    out = {}
    for state in range(1, 5):
        path = root / method / f"state{state}" / "lens.pt"
        if not path.exists():
            continue
        lens = torch.load(path)
        own, at_ref = lens["rows"], lens["rows_ref"]
        out[f"state{state}"] = {
            f"c{c}": {
                f"l{layer}": {
                    "total_rel": float(
                        (own[c][layer] - ref[c][layer]).norm() / ref[c][layer].norm()
                    ),
                    "head_rel": float(
                        (own[c][layer] - at_ref[c][layer]).norm() / ref[c][layer].norm()
                    ),
                    "inner_rel": float(
                        (at_ref[c][layer] - ref[c][layer]).norm() / ref[c][layer].norm()
                    ),
                }
                for layer in own[c]
            }
            for c in range(LEVELS)
        }
    return out


def level0_check(rows, first_code_root, method):
    old = {}
    for state in range(5):
        path = first_code_root / method / f"state{state}" / "readouts.jsonl"
        if path.exists():
            for line in path.read_text().splitlines():
                r = json.loads(line)
                old[state, r["block"], r["index"]] = r
    out = {}
    for name in ("final", "jlens_l4", "jlens_l8", "jlens_l12", "logit_l4", "logit_l8", "logit_l12"):
        shared = [
            (rows[k][f"{name}_c0"], old[k][name]) for k in rows if k in old and name in old[k]
        ]
        if shared:
            out[name] = {"users": len(shared), "identical": statistics.mean(a == b for a, b in shared),
                         "mean_abs_diff": statistics.mean(abs(a - b) for a, b in shared)}  # fmt: skip
    return out


def main(root, first_code_root, output):
    root, first_code_root = Path(root), Path(first_code_root)
    lens0 = torch.load(root / "lora" / "state0" / "lens.pt")
    sizes = [len(t) for t in lens0["tokens"]]
    report = {"source": str(root), "level_sizes": sizes, "methods": {}}
    for method in ("lora", "peso"):
        rows = load(root, method)
        if not rows:
            continue
        names, layers = sources(rows)
        states = sorted({s for s, _, _ in rows})
        report["methods"][method] = {
            "states": states,
            "readout_quality": readout_quality(rows, names, sizes, states),
            "forgetting": forgetting(rows) if 4 in states and 1 in states else None,
            "internal_vs_output": internal_vs_output(rows, layers) if 1 in states else None,
            "drift_split": drift_split(root, method),
            "level0_check": level0_check(rows, first_code_root, method),
        }
    with Path(output).open("x") as f:
        json.dump(report, f, indent=2)
    return report


if __name__ == "__main__":
    main(*sys.argv[1:4])
