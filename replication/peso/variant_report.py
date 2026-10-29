"""Compare a penalty variant with single LoRA and released PESO on the same users (seed 42).

Reads prediction dumps from `<predictions>/<method>/state{i}_block{j}.jsonl` (peso_eval.py).
It uses the pairs every method has: each block's own state, and the last state on every
block. State 0 is the shared block-0 checkpoint, whose dumps live under single LoRA.
Aggregates only:
- NDCG@10/20 and Hit@10/20 per pair;
- backward transfer, the mean over blocks 0-3 of R[4][j] - R[j][j];
- the own-state (new-block) average over blocks 1-3 and 1-4;
- head-to-head flips (variant vs single LoRA) with exact McNemar tests.

    python variant_report.py <predictions> <variant> <user-groups.json> <new report.json>
"""

import json
import statistics
import sys
from pathlib import Path

try:
    from cp4285.experiment import flips
except ImportError:  # on the GPU pod, with flips.py copied alongside
    import flips

PAIRS = [(4, 0), (4, 1), (4, 2), (4, 3), (4, 4), (1, 1), (2, 2), (3, 3)]


def dumps(predictions, method):
    """(state, block) -> {user: row}, with the shared state-0 dumps added."""
    out = flips.load(predictions / method)
    shared = flips.load(predictions / "lora")
    out.update({key: rows for key, rows in shared.items() if key[0] == 0})
    return out


def summary(table):
    cells = {
        f"state{s}_block{b}": {**flips.metrics(table[s, b], 10), **flips.metrics(table[s, b], 20)}
        for s, b in [*PAIRS, (0, 0)]
    }
    out = {"cells": cells}
    for name in ("ndcg@10", "ndcg@20", "hit@10"):
        own = {j: cells[f"state{j}_block{j}"][name] for j in range(5)}
        last = {j: cells[f"state4_block{j}"][name] for j in range(5)}
        out[name] = {
            "backward_transfer": statistics.mean(last[j] - own[j] for j in range(4)),
            "last_state_old_blocks_mean": statistics.mean(last[j] for j in range(4)),
            "own_state_blocks_1_3": statistics.mean(own[j] for j in (1, 2, 3)),
            "own_state_blocks_1_4": statistics.mean(own[j] for j in (1, 2, 3, 4)),
        }
    return out


def main(predictions, variant, groups_path, output):
    predictions = Path(predictions)
    groups = json.loads(Path(groups_path).read_text())
    tables = {m: dumps(predictions, m) for m in ("lora", "peso", variant)}
    report = {
        "variant": variant,
        "pairs": [f"state{s}_block{b}" for s, b in PAIRS],
        "methods": {m: summary(t) for m, t in tables.items()},
        "head_to_head": {},
    }
    for k in (10, 20):
        # "lost" = single-LoRA hit that the variant misses; "gained" = the reverse.
        report["head_to_head"][f"k{k}"] = {
            f"state{s}_block{b}": flips.flips(
                tables["lora"][s, b], tables[variant][s, b], k, groups.get(str(b))
            )
            for s, b in PAIRS
        }
    with Path(output).open("x") as f:
        json.dump(report, f, indent=2)
    return report


if __name__ == "__main__":
    main(*sys.argv[1:5])
