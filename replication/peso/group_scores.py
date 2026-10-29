"""Hit@10 and NDCG@10 per user group, right after each block and at the end (seed 42).

Reads prediction dumps from `<predictions>/<method>/state{i}_block{j}.jsonl` (peso_eval.py)
and the new/dormant/continuing labels from user_groups.py. For each method and blocks
1-4 it scores every group at the block's own state and at state 4, so the change shows
which kinds of users are forgotten. Aggregates only.

    python group_scores.py <predictions> <user-groups.json> <new report.json> <method> [...]
"""

import json
import math
import sys
from pathlib import Path

K = 10


def scores(path, labels):
    """Per group: users, hits in the top 10 and summed NDCG@10."""
    out = {}
    with Path(path).open() as f:
        for line in f:
            row = json.loads(line)
            group = labels.get(row["user"])
            if group is None:
                continue
            cell = out.setdefault(group, {"users": 0, "hits": 0, "ndcg": 0.0})
            cell["users"] += 1
            top = row["ranked"][:K]
            if row["target"] in top:
                cell["hits"] += 1
                cell["ndcg"] += 1 / math.log2(top.index(row["target"]) + 2)
    return {
        g: {"users": c["users"], "hits": c["hits"], "hit@10": c["hits"] / c["users"],
            "ndcg@10": c["ndcg"] / c["users"]}
        for g, c in sorted(out.items())
    }  # fmt: skip


def main(predictions, groups_path, output, *methods):
    groups = json.loads(Path(groups_path).read_text())
    report = {"source": "runs/seed42/matrix-beam20", "methods": {}}
    for method in methods:
        cells = {}
        for block in (1, 2, 3, 4):
            labels = groups[str(block)]
            dumps = Path(predictions) / method
            cells[f"block{block}"] = {
                "own_state": scores(dumps / f"state{block}_block{block}.jsonl", labels),
                "state4": scores(dumps / f"state4_block{block}.jsonl", labels),
            }
        report["methods"][method] = cells
    with Path(output).open("x") as f:
        json.dump(report, f, indent=2)
    return report


if __name__ == "__main__":
    main(*sys.argv[1:])
