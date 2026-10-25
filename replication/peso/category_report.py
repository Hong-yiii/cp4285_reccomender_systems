"""Category-target PESO: standard ranking metrics from prediction dumps, beside baselines.

Every score comes from the 1-based rank of the test target through
cp4285.experiment.metrics.from_ranks (Hit@K, NDCG@K, MRR@10). A target outside the
model's 20 beams counts as a miss, which leaves every metric at K <= 20 exact.
Test users are those upstream builds (continual_data_v2._process_test_data): each user
with at least three events in the block, the last event as target and the 20 before it
as history. Two baselines rank all categories for the same users:

- popular: block b's training-target counts, as popularity_shift.load_blocks counts them;
  categories without one rank last, in token order;
- history: the user's history counts, most recent first on ties, then popular.

Model scores are reported for every target and for targets whose PR 14 label is
observed rather than predicted, each beside both baselines on the same users (the
evaluator caps block 0 at a 2,500-user sample). Aggregates only.

    python category_report.py <category dataset dir> <new report.json> \
        [<dumps dir>/<method> ...]

Needs a checkout with cp4285.experiment.metrics on the path.
"""

import json
import math
import sys
from collections import Counter
from pathlib import Path

from popularity_shift import load_blocks

from cp4285.experiment.metrics import from_ranks

BLOCKS = range(5)
HISTORY = 20
KS = (1, 5, 10, 20)
KEEP = ["hit@1", "hit@5", "hit@10", "hit@20", "ndcg@5", "ndcg@10", "ndcg@20", "mrr@10"]


def metrics(ranks):
    m = from_ranks(ranks, KS)
    return {"users": len(ranks), **{k: m[k] for k in KEEP}}


def test_users(blocks, block, code):
    """user -> (target item, history codes), as upstream's test split builds them."""
    out = {}
    for user, data in blocks[block].items():
        if len(data["current_block"]) < 3:
            continue
        seq = data["previous_block"] + data["current_block"]
        out[user] = (seq[-1], [code[i] for i in seq[:-1][-HISTORY:]])
    return out


def baseline_ranks(users, code, popular):
    """user -> {"popular": rank, "history": rank} of the user's target, over all categories.

    Categories with no training target in the block rank last, in token order.
    """
    order = sorted(set(code.values()), key=lambda c: (-popular.get(c, 0), c))
    ranks = {}
    for user, (item, history) in users.items():
        target = code[item]
        counts = Counter(history)
        last = {c: i for i, c in enumerate(history)}
        own = sorted(counts, key=lambda c: (-counts[c], -last[c]))
        ranked = own + [c for c in order if c not in counts]
        ranks[user] = {"popular": order.index(target) + 1, "history": ranked.index(target) + 1}
    return ranks


def dump_cell(path, users, code, sources, baselines):
    """Metrics of one peso_eval.py dump, with both baselines on the same users."""
    ranks = {"all": [], "observed_targets": [], "popular": [], "history": []}
    for line in Path(path).read_text().splitlines():
        row = json.loads(line)
        item = users[row["user"]][0]
        if row["target"] != code[item]:
            raise ValueError(f"{path}: target mismatch for user {row['user']}")
        top = row["ranked"]
        rank = top.index(row["target"]) + 1 if row["target"] in top else math.inf
        ranks["all"].append(rank)
        if sources[item] == "observed":
            ranks["observed_targets"].append(rank)
        for name in ("popular", "history"):
            ranks[name].append(baselines[row["user"]][name])
    return {name: metrics(r) for name, r in ranks.items()}


def main(data_dir, output, *dumps):
    data_dir, output = Path(data_dir), Path(output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    dataset = data_dir.name  # Instruments_cl_cat, or Instruments_cl_cat<level> for deeper levels
    index = json.loads(next(data_dir.glob(f"{dataset}.index.*.json")).read_text())
    code = {k: "".join(v) for k, v in index.items()}
    sources = json.loads((data_dir / f"{dataset}.label_source.json").read_text())
    blocks = [json.loads((data_dir / f"{dataset}.all.block_{b}.json").read_text()) for b in BLOCKS]
    popular, _ = load_blocks(data_dir, index)
    users = [test_users(blocks, b, code) for b in BLOCKS]
    baselines = [baseline_ranks(users[b], code, popular[b]) for b in BLOCKS]
    report = {"metrics": KEEP, "categories": len(set(code.values())), "baselines": {}}
    for b in BLOCKS:
        for name in ("popular", "history"):
            ranks = [r[name] for r in baselines[b].values()]
            report["baselines"].setdefault(name, {})[f"block{b}"] = metrics(ranks)
    report["methods"] = {}
    for directory in map(Path, dumps):
        cells = {}
        for path in sorted(directory.glob("state*_block*.jsonl")):
            state, block = (int(x) for x in path.stem.removeprefix("state").split("_block"))
            cells[f"state{state}_block{block}"] = dump_cell(
                path, users[block], code, sources, baselines[block]
            )
        report["methods"][directory.name] = cells
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as f:
        json.dump(report, f, indent=2)
    return report


if __name__ == "__main__":
    out = main(*sys.argv[1:])
    for name, cells in out["baselines"].items():
        print(name, {b: round(m["hit@10"], 4) for b, m in cells.items()})
