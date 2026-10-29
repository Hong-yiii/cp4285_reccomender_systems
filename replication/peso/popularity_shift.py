"""Is forgetting a shift toward newer, now-popular items? (seed 42, CPU only)

Compares a method's per-user top-10 lists (peso_eval.py dumps) with how item popularity
moves between blocks. Popularity is counted over each block's training targets, as
upstream builds them (continual_data_v2._process_train_data: every item of a user's
current block except the last two, for users with at least three). Test targets are each
user's last item in the block, as upstream tests.

Three measurements:
- **Popularity baseline matrix:** Hit@10 and NDCG@10 of "recommend block b's ten most
  popular training items" on block j's test users (the dumps' users). If old users are
  served best by old popularity, a model that tracks current popularity must lose them.
- **Lost vs kept targets,** own state j -> state 4, for blocks 0-3: each hit target's
  training share in block j and in block 4, and the same for its first code (the
  broad group, where the J-lens places forgetting). Mann-Whitney tests compare the
  log share ratios of lost and kept targets.
- **Recommendation age:** for block-j users, the share of top-10 slots taken by items
  first trained on after block j, at state j and at state 4.

    python popularity_shift.py <dumps dir of one method> <data dir> <index.json> <new report.json> \
        [<single LoRA dumps dir, for the shared state-0 dumps>]

The data dir holds upstream's Instruments_cl_ori.all.block_{0..4}.json.
"""

import json
import math
import statistics
import sys
from collections import Counter
from pathlib import Path

from scipy.stats import mannwhitneyu

BLOCKS = range(5)
K = 10


def load_blocks(data_dir, index):
    """Per block: training-target counts (item codes) and first-code counts."""
    code = {k: "".join(v) for k, v in index.items()}
    items, groups = [], []
    for b in BLOCKS:
        users = json.loads(next(Path(data_dir).glob(f"*.all.block_{b}.json")).read_text())
        counts = Counter(
            code[i] for u in users.values() if len(u["current_block"]) >= 3
            for i in u["current_block"][:-2]
        )  # fmt: skip
        items.append(counts)
        groups.append(Counter({}))
        for item, n in counts.items():
            groups[b][first_code(item)] += n
    return items, groups


def first_code(item):
    return item[: item.index(">") + 1]


def share(counts, key):
    """Training share with add-one smoothing, so unseen items get a finite log ratio."""
    total = sum(counts.values())
    return (counts.get(key, 0) + 1) / (total + len(counts) + 1)


def dumps(path, state, block):
    rows = (json.loads(line) for line in (Path(path) / f"state{state}_block{block}.jsonl").open())
    return {r["user"]: r for r in rows}


def hit(row):
    return row["target"] in row["ranked"][:K]


def popularity_matrix(items, users_by_block):
    out = {"hit@10": [], "ndcg@10": []}
    for b in BLOCKS:
        top = [i for i, _ in items[b].most_common(K)]
        rank = {i: r for r, i in enumerate(top)}
        hits, ndcgs = [], []
        for j in BLOCKS:
            targets = [r["target"] for r in users_by_block[j].values()]
            hits.append(sum(t in rank for t in targets) / len(targets))
            ndcgs.append(
                sum(1 / math.log2(rank[t] + 2) for t in targets if t in rank) / len(targets)
            )
        out["hit@10"].append(hits)
        out["ndcg@10"].append(ndcgs)
    return out


def log_ratio(counts_late, counts_early, key):
    return math.log(share(counts_late, key) / share(counts_early, key))


def compare(lost, kept):
    if not lost or not kept:
        return {"lost_median": None, "kept_median": None, "p": None}
    return {
        "lost_median": statistics.median(lost), "kept_median": statistics.median(kept),
        "lost_mean": statistics.fmean(lost), "kept_mean": statistics.fmean(kept),
        "p": float(mannwhitneyu(lost, kept).pvalue),
    }  # fmt: skip


def main(dump_dir, data_dir, index_file, output, state0_dir=None):
    index = json.loads(Path(index_file).read_text())
    items, groups = load_blocks(data_dir, index)
    first_seen = {}
    for b in BLOCKS:
        for item in items[b]:
            first_seen.setdefault(item, b)
    # State 0 is the shared block-0 checkpoint: variants without its dumps use single LoRA's.
    shared = dump_dir if (Path(dump_dir) / "state0_block0.jsonl").exists() else state0_dir
    own = {j: dumps(shared if j == 0 else dump_dir, j, j) for j in BLOCKS}
    report = {
        "source": str(dump_dir),
        "training_targets_per_block": [sum(c.values()) for c in items],
        "distinct_items_per_block": [len(c) for c in items],
        "popularity_baseline": popularity_matrix(items, own),
        "blocks": {},
    }
    pooled = {"lost": [], "kept": []}  # blocks 1-3; block 0 is far larger and older
    for j in range(4):
        before, after = own[j], dumps(dump_dir, 4, j)
        users = sorted(before.keys() & after.keys())
        kept = [u for u in users if hit(before[u]) and hit(after[u])]
        lost = [u for u in users if hit(before[u]) and not hit(after[u])]
        gained = [u for u in users if not hit(before[u]) and hit(after[u])]
        item_ratio = {u: log_ratio(items[4], items[j], before[u]["target"]) for u in users}
        group_ratio = {
            u: log_ratio(groups[4], groups[j], first_code(before[u]["target"])) for u in users
        }
        newer = {
            name: statistics.fmean(
                sum(first_seen.get(i, 99) > j for i in rows[u]["ranked"][:K]) / K for u in users
            )
            for name, rows in (("own_state", before), ("state4", after))
        }  # fmt: skip
        report["blocks"][f"block{j}"] = {
            "users": len(users), "kept": len(kept), "lost": len(lost), "gained": len(gained),
            "item_share_log_ratio_4_vs_j": compare(
                [item_ratio[u] for u in lost], [item_ratio[u] for u in kept]
            ),
            "group_share_log_ratio_4_vs_j": compare(
                [group_ratio[u] for u in lost], [group_ratio[u] for u in kept]
            ),
            "gained_item_share_log_ratio_median": (
                statistics.median(item_ratio[u] for u in gained) if gained else None
            ),
            "top10_share_first_trained_after_block": newer,
        }  # fmt: skip
        print(json.dumps({f"block{j}": report["blocks"][f"block{j}"]}), flush=True)
        if j > 0:
            pooled["lost"] += [item_ratio[u] for u in lost]
            pooled["kept"] += [item_ratio[u] for u in kept]
    report["blocks_1_3_item_share_log_ratio"] = compare(pooled["lost"], pooled["kept"])
    with Path(output).open("x") as f:
        json.dump(report, f, indent=2)
    return report


if __name__ == "__main__":
    main(*sys.argv[1:6])
