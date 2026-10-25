"""The prefix run: does a category token before the item codes help item-level prediction?

Both arms train on the same blocks, seed and settings (category_prefix.py builds the
prefixed dataset); they differ only in the item IDs. The prefixed arm decodes
<cat_k><a_..><b_..><c_..><d_..>, the item-only arm the four codes. A hit needs the exact
item in both, so the category token is stripped from the prefixed dumps and both arms are
scored on the same users through cp4285.experiment.metrics:

- own-state cells (Pretrain = single LoRA's state 0) on blocks 1-3, per arm and method;
- prefixed minus item-only per user on those cells, with the standard error over users and
  exact McNemar tests on Hit@10 flips;
- how often each arm's top guess lies in the target's category (the prefixed arm names a
  category first; the item-only arm's guess is mapped to its item's category);
- forgetting: drift after k later updates and backward transfer (ladder.py), per arm.

    python prefix_report.py <Instruments_cl_catitem dir> <new report.json> \
        <item-only matrix dumps dir> <prefixed matrix dumps dir>

Each dumps dir holds lora/ and peso/ from peso_eval.py. Aggregates only.
"""

import json
import math
import re
import statistics
import sys
from pathlib import Path

from ladder import seed_view
from scipy.stats import binomtest

from cp4285.experiment.metrics import from_ranks

OWN = (1, 2, 3)
KS = (1, 5, 10, 20)
KEEP = ("hit@1", "hit@5", "hit@10", "hit@20", "ndcg@10", "mrr@10")
CATEGORY = re.compile(r"^<cat_\d+>")
METHODS = {"pretrain": ("lora", lambda b: 0), "lora": ("lora", lambda b: b),
           "peso": ("peso", lambda b: b)}  # fmt: skip


def strip(item):
    return CATEGORY.sub("", item)


def dump(path):
    """user -> (target, ranked, raw top guess), the category token stripped from both."""
    out = {}
    for line in Path(path).read_text().splitlines():
        row = json.loads(line)
        top = row["ranked"][0] if row["ranked"] else None
        out[row["user"]] = (strip(row["target"]), [strip(s) for s in row["ranked"]], top)
    return out


def rank(target, ranked):
    return ranked.index(target) + 1 if target in ranked else math.inf


def score(r, metric):
    """Per-user Hit@10 or NDCG@10 from a 1-based rank, as cp4285.experiment.metrics defines it."""
    if r > 10:
        return 0.0
    return 1 / math.log2(r + 1) if metric == "ndcg@10" else 1.0


def matrix(folder):
    """{method dir: {(i, j): metrics}} for ladder.seed_view, on stripped strings."""
    out = {}
    for method in ("lora", "peso"):
        for path in sorted((Path(folder) / method).glob("state*_block*.jsonl")):
            i, j = (int(x) for x in path.stem.removeprefix("state").split("_block"))
            ranks = [rank(t, r) for t, r, _ in dump(path).values()]
            out.setdefault(method, {})[i, j] = from_ranks(ranks, KS)
    return out


def main(data_dir, output, item_dir, prefixed_dir):
    data_dir, output = Path(data_dir), Path(output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    index = json.loads(next(data_dir.glob("*.index.catitem.json")).read_text())
    category_of = {"".join(v[1:]): v[0] for v in index.values()}  # item codes -> <cat_k>
    report = {"arms": {"item-only": str(item_dir), "prefixed": str(prefixed_dir)}, "methods": {}}
    for name, (source, state_of) in METHODS.items():
        cells = {}
        for arm, folder in (("item-only", item_dir), ("prefixed", prefixed_dir)):
            cells[arm] = {b: dump(Path(folder) / source / f"state{state_of(b)}_block{b}.jsonl")
                          for b in OWN}  # fmt: skip
        out = {}
        for arm, blocks in cells.items():
            ranks = [rank(t, r) for b in OWN for t, r, _ in blocks[b].values()]
            m = from_ranks(ranks, KS)
            right = []
            for b in OWN:
                for t, r, top in blocks[b].values():
                    guess = (top[: top.index(">") + 1] if arm == "prefixed" and top
                             else category_of.get(r[0]) if r else None)  # fmt: skip
                    right.append(guess == category_of[t])
            out[arm] = {"users": len(ranks), **{k: m[k] for k in KEEP},
                        "top_guess_in_target_category": statistics.mean(right)}  # fmt: skip
        deltas, flips = {}, {"lost": 0, "gained": 0}
        for metric in ("hit@10", "ndcg@10"):
            diffs = []
            for b in OWN:
                a, p = cells["item-only"][b], cells["prefixed"][b]
                if a.keys() != p.keys():
                    raise ValueError(f"Arms scored different users in block {b}")
                for u in a:
                    ra, rp = rank(*a[u][:2]), rank(*p[u][:2])
                    diffs.append(score(rp, metric) - score(ra, metric))
                    if metric == "hit@10" and (ra <= 10) != (rp <= 10):
                        flips["gained" if rp <= 10 else "lost"] += 1
            deltas[metric] = {"mean": statistics.mean(diffs),
                              "se": statistics.stdev(diffs) / math.sqrt(len(diffs))}  # fmt: skip
        n = flips["lost"] + flips["gained"]
        flips["mcnemar_p"] = binomtest(flips["gained"], n).pvalue if n else None
        out["prefixed_minus_item_only"] = {**deltas, "hit@10_flips": flips}
        report["methods"][name] = out
    report["forgetting"] = {arm: seed_view(matrix(folder))
                            for arm, folder in (("item-only", item_dir), ("prefixed", prefixed_dir))}  # fmt: skip
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as f:
        json.dump(report, f, indent=2)
    return report


if __name__ == "__main__":
    result = main(*sys.argv[1:])
    for name, m in result["methods"].items():
        d = m["prefixed_minus_item_only"]
        print(name, {arm: round(m[arm]["ndcg@10"], 5) for arm in ("item-only", "prefixed")},
              "delta ndcg@10", round(d["ndcg@10"]["mean"], 5), "+-", round(d["ndcg@10"]["se"], 5),
              "flips", d["hit@10_flips"])  # fmt: skip
