"""Wave-0 helper: per-domain natural drift, grid_aggregate's measure, dependency-free.

Mirrors the cluster's adapter_blocks.load and shilling.training_examples/js (24-26 October
checkout) without importing them, so it runs anywhere the PESO data lives — including the
login node, where the training venv is not unpacked. Read-only; prints one JSON.

    python3 grid_drift.py [DATA_DIR CATEGORIES_DIR]   # defaults to the cluster workspace
"""

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

DOMAINS = {"ins": "Instruments_cl_ori", "vg": "Video_Games_cl_0.6_0_5_3_0",
           "cd": "CDs_and_Vinyl_cl_0.6_0_5_3_0"}  # fmt: skip
BLOCKS = range(5)


def load(data_dir, categories_path):
    """Blocks and the item -> 0-based category map (adapter_blocks.load, names dropped)."""
    data_dir = Path(data_dir)
    blocks = [json.loads(next(data_dir.glob(f"*.all.block_{b}.json")).read_text())
              for b in BLOCKS]  # fmt: skip
    index = json.loads(Path(categories_path).read_text())
    category = {item: int(v[0].removeprefix("<cat_").removesuffix(">"))
                for item, v in index.items()}  # fmt: skip
    return blocks, category


def training_examples(users):
    """(user, target item) pairs of upstream's training split, nonempty history only."""
    out = []
    for user, data in users.items():
        if len(data["current_block"]) < 3:
            continue
        seq = data["previous_block"] + data["current_block"]
        out += [(user, seq[t]) for t in range(max(len(data["previous_block"]), 1), len(seq) - 2)]
    return out


def js(p, q):
    """Jensen-Shannon divergence (base 2) between two count distributions."""
    keys = sorted(set(p) | set(q))
    a = np.array([p.get(k, 0) for k in keys], float)
    b = np.array([q.get(k, 0) for k in keys], float)
    a, b = a / a.sum(), b / b.sum()
    m = (a + b) / 2
    kl = lambda x: float((x[x > 0] * np.log2(x[x > 0] / m[x > 0])).sum())
    return (kl(a) + kl(b)) / 2


def main():
    if len(sys.argv) > 2:
        data_root, cats_root = Path(sys.argv[1]), Path(sys.argv[2])
    else:
        data_root = Path("/mnt/scratch/w/wyenzhe/peso/peso/data")
        cats_root = Path("/mnt/scratch/w/wyenzhe/peso/categories")
    out = {"domains": {}}
    for short, name in DOMAINS.items():
        blocks, category = load(data_root / name,
                                cats_root / f"{name}-cat2" / f"{name}.index.cat2.json")  # fmt: skip
        mixes = [Counter(category[item] for _, item in training_examples(users))
                 for users in blocks]  # fmt: skip
        out["domains"][short] = {
            "natural_drift_js": [js(mixes[b], mixes[b + 1]) for b in range(len(mixes) - 1)]}
        print(f"{short}: {out['domains'][short]['natural_drift_js']}", file=sys.stderr)
    print(json.dumps(out))


if __name__ == "__main__":
    main()
