"""Item IDs with their PR 14 subcategory in front: [<cat_k>, <a_..>, <b_..>, <c_..>, <d_..>].

Reads the released Instruments_cl_ori data and category_index.py's category index, and
writes Instruments_cl_catitem: the same blocks and item metadata, with each item's
category token before its four semantic-ID codes. Upstream then trains on histories that
carry categories and decodes the category before the codes; a hit still needs the exact
item, so scores stay comparable with item-level runs.

    python category_prefix.py <Instruments_cl_ori dir> <Instruments_cl_cat dir> <new dataset dir>
"""

import json
import shutil
import sys
from collections import Counter
from pathlib import Path

from category_index import BLOCKS, SOURCE
from category_index import DATASET as CATEGORIES
from category_index import INDEX as CATEGORY_INDEX

DATASET = "Instruments_cl_catitem"
INDEX = ".index.catitem.json"
ITEM_INDEX = ".index.epoch20000_edim32_beta0.0.json"


def main(source_dir, category_dir, out_dir):
    source_dir, out_dir = Path(source_dir), Path(out_dir)
    codes = json.loads((source_dir / f"{SOURCE}{ITEM_INDEX}").read_text())
    category = json.loads((Path(category_dir) / f"{CATEGORIES}{CATEGORY_INDEX}").read_text())
    if codes.keys() != category.keys():
        raise ValueError("Item and category indexes cover different items")
    out_dir.mkdir(parents=True, exist_ok=False)
    for b in BLOCKS:
        name = f"all.block_{b}.json"
        shutil.copyfile(source_dir / f"{SOURCE}.{name}", out_dir / f"{DATASET}.{name}")
    shutil.copyfile(source_dir / f"{SOURCE}.item.json", out_dir / f"{DATASET}.item.json")
    index = {i: category[i] + codes[i] for i in sorted(codes, key=int)}
    (out_dir / f"{DATASET}{INDEX}").write_text(json.dumps(index))
    # Upstream scores exact strings, so items sharing an ID are indistinguishable.
    shared = {
        name: sum(n for n in Counter("".join(v) for v in ids.values()).values() if n > 1)
        for name, ids in (("released", codes), ("prefixed", index))
    }
    print(json.dumps({"dataset": DATASET, "items": len(index), "items_sharing_an_id": shared}))


if __name__ == "__main__":
    main(*sys.argv[1:])
