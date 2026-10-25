"""Item-level PESO predictions read as category predictions, for a like-for-like test.

Rewrites peso_eval.py dumps of an item-level run so that category_report.py can score
them beside category-trained runs: each target becomes its item's PR 14 category, taken
from the blocks, and each ranked list becomes its items' categories in order, first
occurrence kept. Position-wise decoding can produce codes that name no item; those are
dropped. A code shared by several items maps to the lowest item ID's category (0.4% of
test targets have a shared code).

    python category_projection.py <item dumps dir> <Instruments_cl_ori dir> \
        <Instruments_cl_cat dir> <new dumps dir>
"""

import json
import sys
from pathlib import Path

from category_index import DATASET, INDEX, SOURCE
from category_prefix import ITEM_INDEX


def main(dumps, source_dir, category_dir, out_dir):
    dumps, out_dir = Path(dumps), Path(out_dir)
    codes = json.loads((Path(source_dir) / f"{SOURCE}{ITEM_INDEX}").read_text())
    category = json.loads((Path(category_dir) / f"{DATASET}{INDEX}").read_text())
    by_code = {}
    for item in sorted(codes, key=int, reverse=True):  # lowest ID written last, so it wins
        by_code["".join(codes[item])] = category[item][0]
    blocks = [
        json.loads((Path(source_dir) / f"{SOURCE}.all.block_{b}.json").read_text())
        for b in range(5)
    ]
    out_dir.mkdir(parents=True, exist_ok=False)
    stats = {"rows": 0, "dropped_codes": 0}
    for path in sorted(dumps.glob("state*_block*.jsonl")):
        block = int(path.stem.split("_block")[1])
        rows = []
        for line in path.read_text().splitlines():
            row = json.loads(line)
            item = blocks[block][row["user"]]["current_block"][-1]
            ranked = []
            for code in row["ranked"]:
                if code not in by_code:
                    stats["dropped_codes"] += 1
                elif by_code[code] not in ranked:
                    ranked.append(by_code[code])
            kept = {k: v for k, v in row.items() if k not in ("scores", "hits")}  # item-level
            rows.append(json.dumps({**kept, "target": category[item][0], "ranked": ranked}))
            stats["rows"] += 1
        (out_dir / path.name).write_text("\n".join(rows) + "\n")
    print(json.dumps(stats))


if __name__ == "__main__":
    main(*sys.argv[1:])
