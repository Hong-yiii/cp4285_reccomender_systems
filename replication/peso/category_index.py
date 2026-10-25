"""Category-target copy of PESO's Instruments data, from the PR 14 item subcategories.

Each PESO item id is matched to its Amazon Reviews 2023 parent_asin by exact title and
full category path, then given PR 14's level-2 subcategory: observed from the metadata,
or predicted by PR 14's text classifier for items without a path. When several items
share a title and path but carry different predictions, the most confident one wins.
The new index gives every item a single token for its subcategory, so upstream trains,
decodes and scores categories without code changes. Users, blocks and their order stay
identical to Instruments_cl_ori; only what each event is called changes.

    python category_index.py <Instruments_cl_ori dir> <PR 14 Musical_Instruments.parquet> \
        <PR 14 item_categories/domain=Musical_Instruments dir> <new dataset dir> <new stats.json> \
        [<level> <ancestor label dir> ...]

Level 2 is PR 14's standard (Instruments_cl_cat). Deeper levels (`cp4285 meta-classify` and
`meta-group --level N`) build Instruments_cl_cat<N>; an item without a label at that depth
takes the first label found in the ancestor label dirs, listed nearest first.

The parquet comes from `cp4285 meta-download --category Musical_Instruments`; the labels
from PR 14's published dataset (randomwish/amazon23-item-categories). Needs duckdb.
"""

import json
import shutil
import sys
from collections import Counter
from pathlib import Path

import duckdb

SOURCE = "Instruments_cl_ori"
DATASET = "Instruments_cl_cat"
INDEX = ".index.cat2.json"
BLOCKS = range(5)


def labels(peso_dir, meta, label_dir):
    """PESO item id -> (subcategory or None, label_source), plus join diagnostics."""
    items = json.loads((Path(peso_dir) / f"{SOURCE}.item.json").read_text())
    con = duckdb.connect()
    con.execute("CREATE TABLE p (id VARCHAR, title VARCHAR, path VARCHAR)")
    con.executemany(
        "INSERT INTO p VALUES (?, ?, ?)",
        [(k, v["title"], " > ".join(v["categories"])) for k, v in items.items()],
    )
    con.read_parquet(str(meta)).create_view("meta")
    con.read_parquet(str(Path(label_dir) / "*.parquet")).create_view("labels")
    rows = con.execute("""
        SELECT p.id, count(DISTINCT m.parent_asin) AS matches,
            count(DISTINCT l.subcategory) AS distinct_labels,
            arg_max(l.subcategory, (l.confidence, l.subcategory)) AS subcategory,
            arg_max(l.label_source, (l.confidence, l.subcategory)) AS label_source
        FROM p
        LEFT JOIN meta m
            ON m.title = p.title
            AND array_to_string(coalesce(m.categories, []), ' > ') = p.path
        LEFT JOIN labels l ON l.parent_asin = m.parent_asin AND l.subcategory IS NOT NULL
        GROUP BY p.id""").fetchall()
    join = {
        "items": len(rows),
        "single_match": sum(r[1] == 1 for r in rows),
        "several_matches": sum(r[1] > 1 for r in rows),
        "conflicting_labels_resolved_by_confidence": sum(r[2] > 1 for r in rows),
        "label_source": dict(Counter(r[4] for r in rows)),
    }
    return {r[0]: (r[3], r[4]) for r in rows}, join


def block_stats(users, category):
    """One block's new events at item and category level, and its test targets."""
    events = [i for u in users.values() for i in u["current_block"]]
    active = sum(bool(u["current_block"]) for u in users.values())
    items, cats = len(set(events)), len({category[i] for i in events})
    tested = [u["current_block"][-1] for u in users.values() if len(u["current_block"]) >= 3]
    top, top_n = Counter(category[i] for i in tested).most_common(1)[0]
    return {
        "users": active, "events": len(events), "items": items, "categories": cats,
        "item_density": len(events) / (active * items),
        "category_density": len(events) / (active * cats),
        "test_users": len(tested), "most_common_test_category": top,
        "most_common_test_category_share": top_n / len(tested),
    }  # fmt: skip


def main(peso_dir, meta, label_dir, out_dir, stats_path, level="2", *ancestors):
    peso_dir, out_dir, stats_path = Path(peso_dir), Path(out_dir), Path(stats_path)
    if stats_path.exists():
        raise FileExistsError(f"Refusing to overwrite {stats_path}")
    level = int(level)
    dataset = DATASET if level == 2 else f"{DATASET}{level}"
    index_file = INDEX if level == 2 else f".index.cat{level}.json"
    by_item, join = labels(peso_dir, meta, label_dir)
    for ancestor in ancestors:  # nearest first; an item keeps the first label it finds
        found, _ = labels(peso_dir, meta, ancestor)
        by_item = {
            i: v
            if v[0] is not None or found[i][0] is None
            else (found[i][0], f"ancestor {found[i][1]}")
            for i, v in by_item.items()
        }
    missing = [i for i, (name, _) in by_item.items() if name is None]
    if missing:
        raise ValueError(f"{len(missing)} PESO items have no PR 14 label, e.g. {missing[:5]}")
    join["label_source"] = dict(Counter(source for _, source in by_item.values()))
    names = sorted({name for name, _ in by_item.values()})
    token = {name: f"<cat_{k}>" for k, name in enumerate(names)}
    category = {i: token[name] for i, (name, _) in by_item.items()}
    out_dir.mkdir(parents=True, exist_ok=False)
    for b in BLOCKS:
        name = f"all.block_{b}.json"
        shutil.copyfile(peso_dir / f"{SOURCE}.{name}", out_dir / f"{dataset}.{name}")
    shutil.copyfile(peso_dir / f"{SOURCE}.item.json", out_dir / f"{dataset}.item.json")
    index = {i: [category[i]] for i in sorted(category, key=int)}
    (out_dir / f"{dataset}{index_file}").write_text(json.dumps(index))
    sources = {i: by_item[i][1] for i in index}  # category_report.py splits targets by this
    (out_dir / f"{dataset}.label_source.json").write_text(json.dumps(sources))
    events = Counter()
    blocks = {}
    for b in BLOCKS:
        users = json.loads((peso_dir / f"{SOURCE}.all.block_{b}.json").read_text())
        blocks[f"block_{b}"] = block_stats(users, category)
        events.update(by_item[i][1] for u in users.values() for i in u["current_block"])
    items_per = Counter(name for name, _ in by_item.values())
    stats = {
        "source": SOURCE, "dataset": dataset, "index_file": index_file, "level": level, "join": join,
        "events_by_label_source": dict(events),
        "categories": {
            token[n]: {"subcategory": n, "items": items_per[n]} for n in names
        },
        "blocks": blocks,
    }  # fmt: skip
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    with stats_path.open("x") as f:
        json.dump(stats, f, indent=2)
    return stats


if __name__ == "__main__":
    report = main(*sys.argv[1:])
    print(json.dumps({k: report[k] for k in ("join", "events_by_label_source")}, indent=2))
