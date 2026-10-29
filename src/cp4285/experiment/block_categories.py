"""Category labels for any PESO-format dataset, by PR 14's rule, from its own item paths.

PESO's released datasets carry each item's Amazon `categories` path in `<name>.item.json`.
An item's level-N label is the first N levels of that path, the observed label PR 14 takes
from Amazon 2023 metadata; a shorter path keeps its deepest level ("ancestor"). Items with
no path get a prediction from PR 14's text classifier (experiment/metadata.py `fit`: hashed
TF-IDF of title and description, logistic loss), trained on this dataset's labelled items
whose label has at least `min_items` items. A 10% holdout of labelled items measures its
accuracy before the final fit uses all of them.

Writes, in a new folder, the same files category_index.py does for the PESO study, so
`cp4285 adapter-blocks` reads every dataset alike: `<name>.index.cat<N>.json` (item ->
["<cat_k>"], categories ordered by name), `<name>.label_source.json` and
`category_stats.json` (names, label sources, classifier accuracy and per-block statistics).
Needs the `meta` extra (scikit-learn).
"""

import json
import random
from collections import Counter
from pathlib import Path

import numpy as np

from .metadata import fit
from .utils import save_json

BLOCKS = range(5)


def add_arguments(parser):
    parser.add_argument("--data", required=True, help="PESO-format dataset folder")
    parser.add_argument("--output", required=True, help="New folder for the category files")
    parser.add_argument("--level", type=int, default=2)
    parser.add_argument(
        "--min-items", type=int, default=20, help="Smallest class the classifier learns"
    )
    parser.add_argument("--seed", type=int, default=0)


def text(item):
    return f"{item.get('title') or ''} {(item.get('description') or '')[:2000]}"


def labels_for(items, level, min_items, seed):
    """item -> (label, source), and the classifier's report."""
    out = {}
    for key, item in items.items():
        path = item.get("categories") or []
        if len(path) >= level:
            out[key] = (" > ".join(path[:level]), "observed")
        elif path:
            out[key] = (" > ".join(path), "ancestor observed")
    counts = Counter(label for label, _ in out.values())
    classes = sorted(label for label, n in counts.items() if n >= min_items)
    index = {label: k for k, label in enumerate(classes)}
    known = [k for k in sorted(out, key=int) if out[k][0] in index]
    missing = [k for k in sorted(items, key=int) if k not in out]
    report = {"classes": len(classes), "min_items": min_items, "labelled": len(out),
              "unlabelled": len(missing)}  # fmt: skip
    if missing and classes:
        rng = random.Random(seed)
        held = set(rng.sample(known, len(known) // 10))
        train = [k for k in known if k not in held]
        y = np.array([index[out[k][0]] for k in train])
        predict = fit([text(items[k]) for k in train], y, len(classes), seed=seed)
        test = sorted(held, key=int)
        guess = predict([text(items[k]) for k in test]).argmax(1)
        truth = np.array([index[out[k][0]] for k in test])
        report["holdout_accuracy"] = float((guess == truth).mean())
        report["holdout_items"] = len(test)
        predict = fit([text(items[k]) for k in known],
                      np.array([index[out[k][0]] for k in known]), len(classes), seed=seed)  # fmt: skip
        for key, column in zip(missing, predict([text(items[k]) for k in missing]).argmax(1)):
            out[key] = (classes[column], "predicted")
    return out, report


def run(args):
    data, output = Path(args.data), Path(args.output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite: {output}")
    name = next(data.glob("*.item.json")).name.removesuffix(".item.json")
    items = json.loads((data / f"{name}.item.json").read_text())
    labels, classifier = labels_for(items, args.level, args.min_items, args.seed)
    if len(labels) != len(items):
        raise ValueError("Some items have neither a path nor a prediction")
    names = sorted({label for label, _ in labels.values()})
    token = {label: f"<cat_{k}>" for k, label in enumerate(names)}
    index = {k: [token[labels[k][0]]] for k in sorted(items, key=int)}
    sources = {k: labels[k][1] for k in sorted(items, key=int)}
    sizes = Counter(token[label] for label, _ in labels.values())
    blocks = []
    for b in BLOCKS:
        users = json.loads((data / f"{name}.all.block_{b}.json").read_text())
        kept = [u["current_block"] for u in users.values() if len(u["current_block"]) >= 3]
        test = Counter(index[c[-1]][0] for c in kept)
        events = sum(len(u["current_block"]) for u in users.values())
        blocks.append({
            "users": len(users), "test_users": len(kept), "events": events,
            "categories": len({index[i][0] for u in users.values() for i in u["current_block"]}),
            "category_density": events / (len(users) * len(names)),
            "most_common_test_category": test.most_common(1)[0][0],
            "most_common_test_category_share": test.most_common(1)[0][1] / len(kept),
        })  # fmt: skip
    output.mkdir(parents=True)
    level = args.level
    (output / f"{name}.index.cat{level}.json").write_text(json.dumps(index))
    (output / f"{name}.label_source.json").write_text(json.dumps(sources))
    stats = {
        "source": name, "level": level, "rule": "first N levels of the item's categories path",
        "categories": {token[n]: {"subcategory": n, "items": sizes[token[n]]} for n in names},
        "label_source": dict(Counter(sources.values())), "classifier": classifier,
        "blocks": {f"block_{b}": s for b, s in enumerate(blocks)},
    }  # fmt: skip
    save_json(output / "category_stats.json", stats)
    return stats
