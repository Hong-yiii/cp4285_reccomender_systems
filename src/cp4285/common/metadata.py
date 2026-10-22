"""Item metadata: category audit, subcategory groups and a text classifier for missing paths.

Amazon Reviews 2023 item files give each item a hierarchical `categories` path, but many paths
are empty, missing or too shallow. Items are grouped by the path prefix at a fixed `level`
(2 = root and one subcategory). An item without a path that deep gets a predicted subcategory
from a TF-IDF text classifier. The classifier is trained on items of the same
category file whose path is observed (or of a donor file, see DONORS), and only labels that
extend the item's own shallow path are allowed. Predictions below a confidence threshold, chosen on held-out labelled items for a
target precision, stay `unknown`. Every output row records whether its label was observed or
predicted.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from tempfile import mkdtemp

import numpy as np

from .data import CATEGORIES, connection, fetch
from .utils import save_json

BASE = "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/meta_categories"
SOURCE = "https://amazon-reviews-2023.github.io/"
SEP = " > "
# Only the fields used for grouping and item text; images, videos and prices are dropped.
COLUMNS = {
    "parent_asin": "VARCHAR",
    "main_category": "VARCHAR",
    "title": "VARCHAR",
    "categories": "VARCHAR[]",
    "features": "VARCHAR[]",
    "description": "VARCHAR[]",
    "store": "VARCHAR",
    "details": "JSON",
}
# Files whose `categories` are (nearly) all empty learn labels from a related file instead;
# Unknown and Subscription_Boxes have no natural donor, so they use every labelled file.
DONORS = {
    "All_Beauty": ("Beauty_and_Personal_Care",),
    "Amazon_Fashion": ("Clothing_Shoes_and_Jewelry",),
    "Digital_Music": ("CDs_and_Vinyl",),
    "Health_and_Personal_Care": ("Health_and_Household",),
    "Subscription_Boxes": CATEGORIES,
    "Unknown": CATEGORIES,
}
MODEL = "hashed TF-IDF word 1-2 grams, SGD logistic regression"
TEXT = """left(concat_ws(' | ', nullif(trim(title), ''), nullif(trim(brand), ''),
    nullif(array_to_string(features, ' '), ''), nullif(array_to_string(description, ' '), '')),
    2000)"""


def raw_path(directory, category):
    return Path(directory) / "raw" / f"meta_{category}.jsonl.gz"


def items_path(directory, category):
    return Path(directory) / "items" / f"{category}.parquet"


def download(category, directory):
    if category not in CATEGORIES:
        raise ValueError(f"Choose one of {CATEGORIES}")

    def check(f):
        first = json.loads(f.readline() or "null")
        if not isinstance(first, dict) or "parent_asin" not in first:
            raise ValueError("Downloaded file is not Amazon 2023 item metadata")

    return fetch(f"{BASE}/meta_{category}.jsonl.gz", raw_path(directory, category), check)


def publish(dest, write):
    """Write a file or directory beside `dest` and move it into place only when complete."""
    dest = Path(dest)
    if os.path.lexists(dest):
        raise FileExistsError(f"Refusing to overwrite artifact: {dest}; choose a new path")
    dest.parent.mkdir(parents=True, exist_ok=True)
    scratch = Path(mkdtemp(dir=dest.parent, prefix=f".{dest.name}-"))
    try:
        write(scratch / dest.name)
        os.rename(scratch / dest.name, dest)
    finally:
        shutil.rmtree(scratch)


def quote(path):
    return "'" + str(path).replace("'", "''") + "'"


def convert(category, directory):
    """Project one raw metadata file to the parquet columns the later stages read."""
    source = raw_path(directory, category)
    if not source.is_file():
        raise FileNotFoundError(f"Missing metadata file: {source}; run meta-download")
    with connection() as con:
        con.read_json(
            str(source.resolve()), columns=COLUMNS, format="newline_delimited"
        ).create_view("raw")
        rows = con.execute("SELECT count(*) FROM raw").fetchall()[0][0]
        relation = con.sql(
            """SELECT ?::VARCHAR AS domain, parent_asin, main_category, title, categories,
            coalesce(features, []) AS features, coalesce(description, []) AS description, store,
            details->>'Brand' AS brand FROM raw""",
            params=[category],
        )
        publish(items_path(directory, category), lambda path: relation.write_parquet(str(path)))
    return {"category": category, "rows": rows}


def items(con, directory, level):
    """Views over all converted files; one row per (domain, parent_asin), deepest path first."""
    pattern = Path(directory).resolve() / "items" / "*.parquet"
    if not list(pattern.parent.glob(pattern.name)):
        raise FileNotFoundError(f"No converted metadata under {pattern.parent}; run meta-download")
    con.read_parquet(str(pattern)).create_view("items")
    con.execute(f"""CREATE OR REPLACE VIEW labelled AS
        WITH deduped AS (SELECT * FROM items QUALIFY row_number() OVER (
            PARTITION BY domain, parent_asin
            ORDER BY len(coalesce(categories, [])) DESC, array_to_string(categories, '{SEP}'),
            title) = 1)
        SELECT domain, parent_asin, categories,
            array_to_string(coalesce(categories, []), '{SEP}') AS full_path,
            CASE WHEN len(coalesce(categories, [])) >= {int(level)}
                THEN array_to_string(categories[1:{int(level)}], '{SEP}') END AS observed,
            {TEXT} AS text
        FROM deduped""")


def audit(directory, output, level=2):
    """Per-domain category coverage plus the full category tree as a parquet table."""
    with connection() as con:
        items(con, directory, level)
        rows = con.execute(
            f"""SELECT domain, count(*), count(DISTINCT parent_asin),
            count(*) FILTER (WHERE categories IS NULL),
            count(*) FILTER (WHERE len(categories) = 0),
            count(*) FILTER (WHERE main_category IS NULL OR trim(main_category) = ''),
            count(*) FILTER (WHERE len(categories) BETWEEN 1 AND {int(level) - 1}),
            count(*) FILTER (WHERE len(categories) >= {int(level)}),
            count(*) FILTER (WHERE {TEXT} = '')
            FROM items GROUP BY domain ORDER BY domain"""
        ).fetchall()
        keys = [
            "rows",
            "unique_parent_asin",
            "categories_missing",
            "categories_empty",
            "main_category_missing",
            "categories_shallower_than_level",
            "categories_at_level",
            "no_text",
        ]
        domains = {row[0]: dict(zip(keys, row[1:])) for row in rows}
        for domain, depth, n in con.execute(
            """SELECT domain, len(categories), count(*) FROM items
            GROUP BY ALL ORDER BY ALL"""
        ).fetchall():
            domains[domain].setdefault("depth_counts", {})[str(depth)] = n
        for domain, root, n in con.execute(
            """SELECT domain, categories[1], count(*) n FROM items WHERE len(categories) > 0
            GROUP BY 1, 2 QUALIFY row_number() OVER (PARTITION BY domain ORDER BY n DESC) <= 5
            ORDER BY domain, n DESC"""
        ).fetchall():
            domains[domain].setdefault("top_roots", {})[root] = n
        tree = con.sql(f"""SELECT domain, l AS level,
            array_to_string(categories[1:l], '{SEP}') AS path, count(*) AS n_items
            FROM items, range(1, len(categories) + 1) t(l)
            GROUP BY ALL ORDER BY domain, path""")
        publish(Path(output) / "category_tree.parquet", lambda p: tree.write_parquet(str(p)))
    result = {
        "scope": "Amazon Reviews 2023 item metadata files as downloaded; rows before dedup",
        "level": level,
        "separator": SEP,
        "domains": domains,
    }
    save_json(Path(output) / "audit.json", result)
    return result


def fit(texts, labels, n_classes, seed=0, features=2**18):
    """Hashed word and word-pair TF-IDF with a logistic-loss linear model.

    Returns a function mapping texts (and an optional allowed-class mask) to class probabilities.
    """
    from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
    from sklearn.linear_model import SGDClassifier

    hashing = HashingVectorizer(
        n_features=features, ngram_range=(1, 2), alternate_sign=False, norm=None
    )
    tfidf = TfidfTransformer(sublinear_tf=True)
    linear = SGDClassifier(
        loss="log_loss", alpha=1e-6, max_iter=20, tol=None, random_state=seed, n_jobs=-1
    ).fit(tfidf.fit_transform(hashing.transform(texts)), labels)

    def predict(texts, mask=None):
        probs = np.zeros((len(texts), n_classes))
        probs[:, linear.classes_] = linear.predict_proba(tfidf.transform(hashing.transform(texts)))
        if mask is not None:
            probs *= mask
            total = probs.sum(axis=1, keepdims=True)
            probs = np.divide(probs, total, out=np.zeros_like(probs), where=total > 0)
        return probs

    return predict


def write_part(con, path, asin, label, confidence):
    """One parquet part of predictions; a missing label is NULL."""
    # DuckDB sniffs object columns from their first rows and fails when those are all None,
    # so labels travel as strings with "" for none.
    label = np.array(["" if name is None else name for name in label], dtype=object)
    con.register("part", {"parent_asin": asin, "subcategory": label, "confidence": confidence})
    con.execute(f"""COPY (SELECT parent_asin, nullif(subcategory, '') AS subcategory, confidence
        FROM part) TO {quote(path)} (FORMAT parquet)""")
    con.unregister("part")


def threshold(confidence, correct, precision):
    """Lowest confidence whose predictions at or above it reach the target precision."""
    order = np.argsort(-confidence, kind="stable")
    hits = np.cumsum(correct[order]) / np.arange(1, len(order) + 1)
    # Ties share a threshold, so only positions where the next confidence is lower can cut.
    cut = np.r_[confidence[order][1:] < confidence[order][:-1], True]
    good = np.flatnonzero((hits >= precision) & cut)
    return float(confidence[order][good[-1]]) if len(good) else float("inf")


def macro_f1(truth, guess, n_classes):
    scores = []
    for k in range(n_classes):
        tp = np.sum((guess == k) & (truth == k))
        denominator = np.sum(guess == k) + np.sum(truth == k)
        if np.sum(truth == k):
            scores.append(2 * tp / denominator)
    return float(np.mean(scores))


def classify(
    directory,
    output,
    category,
    level=2,
    min_items=200,
    sample=200_000,
    precision=0.8,
    seed=0,
    chunk=65_536,
    train_on=None,
):
    """Train on observed paths of one domain (or its donors), label its unlabelled items."""
    train_on = list(train_on or DONORS.get(category, (category,)))
    dest = Path(output) / "predictions" / category
    metrics_path = Path(output) / "metrics" / f"{category}.json"
    for path in (dest, metrics_path):
        if os.path.lexists(path):
            raise FileExistsError(f"Refusing to overwrite artifact: {path}; choose a new path")
    with connection() as con:
        items(con, directory, level)
        con.register("donors", {"domain": np.array(train_on, dtype=object)})
        classes = [
            row[0]
            for row in con.execute(
                """SELECT observed FROM labelled JOIN donors USING (domain)
                WHERE observed IS NOT NULL GROUP BY observed HAVING count(*) >= ?
                ORDER BY observed""",
                [min_items],
            ).fetchall()
        ]
        metrics = {
            "category": category,
            "model": MODEL,
            "level": level,
            "min_items": min_items,
            "target_precision": precision,
            "seed": seed,
            "train_on": train_on if len(train_on) < len(CATEGORIES) else "all",
            "classes": len(classes),
        }
        model = None
        if len(classes) >= 2:
            # Uniform sample of labelled items, so held-out scores reflect the real label mix.
            con.register("classes", {"observed": np.array(classes, dtype=object)})
            rows = con.execute(
                """SELECT text, observed, hash(parent_asin || ?) % 5 = 0 FROM labelled
                JOIN classes USING (observed) JOIN donors USING (domain) WHERE text <> ''
                ORDER BY hash(parent_asin || ?), domain, parent_asin LIMIT ?""",
                [str(seed), str(seed + 1), sample],
            ).fetchall()
            index = {name: k for k, name in enumerate(classes)}
            labels = np.array([index[row[1]] for row in rows])
            held = np.array([row[2] for row in rows], dtype=bool)
            texts = np.array([row[0] for row in rows], dtype=object)
            if held.all() or not held.any():
                raise ValueError(f"{category}: too few labelled items to train and hold out")
            model = fit(texts[~held], labels[~held], len(classes), seed)
            probs = model(texts[held])
            guess, confidence = probs.argmax(axis=1), probs.max(axis=1)
            correct = guess == labels[held]
            cutoff = threshold(confidence, correct, precision)
            kept = confidence >= cutoff
            metrics.update(
                train=int((~held).sum()),
                holdout=int(held.sum()),
                holdout_accuracy=float(correct.mean()),
                holdout_macro_f1=macro_f1(labels[held], guess, len(classes)),
                # JSON has no infinity: None means no threshold reached the target precision.
                confidence_threshold=None if np.isinf(cutoff) else cutoff,
                holdout_coverage_at_threshold=float(kept.mean()),
                holdout_precision_at_threshold=float(correct[kept].mean()) if kept.any() else None,
            )
            # Shallow paths may only extend to classes below them.
            prefixes = {}

            def mask(prefix):
                if prefix not in prefixes:
                    prefixes[prefix] = np.array(
                        [not prefix or name.startswith(prefix + SEP) for name in classes]
                    )
                return prefixes[prefix]

        counts = {"pending": 0, "predicted": 0, "unknown": 0, "no_text": 0}
        reader = con.cursor()
        reader.execute(
            f"""SELECT parent_asin, text, array_to_string(coalesce(categories, []), '{SEP}')
            FROM labelled WHERE domain = ? AND observed IS NULL ORDER BY parent_asin""",
            [category],
        )

        def write(path):
            path.mkdir()
            part = 0
            while batch := reader.fetchmany(chunk):
                asin, text, prefix = (np.array(column, dtype=object) for column in zip(*batch))
                label = np.full(len(batch), None, dtype=object)
                confidence = np.full(len(batch), np.nan, dtype=np.float32)
                has_text = text != ""
                counts["pending"] += len(batch)
                counts["no_text"] += int((~has_text).sum())
                if model is not None and has_text.any():
                    allowed = np.stack([mask(p) for p in prefix[has_text]])
                    usable = allowed.any(axis=1)
                    rows = np.flatnonzero(has_text)[usable]
                    probs = model(text[rows], allowed[usable])
                    best = probs.max(axis=1)
                    confident = best >= cutoff
                    label[rows[confident]] = np.array(classes, dtype=object)[
                        probs.argmax(axis=1)[confident]
                    ]
                    confidence[rows] = best
                counts["predicted"] += sum(name is not None for name in label)
                write_part(con, path / f"part-{part:05d}.parquet", asin, label, confidence)
                part += 1

        publish(dest, write)
    counts["unknown"] = counts["pending"] - counts["predicted"]
    metrics.update(counts)
    save_json(metrics_path, metrics)
    return metrics


def group(directory, output, level=2, ratings=None):
    """Join observed and predicted labels into one partitioned table plus a density summary."""
    output = Path(output)
    with connection() as con:
        items(con, directory, level)
        con.execute(
            "CREATE TABLE predictions (domain VARCHAR, parent_asin VARCHAR, subcategory VARCHAR, "
            "confidence FLOAT)"
        )
        classified = []
        for path in sorted((output / "metrics").glob("*.json")):
            metrics = json.loads(path.read_text())
            if metrics["level"] != level:
                raise ValueError(f"{path} used level {metrics['level']}, not {level}")
            parts = output / "predictions" / metrics["category"] / "*.parquet"
            if list(parts.parent.glob(parts.name)):
                con.read_parquet(str(parts)).create_view("incoming")
                con.execute(
                    "INSERT INTO predictions SELECT ?, parent_asin, subcategory, confidence "
                    "FROM incoming",
                    [metrics["category"]],
                )
            classified.append(metrics["category"])
        con.register("classified", {"domain": np.array(classified or [""], dtype=object)})
        con.execute("""CREATE TABLE grouped AS SELECT l.domain, l.parent_asin,
            coalesce(l.observed, p.subcategory) AS subcategory, l.full_path,
            CASE WHEN l.observed IS NOT NULL THEN 'observed'
                WHEN p.subcategory IS NOT NULL THEN 'predicted'
                WHEN l.domain IN (SELECT domain FROM classified) THEN 'unknown'
                ELSE 'unclassified' END AS label_source,
            CASE WHEN l.observed IS NOT NULL THEN 1.0 ELSE p.confidence END AS confidence
            FROM labelled l LEFT JOIN predictions p USING (domain, parent_asin)""")
        publish(
            output / "item_categories",
            lambda path: con.execute(
                f"""COPY (SELECT * FROM grouped ORDER BY domain, subcategory, parent_asin)
                TO {quote(path)} (FORMAT parquet, PARTITION_BY (domain),
                WRITE_PARTITION_COLUMNS true)"""
            ),
        )
        con.execute("CREATE TABLE ratings (domain VARCHAR, user_id VARCHAR, parent_asin VARCHAR)")
        rated = []
        for category in CATEGORIES:
            path = Path(ratings or "") / f"{category}.csv.gz"
            if ratings and path.is_file():
                con.read_csv(str(path.resolve()), header=True, all_varchar=True).create_view(
                    "incoming"
                )
                con.execute(
                    "INSERT INTO ratings SELECT ?, user_id, parent_asin FROM incoming", [category]
                )
                rated.append(category)
        summary = con.sql("""WITH counts AS (SELECT domain, subcategory, count(*) AS items,
                count(*) FILTER (WHERE label_source = 'observed') AS observed,
                count(*) FILTER (WHERE label_source = 'predicted') AS predicted,
                count(*) FILTER (WHERE label_source IN ('unknown', 'unclassified')) AS unlabelled
                FROM grouped GROUP BY ALL),
            rated AS (SELECT g.domain, g.subcategory, count(DISTINCT r.parent_asin) AS rated_items,
                count(DISTINCT r.user_id) AS users, count(*) AS ratings
                FROM ratings r JOIN grouped g USING (domain, parent_asin) GROUP BY ALL)
            SELECT c.*, r.rated_items, r.users, r.ratings,
                r.ratings / (r.users * r.rated_items) AS density
            FROM counts c LEFT JOIN rated r ON c.domain = r.domain
                AND c.subcategory IS NOT DISTINCT FROM r.subcategory
            ORDER BY c.domain, c.items DESC""")
        publish(output / "subcategory_summary.parquet", lambda p: summary.write_parquet(str(p)))
        totals = dict(
            con.execute("SELECT label_source, count(*) FROM grouped GROUP BY ALL").fetchall()
        )
    return {"level": level, "rated_domains": rated, "label_sources": totals}


CARD = """---
license: other
configs:
- config_name: item_categories
  data_files: item_categories/*/*.parquet
  default: true
- config_name: subcategory_summary
  data_files: subcategory_summary.parquet
- config_name: category_tree
  data_files: category_tree.parquet
---

# Amazon Reviews 2023 item subcategories

Derived from the item metadata of [Amazon Reviews 2023]({source}) (Hou et al., 2024).
Only `parent_asin` keys and category labels are included; titles, descriptions and
reviews remain at the original source and its terms.

- `subcategory` is the first {level} levels of each item's `categories` path, joined with
  `"{sep}"`.
- `label_source = observed` comes from the metadata. `predicted` comes from a
  text classifier ({model}) trained on observed items of the same category file
  (files without paths use a donor file, see `train_on` in `metrics/`), kept only above a confidence threshold chosen on held-out items for
  {precision:.0%} precision. `unknown` items fell below it or had no text.
- `subcategory_summary` joins groups to the official 5-core rating files where available
  (`density = ratings / (users * rated_items)`).

Label sources: {sources}

## Classifier hold-out metrics

| category | classes | accuracy | macro-F1 | threshold | coverage |
|---|---:|---:|---:|---:|---:|
{rows}
"""


def card(output, level, totals):
    rows, model, precision = [], None, None
    for path in sorted((Path(output) / "metrics").glob("*.json")):
        m = json.loads(path.read_text())
        model, precision = m["model"], m["target_precision"]
        if "holdout_accuracy" in m:
            rows.append(
                f"| {m['category']} | {m['classes']} | {m['holdout_accuracy']:.3f} "
                f"| {m['holdout_macro_f1']:.3f} | {m['confidence_threshold'] or float('inf'):.3f} "
                f"| {m['holdout_coverage_at_threshold']:.3f} |"
            )
        else:
            rows.append(f"| {m['category']} | {m['classes']} | – | – | – | – |")
    return CARD.format(
        source=SOURCE,
        level=level,
        sep=SEP,
        model=model,
        precision=precision or 0,
        sources=", ".join(f"{k}: {v:,}" for k, v in sorted(totals.items())),
        rows="\n".join(rows),
    )


def upload(output, repo, level, totals):
    """Push the derived tables (never raw metadata) to a private Hugging Face dataset repo."""
    from huggingface_hub import HfApi

    output = Path(output)
    readme = output / "README.md"
    if not readme.exists():
        readme.write_text(card(output, level, totals))
    api = HfApi()
    api.create_repo(repo, repo_type="dataset", private=True, exist_ok=True)
    return api.upload_folder(
        folder_path=str(output),
        repo_id=repo,
        repo_type="dataset",
        allow_patterns=[
            "README.md",
            "audit.json",
            "category_tree.parquet",
            "subcategory_summary.parquet",
            "item_categories/**",
            "metrics/*.json",
        ],
        commit_message="Publish Amazon 2023 item subcategories",
    )
