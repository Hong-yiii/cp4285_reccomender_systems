"""Offline checks for item-metadata grouping, using synthetic items and a fake embedder."""

import csv
import gzip
import json
from contextlib import nullcontext

import duckdb
import httpx
import numpy as np
import pytest

from cp4285.experiment import metadata

WORDS = {"Hair": 0, "Skin": 1, "Nails": 2}


def item(asin, categories=..., title="", **extra):
    row = {"parent_asin": asin, "title": title, "features": [], "description": [], **extra}
    if categories is not ...:
        row["categories"] = categories
    row["details"] = {"Brand": "Acme", "Size": "3"}
    row["images"] = [{"large": "x"}]
    return row


def synthetic():
    """Three learnable subcategories plus missing, empty, shallow and text-less items."""
    rows = []
    for k, sub in enumerate(WORDS):
        for n in range(12):
            rows.append(item(f"{sub}-{n}", ["Beauty", sub, "Leaf"], title=f"{sub.lower()} thing"))
    rows += [
        item("missing", title="hair thing"),
        item("empty", [], title="skin thing"),
        item("shallow", ["Beauty"], title="nails thing"),
        item("wrong-root", ["Garden"], title="hair thing"),
        item("blank", []),
        item("Hair-0", ["Beauty"], title="duplicate with a shallower path"),
    ]
    return rows


@pytest.fixture
def converted(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # DuckDB spill directory is relative to the working directory
    raw = metadata.raw_path(tmp_path, "Appliances")
    raw.parent.mkdir(parents=True)
    with gzip.open(raw, "wt") as f:
        for row in synthetic():
            f.write(json.dumps(row) + "\n")
    metadata.convert("Appliances", tmp_path)
    return tmp_path


def test_convert_keeps_missing_distinct_from_empty_and_refuses_overwrite(converted):
    rows = dict(
        duckdb.sql(
            f"SELECT parent_asin, categories FROM '{metadata.items_path(converted, 'Appliances')}'"
        ).fetchall()
    )
    assert rows["missing"] is None and rows["empty"] == []
    with pytest.raises(FileExistsError):
        metadata.convert("Appliances", converted)


def test_audit_counts_and_tree(converted):
    result = metadata.audit(converted, converted / "out")["domains"]["Appliances"]
    assert result["rows"] == 42 and result["unique_parent_asin"] == 41
    assert result["categories_missing"] == 1 and result["categories_empty"] == 2
    assert result["categories_shallower_than_level"] == 3
    assert result["no_text"] == 0  # Brand counts as text.
    tree = dict(
        duckdb.sql(
            f"SELECT path, n_items FROM '{converted / 'out' / 'category_tree.parquet'}'"
        ).fetchall()
    )
    assert tree["Beauty > Hair"] == 12 and tree["Beauty"] == 38 and tree["Garden"] == 1


def test_classify_and_group_keep_every_item_once(converted):
    out = converted / "out"
    metrics = metadata.classify(converted, out, "Appliances", min_items=5, precision=0.9, seed=3)
    assert metrics["classes"] == 3 and metrics["holdout_accuracy"] == 1.0
    result = metadata.group(converted, out)
    rows = {
        r[0]: r[1:]
        for r in duckdb.sql(
            f"SELECT parent_asin, subcategory, label_source, domain "
            f"FROM '{out / 'item_categories' / '*' / '*.parquet'}'"
        ).fetchall()
    }
    assert len(rows) == 41 and sum(result["label_sources"].values()) == 41
    assert rows["Hair-0"][:2] == ("Beauty > Hair", "observed")
    assert rows["missing"][:2] == ("Beauty > Hair", "predicted")
    assert rows["empty"][:2] == ("Beauty > Skin", "predicted")
    assert rows["shallow"][:2] == ("Beauty > Nails", "predicted")
    # No class extends "Garden", so the classifier must not move it into Beauty.
    assert rows["wrong-root"][:2] == (None, "unknown")
    assert rows["Skin-1"][2] == "Appliances"
    with pytest.raises(FileExistsError):
        metadata.group(converted, out)


def test_group_marks_unclassified_domains_and_joins_ratings(converted):
    ratings = converted / "ratings"
    ratings.mkdir()
    with gzip.open(ratings / "Appliances.csv.gz", "wt", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["user_id", "parent_asin", "rating", "timestamp"])
        writer.writerows([["u1", "Hair-1", 5, 1], ["u2", "Hair-1", 4, 2], ["u1", "Hair-2", 5, 3]])
    result = metadata.group(converted, converted / "out", ratings=ratings)
    assert result["rated_domains"] == ["Appliances"]
    assert result["label_sources"] == {"observed": 36, "unclassified": 5}
    hair = duckdb.sql(
        f"SELECT users, rated_items, ratings, density FROM "
        f"'{converted / 'out' / 'subcategory_summary.parquet'}' "
        f"WHERE subcategory = 'Beauty > Hair'"
    ).fetchall()
    assert hair == [(2, 2, 3, 0.75)]


def test_threshold_reaches_target_precision_or_rejects_everything():
    confidence = np.array([0.9, 0.8, 0.8, 0.6, 0.5])
    correct = np.array([True, True, False, True, False])
    assert metadata.threshold(confidence, correct, 0.75) == 0.6
    assert metadata.threshold(confidence, correct, 1.0) == 0.9
    assert metadata.threshold(confidence, ~correct | True, 1.0) == 0.5
    assert np.isinf(metadata.threshold(confidence, np.zeros(5, dtype=bool), 0.5))


def test_download_rejects_non_metadata_without_publishing(tmp_path, monkeypatch):
    response = httpx.Response(
        200,
        content=gzip.compress(b"user_id,parent_asin\n"),
        request=httpx.Request("GET", metadata.BASE),
    )
    monkeypatch.setattr(httpx, "stream", lambda *a, **kw: nullcontext(response))
    with pytest.raises(ValueError):
        metadata.download("Appliances", tmp_path)
    assert list((tmp_path / "raw").iterdir()) == []


def test_empty_file_borrows_labels_from_its_donor(converted):
    raw = metadata.raw_path(converted, "Unknown")
    with gzip.open(raw, "wt") as f:
        f.write(json.dumps(item("stray", [], title="nails thing")) + "\n")
    metadata.convert("Unknown", converted)
    out = converted / "out"
    metrics = metadata.classify(converted, out, "Unknown", min_items=5, precision=0.9)
    assert metrics["train_on"] == "all" and metrics["predicted"] == 1
    [(label,)] = duckdb.sql(
        f"SELECT subcategory FROM '{out / 'predictions' / 'Unknown' / '*.parquet'}'"
    ).fetchall()
    assert label == "Beauty > Nails"


def test_prediction_parts_survive_long_runs_of_missing_labels(tmp_path):
    n = 70_000  # Longer than DuckDB's type-sniffing sample, all None at the start.
    label = np.array([None] * (n - 1) + ["Beauty > Hair"], dtype=object)
    asin = np.array([f"i{k}" for k in range(n)], dtype=object)
    path = tmp_path / "part.parquet"
    metadata.write_part(duckdb.connect(), path, asin, label, np.full(n, np.nan, dtype=np.float32))
    assert duckdb.sql(f"SELECT count(*), count(subcategory) FROM '{path}'").fetchall() == [(n, 1)]
