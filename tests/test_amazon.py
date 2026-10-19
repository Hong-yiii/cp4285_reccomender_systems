"""Source-specific offline schema fixtures; no Amazon observations or downloads."""

import csv
import gzip
import json
from dataclasses import replace

import pytest

from cp4285.experiment.amazon import AmazonSource


def files(tmp_path, *, jsonl=False):
    paths = {}
    for domain in ("A", "B"):
        path = tmp_path / f"{domain}.{'jsonl.gz' if jsonl else 'csv'}"
        rows = [
            {
                "user_id": f"user-{u}",
                "parent_asin": f"item-{i % 3}",
                "timestamp": 1000 * i + u,
                "rating": 1 + i % 5,
            }
            for u in range(16)
            for i in range(7)
        ]
        if jsonl:
            with gzip.open(path, "wt") as stream:
                stream.writelines(json.dumps(row) + "\n" for row in rows)
        else:
            with path.open("w") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
        paths[domain] = path
    return paths


@pytest.mark.parametrize("jsonl", [False, True])
def test_source_ids_holdouts_catalogues_and_budget(tmp_path, jsonl):
    source = AmazonSource(files(tmp_path, jsonl=jsonl))
    data = source.load_data()
    data.validate()
    assert len(data.initial) == 32
    assert source.audit["A"]["rows_scanned"] == 112
    assert source.audit["selected_user_overlap"] == 16
    assert {e.target.user for e in data.validation.examples}.isdisjoint(
        e.target.user for e in data.tests[0].examples
    )
    held = {e.target.key for t in (data.validation, *data.tests) for e in t.examples}
    for condition in data.conditions:
        for block in condition.blocks:
            assert all(e.key not in held for x in block for e in (*x.history, x.target))
    assert data.initial[0].target.event_id.startswith("row:")
    assert source.vocabularies == {
        "A": ("item-0", "item-1", "item-2"),
        "B": ("item-0", "item-1", "item-2"),
    }
    assert source.load_data() == data  # deterministic prefix hash/preparation
    with pytest.raises(ValueError, match="Insufficient"):
        AmazonSource(source.paths, block_size=100).load_data()
    # A malformed holdout identity is not silently remapped.
    with pytest.raises(ValueError, match="Original"):
        replace(
            data,
            initial=(replace(data.initial[0], target=replace(data.initial[0].target, item="")),),
        ).validate()


def test_ties_unknown_holdouts_and_schema_rejection(tmp_path):
    paths = files(tmp_path)
    for path in paths.values():
        rows = list(csv.DictReader(path.open()))
        rows[5]["parent_asin"] = "heldout-only-item"
        rows.append(dict(rows[0]))  # tied timestamp; first row retained
        with path.open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    source = AmazonSource(paths)
    source.load_data()
    assert source.audit["A"]["timestamp_ties_dropped"] == 1
    assert source.audit["A"]["exclusions"]["unknown_heldout_item"] == 1
    assert "heldout-only-item" not in source.vocabularies["A"]
    paths["A"].write_text("user_id,parent_asin,timestamp\nu,i,1.5\n")
    with pytest.raises(ValueError, match="Invalid Amazon schema"):
        source.load_data()
