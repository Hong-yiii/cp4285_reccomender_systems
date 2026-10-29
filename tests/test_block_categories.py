"""PR 14-rule labels for a PESO-format dataset: observed, ancestor and predicted items."""

import argparse
import json

import pytest

pytest.importorskip("sklearn")

from cp4285.experiment import block_categories

GUITAR = ["Musical Instruments", "Guitars", "Electric"]
DRUMS = ["Musical Instruments", "Drums", "Cymbals"]


def dataset(path):
    path.mkdir()
    items = {}
    for i in range(60):  # two well-separated classes the classifier can learn
        guitar = i % 2 == 0
        words = "guitar string pick fret amp" if guitar else "drum cymbal stick snare kick"
        items[str(i)] = {
            "title": words,
            "description": words,
            "categories": GUITAR if guitar else DRUMS,
        }
    items["60"] = {"title": "guitar fret amp", "description": "", "categories": []}  # predicted
    items["61"] = {"title": "anything", "description": "", "categories": ["Musical Instruments"]}
    (path / "Toy.item.json").write_text(json.dumps(items))
    for b in range(5):
        users = {f"u{u}": {"previous_block": [], "current_block": [str(u), str(u + 2), "60", "61"]}
                 for u in range(4)}  # fmt: skip
        (path / f"Toy.all.block_{b}.json").write_text(json.dumps(users))
    return path


def test_labels_follow_paths_and_fill_gaps_with_the_classifier(tmp_path):
    data = dataset(tmp_path / "data")
    parser = argparse.ArgumentParser()
    block_categories.add_arguments(parser)
    args = parser.parse_args(["--data", str(data), "--output", str(tmp_path / "out")])
    stats = block_categories.run(args)
    names = {k: v["subcategory"] for k, v in stats["categories"].items()}
    assert sorted(names.values()) == ["Musical Instruments", "Musical Instruments > Drums",
                                      "Musical Instruments > Guitars"]  # fmt: skip
    index = json.loads((tmp_path / "out/Toy.index.cat2.json").read_text())
    sources = json.loads((tmp_path / "out/Toy.label_source.json").read_text())
    assert names[index["0"][0]] == "Musical Instruments > Guitars" and sources["0"] == "observed"
    assert names[index["61"][0]] == "Musical Instruments" and sources["61"] == "ancestor observed"
    assert names[index["60"][0]] == "Musical Instruments > Guitars" and sources["60"] == "predicted"
    assert stats["label_source"] == {"observed": 60, "ancestor observed": 1, "predicted": 1}
    assert stats["classifier"]["holdout_accuracy"] == 1.0
    assert stats["blocks"]["block_1"]["test_users"] == 4
    with pytest.raises(FileExistsError):
        block_categories.run(args)
