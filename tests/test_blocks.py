import argparse
import json
import random

import pytest
import torch

from cp4285.experiment.blocks import add_arguments, markdown, run, split, summarize
from cp4285.models.category_sasrec import CategorySASRec, CategorySASRecLearner


def test_category_channel_scores_and_trains():
    torch.manual_seed(0)
    model = CategorySASRec([0, 1, 1, 2, 3, 3, 2], 4)
    learner = CategorySASRecLearner(model, 1e-3, 1e-4, random.Random(0))
    history = torch.tensor([[0, 0, 1, 2], [0, 3, 4, 5]])
    before = learner.score(history)
    assert before.shape == (2, 6) and torch.isfinite(before).all()
    batch = [{"history": [1, 2], "target": 3, "seen": {1, 2, 3}}]
    assert learner.train(batch) > 0
    assert not torch.equal(learner.score(history), before)
    assert learner.describe()["categories"] == 3


def test_one_category_ranks_by_item_logit_alone():
    torch.manual_seed(0)
    model = CategorySASRec([0, 1, 1, 1, 1], 3).eval()
    history = torch.tensor([[0, 1, 2]])
    with torch.no_grad():
        items, intents = model.represent(history)
        logits = (items[:, -1] + intents[:, -1]) @ model.encoder.items.weight[1:].T
        assert torch.allclose(model.scores(history), torch.nn.functional.logsigmoid(logits))


def test_category_ids_must_cover_every_item():
    with pytest.raises(ValueError):
        CategorySASRec([0, 1, 0], 3)


def peso_blocks(path):
    """Five blocks of four users over items 0-9; categories by item // 4."""
    path.mkdir()
    rng = random.Random(0)
    for b in range(5):
        users = {
            f"u{u}": {
                "previous_block": [str(rng.randrange(10)) for _ in range(b)],
                "current_block": [str(rng.randrange(10)) for _ in range(4)],
            }
            for u in range(4)
        }
        (path / f"Toy.all.block_{b}.json").write_text(json.dumps(users))
    index = {str(i): [f"<cat_{i // 4}>"] for i in range(10)}
    (path / "Toy.index.cat2.json").write_text(json.dumps(index))
    return path


def test_split_keeps_upstream_targets_and_counts_empty_histories():
    users = {"a": {"previous_block": [], "current_block": ["0", "1", "2", "3"]}}
    out = split(users, {str(i): i + 1 for i in range(4)})
    assert [r["target"] for r in out["train"]] == [2]  # item "0" has no history
    assert out["skipped_empty_history"] == 1
    assert out["valid"][0] == {"user": "a", "history": [1, 2], "target": 3}
    assert out["test"][0] == {"user": "a", "history": [1, 2, 3], "target": 4}


@pytest.mark.parametrize("arm", ["sasrec", "category"])
def test_blocks_run_end_to_end(tmp_path, arm):
    data = peso_blocks(tmp_path / "data")
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args(
        ["--data", str(data), "--categories", str(data / "Toy.index.cat2.json"), "--arm", arm,
         "--output", str(tmp_path / arm), "--epochs", "2", "--block-epochs", "1",
         "--patience", "1", "--batch", "4", "--threads", "1"]
    )  # fmt: skip
    result = run(args)
    assert len(result["matrix"]) == 25 and len(result["popular"]) == 5
    cell = result["matrix"]["state4_block1"]
    assert cell["all"]["users"] == 4
    assert sum(cell[b]["users"] for b in ("unseen", "1-4", "5+")) == 4
    assert 0 <= cell["all"]["hit@10"] <= 1
    assert json.loads((tmp_path / arm / "metrics.json").read_text())["arm"] == arm
    with pytest.raises(FileExistsError):
        run(args)


def test_dumps_match_the_cell_metrics(tmp_path):
    data = peso_blocks(tmp_path / "data")
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args(
        ["--data", str(data), "--categories", str(data / "Toy.index.cat2.json"), "--arm", "sasrec",
         "--output", str(tmp_path / "run"), "--epochs", "1", "--block-epochs", "1",
         "--patience", "1", "--batch", "4", "--threads", "1", "--dumps"]
    )  # fmt: skip
    result = run(args)
    dumps = sorted((tmp_path / "run" / "dumps").glob("state*_block*.jsonl"))
    assert len(dumps) == 25
    rows = [
        json.loads(x) for x in (tmp_path / "run/dumps/state4_block1.jsonl").read_text().splitlines()
    ]
    assert {r["user"] for r in rows} == {f"u{u}" for u in range(4)}
    hit20 = sum(r["rank"] <= 20 for r in rows) / len(rows)
    assert hit20 == result["matrix"]["state4_block1"]["all"]["hit@20"]
    for r in rows:  # PESO item IDs; the top-20 list has every item of the toy catalogue
        assert r["target"] in r["ranked"] and r["ranked"].index(r["target"]) + 1 == r["rank"]


def test_summary_pairs_arms_by_seed(tmp_path):
    data = peso_blocks(tmp_path / "data")
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    paths = []
    for arm in ("sasrec", "category"):
        for seed in (1, 2):
            out = tmp_path / f"{arm}{seed}"
            run(parser.parse_args(
                ["--data", str(data), "--categories", str(data / "Toy.index.cat2.json"),
                 "--arm", arm, "--seed", str(seed), "--output", str(out), "--epochs", "1",
                 "--block-epochs", "1", "--patience", "1", "--batch", "4", "--threads", "1"]
            ))  # fmt: skip
            paths.append(out / "metrics.json")
    summary = summarize(paths)
    rows = summary["averages"]["blocks 1-3"]
    diff = rows["category minus sasrec (paired)"]["all"]["hit@10"]
    assert diff["seeds"] == pytest.approx(
        [rows["category"]["all"]["hit@10"]["seeds"][i] - rows["sasrec"]["all"]["hit@10"]["seeds"][i]
         for i in range(2)]
    )  # fmt: skip
    assert set(summary["backward_transfer"]) == {"sasrec", "category"}
    assert "| popular | all |" in markdown(summary)
    json.dumps(summary, allow_nan=False)
