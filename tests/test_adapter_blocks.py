"""The corrupted-stream runner end to end on toy blocks with a tiny random Llama."""

import argparse
import json
import random

import pytest

pytest.importorskip("transformers")
pytest.importorskip("peft")

from test_category_lm import ByteTokenizer, tiny_model

from cp4285.experiment import adapter_blocks
from cp4285.models.category_lm import CategoryLMLearner

NAMES = ["Guitars", "Drums", "Microphones"]


def toy_data(path):
    path.mkdir()
    rng = random.Random(0)
    for b in range(5):
        users = {f"u{u}": {"previous_block": [str(rng.randrange(12)) for _ in range(b)],
                           "current_block": [str(rng.randrange(12)) for _ in range(5)]}
                 for u in range(8)}  # fmt: skip
        (path / f"Toy.all.block_{b}.json").write_text(json.dumps(users))
    index = {str(i): [f"<cat_{i // 4}>"] for i in range(12)}
    (path / "Toy.index.cat2.json").write_text(json.dumps(index))
    names = {"categories": {f"<cat_{k}>": {"subcategory": n} for k, n in enumerate(NAMES)}}
    (path / "stats.json").write_text(json.dumps(names))
    return path


def parse(data, output, *extra):
    parser = argparse.ArgumentParser()
    adapter_blocks.add_arguments(parser)
    return parser.parse_args(
        ["--data", str(data), "--categories", str(data / "Toy.index.cat2.json"),
         "--names", str(data / "stats.json"), "--model", "tiny", "--output", str(output),
         "--batch", "4", "--lr", "1e-2", "--continuation-lr", "1e-2", "--device", "cpu", *extra]
    )  # fmt: skip


@pytest.fixture(autouse=True)
def tiny(monkeypatch):
    def learner_for(args, names):
        return CategoryLMLearner(tiny_model(), ByteTokenizer(), names, learning_rate=args.lr,
                                 continuation_learning_rate=args.continuation_lr, rank=4)  # fmt: skip

    monkeypatch.setattr(adapter_blocks, "learner_for", learner_for)


def test_init_then_clean_and_attacked_branches(tmp_path):
    data = toy_data(tmp_path / "data")
    init = adapter_blocks.run(parse(data, tmp_path / "init", "--stage", "init", "--push", "1"))
    assert (tmp_path / "init/theta0.pt").exists() and len(init["cells"]) == 5
    assert (
        "push" in init["cells"]["state0_block1"] and "attack" not in init
    )  # measured, not injected
    clean = adapter_blocks.run(
        parse(data, tmp_path / "clean", "--stage", "branch", "--init", str(tmp_path / "init"))
    )
    expected = {f"state{b}_block{b}" for b in range(1, 5)} | {f"state4_block{j}" for j in range(4)}
    assert set(clean["cells"]) == expected and "push" not in clean["cells"]["state1_block1"]
    attacked = adapter_blocks.run(parse(
        data, tmp_path / "attacked", "--stage", "branch", "--init", str(tmp_path / "init"),
        "--condition", "isolated", "--attack", "bandwagon", "--push", "1", "--alpha", "0.2",
        "--dumps",
    ))  # fmt: skip
    report = attacked["attack"]
    assert report["block"] == 1 and report["alpha"] >= 0.2 and report["shill_users"] > 0
    assert attacked["learner"]["adapters"] == ["a0", "s1", "s2", "s3", "s4"]
    assert set(attacked["rollback"]) == {f"state4_block{j}" for j in range(5)}
    cell = attacked["cells"]["state1_block1"]
    assert cell["users"] == 8 and 1 <= cell["push"]["mean_rank"] <= 3
    dumped = [json.loads(x) for x in (tmp_path / "attacked/dumps/state1_block1.jsonl").open()]
    assert len(dumped) == 8 and not any(r["user"].startswith("inj-") for r in dumped)
    with pytest.raises(FileExistsError):
        adapter_blocks.run(parse(data, tmp_path / "init", "--stage", "init"))
    adapter_blocks.run(parse(
        data, tmp_path / "iso-clean", "--stage", "branch", "--init", str(tmp_path / "init"),
        "--condition", "isolated", "--push", "1",
    ))  # fmt: skip
    summary = adapter_blocks.summarize(
        tmp_path / "init/metrics.json",
        [tmp_path / n / "metrics.json" for n in ("clean", "attacked", "iso-clean")],
    )
    row = summary["branches"]["attacked"]
    assert row["shills"]["shill_users"] == report["shill_users"]
    clean_rank = json.loads((tmp_path / "iso-clean/metrics.json").read_text())["cells"]
    assert row["push_vs_clean"]["state1_block1"]["mean_rank"] == pytest.approx(
        cell["push"]["mean_rank"] - clean_rank["state1_block1"]["push"]["mean_rank"]
    )
    assert set(row["rollback_vs_clean"]) == {f"state4_block{j}" for j in range(5)}
    assert row["retention_loss_A"]["hit@1"] == pytest.approx(
        init["cells"]["state0_block0"]["hit@1"] - attacked["cells"]["state4_block0"]["hit@1"]
    )
    assert "mixture_effect_A" not in summary["branches"]["iso-clean"]
    assert "| attacked | isolated | bandwagon 0.20 |" in adapter_blocks.markdown(summary)


def test_injected_users_never_reach_evaluation(tmp_path):
    users = {"u0": {"previous_block": [], "current_block": ["0", "1", "2"]},
             "inj-0": {"previous_block": [], "current_block": ["0", "4", "4", "4"]}}  # fmt: skip
    rows = adapter_blocks.split(users, {str(i): i // 4 for i in range(12)})
    assert {r["user"] for r in rows["test"]} == {"u0"}
    assert any(r["user"] == "inj-0" for r in rows["train"])
    with pytest.raises(ValueError):
        adapter_blocks.evaluate(None, [{"user": "inj-0", "history": [0], "target": 1}])


def test_immediate_rollback_and_rule_chosen_push(tmp_path):
    data = toy_data(tmp_path / "data")
    adapter_blocks.run(parse(data, tmp_path / "init", "--stage", "init"))
    with pytest.raises(ValueError):  # needs isolated adapters
        adapter_blocks.run(parse(data, tmp_path / "bad", "--stage", "branch", "--init",
                                 str(tmp_path / "init"), "--attack", "clone", "--push", "1",
                                 "--rollback-after"))  # fmt: skip
    run = adapter_blocks.run(parse(
        data, tmp_path / "dropped", "--stage", "branch", "--init", str(tmp_path / "init"),
        "--condition", "isolated", "--attack", "clone", "--push-share", "0.3", "--alpha", "0.2",
        "--rollback-after",
    ))  # fmt: skip
    shares = {}
    for b in range(1, 5):
        users = json.loads((data / f"Toy.all.block_{b}.json").read_text())
        for u in users.values():
            c = int(u["current_block"][-1]) // 4
            shares[c] = shares.get(c, 0) + 1
    total = sum(shares.values())
    expected = min(sorted(shares), key=lambda c: (abs(shares[c] / total - 0.3), c))
    assert run["config"]["push"] == expected and run["push_name"] == NAMES[expected]
    assert run["learner"]["adapters"] == ["a0", "s2", "s3", "s4"]
    assert "dropped1_block1" in run["cells"] and "rollback" not in run
    assert set(run["cells"]["state1_block1"]["push"]) >= {"hit@1", "hit@5", "hit@10", "mean_rank"}


def test_summary_compares_only_runs_pushing_the_same_category(tmp_path):
    def cells(rank):
        cell = {"hit@1": 0.5, "hit@10": 0.9, "ndcg@10": 0.7, "mrr@10": 0.6,
                "push": {"mean_rank": rank, "hit@10": 0.5}}  # fmt: skip
        keys = [f"state{b}_block{b}" for b in range(5)] + [f"state4_block{j}" for j in range(4)]
        return {k: dict(cell, push=dict(cell["push"])) for k in keys}

    def write(name, attack, push, rank):
        config = {"condition": "isolated", "attack": attack, "push": push, "alpha": 0.03,
                  "attack_block": 1}  # fmt: skip
        (tmp_path / name).mkdir()
        (tmp_path / name / "metrics.json").write_text(
            json.dumps({"cells": cells(rank), "config": config})
        )
        return tmp_path / name / "metrics.json"

    init = write("init", "none", 3, 5.0)
    paths = [write("clean3", "none", 3, 5.0), write("clean7", "none", 7, 9.0),
             write("attacked7", "clone", 7, 6.0)]  # fmt: skip
    row = adapter_blocks.summarize(init, paths)["branches"]["attacked7"]
    assert row["push_vs_clean"]["state1_block1"]["mean_rank"] == pytest.approx(-3.0)
