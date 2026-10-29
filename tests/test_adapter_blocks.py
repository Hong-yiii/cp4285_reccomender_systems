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
    assert all(len(r["top"]) == len(NAMES) for r in dumped)  # top-K set for churn
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


def test_delayed_rollback_sweep(tmp_path):
    data = toy_data(tmp_path / "data")
    adapter_blocks.run(parse(data, tmp_path / "init", "--stage", "init", "--push", "1"))
    with pytest.raises(ValueError):  # a delayed rollback still needs isolated adapters
        adapter_blocks.run(parse(data, tmp_path / "bad", "--stage", "branch", "--init",
                                 str(tmp_path / "init"), "--attack", "bandwagon", "--push", "1",
                                 "--rollback-block", "2"))  # fmt: skip
    with pytest.raises(ValueError):  # and cannot precede the attacked block
        adapter_blocks.run(parse(data, tmp_path / "early", "--stage", "branch", "--init",
                                 str(tmp_path / "init"), "--condition", "isolated", "--attack",
                                 "bandwagon", "--push", "1", "--rollback-block", "0"))  # fmt: skip
    with pytest.raises(ValueError):  # and takes one timing, not two
        adapter_blocks.run(parse(data, tmp_path / "both", "--stage", "branch", "--init",
                                 str(tmp_path / "init"), "--condition", "isolated", "--attack",
                                 "bandwagon", "--push", "1", "--rollback-after",
                                 "--rollback-block", "2"))  # fmt: skip
    run = adapter_blocks.run(parse(
        data, tmp_path / "late", "--stage", "branch", "--init", str(tmp_path / "init"),
        "--condition", "isolated", "--attack", "bandwagon", "--push", "1", "--alpha", "0.2",
        "--rollback-block", "2",
    ))  # fmt: skip
    assert run["learner"]["adapters"] == ["a0", "s2", "s3", "s4"]  # s1 dropped after block 2
    assert "dropped2_block2" in run["cells"] and "rollback" not in run
    adapter_blocks.run(parse(data, tmp_path / "iso-clean", "--stage", "branch", "--init",
                             str(tmp_path / "init"), "--condition", "isolated", "--push", "1"))
    row = adapter_blocks.summarize(
        tmp_path / "init/metrics.json",
        [tmp_path / n / "metrics.json" for n in ("late", "iso-clean")])["branches"]["late"]
    assert row["condition"] == "isolated+drop2" and row["drop_at"] == 2
    assert set(row["drop_effect"]) == {"mean_rank", "hit@10"}  # vs the clean state at block 2


def test_low_and_slow_poisons_every_block(tmp_path):
    data = toy_data(tmp_path / "data")
    adapter_blocks.run(parse(data, tmp_path / "init", "--stage", "init", "--push", "1"))
    with pytest.raises(ValueError):  # a rollback needs one attacked block, not all of them
        adapter_blocks.run(parse(data, tmp_path / "nope", "--stage", "branch", "--init",
                                 str(tmp_path / "init"), "--condition", "isolated", "--attack",
                                 "bandwagon", "--push", "1", "--placement", "all",
                                 "--rollback-after"))  # fmt: skip
    run = adapter_blocks.run(parse(
        data, tmp_path / "slow", "--stage", "branch", "--init", str(tmp_path / "init"),
        "--attack", "bandwagon", "--alpha", "0.1", "--placement", "all", "--push", "1",
    ))  # fmt: skip
    assert set(run["attacks"]) == {f"block{b}" for b in range(1, 5)} and "attack" not in run
    assert all(report["block"] == b and report["shill_users"] > 0
               for b, report in zip(range(1, 5), run["attacks"].values()))  # fmt: skip


def test_poisoned_starting_data(tmp_path):
    data = toy_data(tmp_path / "data")
    adapter_blocks.run(parse(data, tmp_path / "init", "--stage", "init", "--push", "1"))
    with pytest.raises(ValueError):  # placement init without a corruption does nothing
        adapter_blocks.run(parse(data, tmp_path / "x", "--stage", "init", "--placement", "init"))
    poisoned = adapter_blocks.run(parse(
        data, tmp_path / "poisoned-init", "--stage", "init", "--push", "1", "--attack",
        "bandwagon", "--alpha", "0.3", "--placement", "init",
    ))  # fmt: skip
    assert poisoned["attack"]["block"] == 0 and poisoned["attack"]["shill_users"] > 0
    with pytest.raises(ValueError):  # the base weights cannot be rolled back
        adapter_blocks.run(parse(data, tmp_path / "nope", "--stage", "branch", "--init",
                                 str(tmp_path / "poisoned-init"), "--condition", "isolated",
                                 "--attack", "bandwagon", "--push", "1", "--rollback-after"))
    branch = adapter_blocks.run(parse(
        data, tmp_path / "from-poisoned", "--stage", "branch", "--init",
        str(tmp_path / "poisoned-init"), "--condition", "isolated", "--push", "1",
    ))  # fmt: skip
    assert "attack" not in branch  # the corruption entered through Theta_0, not the stream
    assert branch["learner"]["adapters"] == ["a0", "s1", "s2", "s3", "s4"]


def test_relabel_keeps_the_true_map_for_evaluation(tmp_path):
    data = toy_data(tmp_path / "data")
    adapter_blocks.run(parse(data, tmp_path / "init", "--stage", "init", "--push", "1"))
    index_text = (data / "Toy.index.cat2.json").read_text()
    with pytest.raises(ValueError):  # relabel needs its fraction
        adapter_blocks.run(parse(data, tmp_path / "nope", "--stage", "branch", "--init",
                                 str(tmp_path / "init"), "--corruption", "relabel",
                                 "--push", "1"))  # fmt: skip
    run = adapter_blocks.run(parse(
        data, tmp_path / "relabeled", "--stage", "branch", "--init", str(tmp_path / "init"),
        "--corruption", "relabel", "--relabel-frac", "0.3", "--push", "1",
    ))  # fmt: skip
    report = run["attack"]
    assert report["kind"] == "relabel" and report["items_relabeled"] > 0
    assert report["separability_auc"] is None  # no profiles exist to separate
    assert (data / "Toy.index.cat2.json").read_text() == index_text  # the true map on disk


def test_demotion_and_rival_tracking(tmp_path):
    data = toy_data(tmp_path / "data")
    adapter_blocks.run(parse(data, tmp_path / "init", "--stage", "init"))
    adapter_blocks.run(parse(data, tmp_path / "iso-clean", "--stage", "branch", "--init",
                             str(tmp_path / "init"), "--condition", "isolated",
                             "--push", "1", "--rival", "0"))  # fmt: skip
    run = adapter_blocks.run(parse(
        data, tmp_path / "demote", "--stage", "branch", "--init", str(tmp_path / "init"),
        "--condition", "isolated", "--corruption", "demotion", "--alpha", "0.2",
        "--push", "1", "--rival", "0",
    ))  # fmt: skip
    assert run["attack"]["kind"] == "demotion" and run["attack"]["rival_category"] == 0
    assert "rival" in run["cells"]["state1_block1"]
    row = adapter_blocks.summarize(
        tmp_path / "init/metrics.json",
        [tmp_path / n / "metrics.json" for n in ("iso-clean", "demote")])["branches"]["demote"]
    assert "state4_block0" in row["rival_vs_clean"] and "shill_users" in row["shills"]
    with pytest.raises(ValueError):  # demotion needs its rival
        adapter_blocks.run(parse(data, tmp_path / "nope", "--stage", "branch", "--init",
                                 str(tmp_path / "init"), "--corruption", "demotion",
                                 "--alpha", "0.2", "--push", "1"))  # fmt: skip
    resolved = adapter_blocks.run(parse(
        data, tmp_path / "demote-rule", "--stage", "branch", "--init", str(tmp_path / "init"),
        "--corruption", "demotion", "--alpha", "0.2", "--push", "1", "--rival-share", "0.3",
    ))  # fmt: skip
    assert resolved["config"]["rival"] is not None and resolved["config"]["rival"] != 1


def test_trigger_splits_the_push_measurement(tmp_path):
    from test_category_lm import ByteTokenizer, tiny_model

    data = toy_data(tmp_path / "data")
    adapter_blocks.run(parse(data, tmp_path / "init", "--stage", "init"))
    run = adapter_blocks.run(parse(
        data, tmp_path / "trig", "--stage", "branch", "--init", str(tmp_path / "init"),
        "--attack", "segment", "--segment", "0", "--push", "1", "--trigger", "0",
        "--alpha", "0.2",
    ))  # fmt: skip
    cell = run["cells"]["state1_block1"]
    groups = [k for k in ("push_trigger", "push_other") if k in cell]
    assert groups and sum(cell[k]["users"] for k in groups) == cell["push"]["users"]
    rule_run = adapter_blocks.run(parse(
        data, tmp_path / "trig-rule", "--stage", "branch", "--init", str(tmp_path / "init"),
        "--attack", "segment", "--push", "1", "--trigger-share", "0.3", "--alpha", "0.2",
    ))  # fmt: skip
    config = rule_run["config"]  # a hidden-trigger run's segment is its resolved trigger
    assert config["segment"] == config["trigger"] is not None
    learner = CategoryLMLearner(tiny_model(), ByteTokenizer(), NAMES, learning_rate=1e-2,
                                continuation_learning_rate=1e-2, rank=4)  # fmt: skip
    rows = [{"user": "u0", "history": [0], "target": 1},
            {"user": "u1", "history": [1], "target": 2},
            {"user": "u2", "history": [2], "target": 0}]  # fmt: skip
    cell = adapter_blocks.evaluate(learner, rows, push=2, trigger=0)
    assert cell["push_trigger"]["users"] == 1 and cell["push_other"]["users"] == 1
    cell = adapter_blocks.evaluate(learner, rows, push=2, rival=0)
    assert cell["rival"]["users"] == 2


def test_vandalism_needs_no_target(tmp_path):
    data = toy_data(tmp_path / "data")
    adapter_blocks.run(parse(data, tmp_path / "init", "--stage", "init"))
    run = adapter_blocks.run(parse(
        data, tmp_path / "vandal", "--stage", "branch", "--init", str(tmp_path / "init"),
        "--corruption", "vandalism", "--alpha", "0.2",
    ))  # fmt: skip
    assert run["attack"]["kind"] == "vandalism" and run["attack"]["shill_users"] > 0
    assert "push" not in run["cells"]["state1_block1"]  # nothing was promoted


def test_stratified_real_history_partitions_the_push(tmp_path):
    data = toy_data(tmp_path / "data")
    adapter_blocks.run(parse(data, tmp_path / "init", "--stage", "init", "--push", "1"))
    run = adapter_blocks.run(parse(
        data, tmp_path / "strata", "--stage", "branch", "--init", str(tmp_path / "init"),
        "--corruption", "real-history", "--user-stratum", "activity", "--user-side", "top",
        "--user-share", "0.5", "--alpha", "0.2", "--eval-strata", "--push", "1",
    ))  # fmt: skip
    report = run["attack"]
    assert report["selection"]["accounts_selected"] == report["users_corrupted"]
    assert report["selection"]["share_of_users_touched"] == 0.5
    assert "separability_auc_same_stratum" in report
    cell = run["cells"]["state1_block1"]
    assert cell["push_activity_top"]["users"] + cell["push_activity_bottom"]["users"] == cell["push"]["users"]
    assert cell["push_reach_top"]["users"] + cell["push_reach_bottom"]["users"] == cell["push"]["users"]
    assert set(run["strata"]["blocks"]) == {f"block{b}" for b in range(5)}
    with pytest.raises(ValueError):  # a stratum needs real-history corruption
        adapter_blocks.run(parse(
            data, tmp_path / "bad", "--stage", "branch", "--init", str(tmp_path / "init"),
            "--attack", "bandwagon", "--user-stratum", "activity", "--user-share", "0.5",
            "--push", "1",
        ))  # fmt: skip
