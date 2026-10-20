"""Flip counts and retention matrices on hand-built top-k dumps."""

import json

import pytest

from cp4285 import cli
from cp4285.experiment import flips


def row(user, target, ranked):
    return {"user": user, "target": target, "ranked": ranked}


def write(directory, state, block, rows):
    directory.mkdir(exist_ok=True)
    lines = "".join(json.dumps(r) + "\n" for r in rows)
    (directory / f"state{state}_block{block}.jsonl").write_text(lines)


BEFORE = [
    row("kept", "a", ["a", "x", "y"]),  # position 1 -> 3
    row("lost", "b", ["b", "x", "y"]),
    row("gained", "c", ["x", "y", "z"]),
    row("never", "d", ["x", "y", "z"]),
]
AFTER = [
    row("kept", "a", ["x", "y", "a"]),
    row("lost", "b", ["x", "y", "z"]),
    row("gained", "c", ["x", "c", "z"]),
    row("never", "d", ["x", "y", "z"]),
]


def test_hand_computed_flips_partition_users_and_slice_by_group(tmp_path):
    write(tmp_path, 0, 1, BEFORE)
    write(tmp_path, 1, 1, AFTER)
    dumps = flips.load(tmp_path)
    groups = {"kept": "returning", "lost": "returning", "gained": "new"}
    out = flips.flips(dumps[0, 1], dumps[1, 1], k=3, groups=groups)
    assert out["all"] == {
        "users": 4,
        "kept": 1,
        "lost": 1,
        "gained": 1,
        "never": 1,
        "moved_up": 0,
        "moved_down": 1,
        "same_position": 0,
        "net": 0,
        "lost_hit_rate": 0.5,
        "mcnemar_p": 1.0,
    }
    assert out["returning"]["users"] == 2 and out["returning"]["lost"] == 1
    assert out["new"]["gained"] == 1 and out["unlabelled"]["never"] == 1
    for counts in out.values():
        assert sum(counts[key] for key in ("kept", "lost", "gained", "never")) == counts["users"]
    # With k=2 the kept user's target leaves the list instead of moving down.
    assert flips.flips(dumps[0, 1], dumps[1, 1], k=2)["all"]["lost"] == 2


def test_identical_states_have_no_flips_and_metrics_match_upstream_definitions(tmp_path):
    write(tmp_path, 0, 0, BEFORE)
    write(tmp_path, 1, 0, BEFORE)
    dumps = flips.load(tmp_path)
    counts = flips.flips(dumps[0, 0], dumps[1, 0])["all"]
    assert counts["lost"] == counts["gained"] == counts["net"] == 0
    assert counts["mcnemar_p"] is None and counts["same_position"] == counts["kept"] == 2
    assert flips.metrics(dumps[1, 0], 3) == {"hit@3": 0.5, "ndcg@3": 0.5, "users": 4}
    third = flips.metrics({"u": row("u", "a", ["x", "y", "a"])}, 3)
    assert third["ndcg@3"] == pytest.approx(0.5)  # 1 / log2(3 + 1)


def test_matrix_backward_transfer_sign_and_forgetting_events(tmp_path):
    hit = [row("u", "a", ["a"]), row("v", "b", ["b"])]
    miss = [row("u", "a", ["x"]), row("v", "b", ["b"])]
    write(tmp_path, 0, 0, hit)
    write(tmp_path, 1, 0, miss)
    write(tmp_path, 2, 0, hit)
    write(tmp_path, 1, 1, hit)
    write(tmp_path, 2, 1, hit)
    dumps = flips.load(tmp_path)
    table = flips.matrix(dumps, k=1, metric="hit")
    assert table["values"] == [[1.0, None], [0.5, 1.0], [1.0, 1.0]]
    assert flips.backward_transfer(table) == 0.0
    del dumps[2, 0], dumps[2, 1]
    assert flips.backward_transfer(flips.matrix(dumps, k=1, metric="hit")) == -0.5
    events = flips.forgetting_events(flips.load(tmp_path), block=0, k=1)
    assert events == {"states": [0, 1, 2], "users_by_event_count": {0: 1, 1: 1}}


def test_inconsistent_test_sets_are_rejected(tmp_path):
    write(tmp_path, 0, 0, BEFORE)
    write(tmp_path, 1, 0, BEFORE[:3])
    dumps = flips.load(tmp_path)
    with pytest.raises(ValueError, match="same test users"):
        flips.flips(dumps[0, 0], dumps[1, 0])
    changed = {**dumps[0, 0], "kept": row("kept", "z", ["a"])}
    with pytest.raises(ValueError, match="target changed"):
        flips.flips(dumps[0, 0], changed)
    write(tmp_path / "dup", 0, 0, [BEFORE[0], BEFORE[0]])
    with pytest.raises(ValueError, match="Duplicate user"):
        flips.load(tmp_path / "dup")
    with pytest.raises(FileNotFoundError):
        flips.load(tmp_path / "missing")


def test_cli_report_is_create_only(tmp_path, monkeypatch, capsys):
    write(tmp_path / "dumps", 0, 0, BEFORE)
    write(tmp_path / "dumps", 1, 0, AFTER)
    output = tmp_path / "report.json"
    argv = ["cp4285", "flips", "--predictions", str(tmp_path / "dumps"), "--output", str(output)]
    monkeypatch.setattr("sys.argv", [*argv, "--k", "3"])
    cli.main()
    report = json.loads(output.read_text())
    assert report["consecutive_flips"]["block0:state0->state1"]["all"]["lost"] == 1
    assert report["backward_transfer"] < 0  # NDCG@3 on block 0 fell from 0.5
    groups = tmp_path / "groups.json"
    groups.write_text(json.dumps({"0": {"lost": "dormant", "gained": "new"}}))
    sliced = tmp_path / "sliced.json"
    monkeypatch.setattr("sys.argv", [*argv[:-1], str(sliced), "--k", "3", "--groups", str(groups)])
    cli.main()
    counts = json.loads(sliced.read_text())["consecutive_flips"]["block0:state0->state1"]
    assert counts["dormant"]["lost"] == counts["new"]["gained"] == 1
    assert counts["unlabelled"]["users"] == 2
    monkeypatch.setattr("sys.argv", [*argv, "--k", "3"])
    original = output.read_bytes()
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 2 and output.read_bytes() == original
    assert "Refusing to overwrite" in capsys.readouterr().err
