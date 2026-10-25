"""Experiment 3's exposure groups, retention and cold mechanism on a hand-checked toy."""

import importlib.util
import json
import math
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PESO = ROOT / "replication/peso"
CODES = {str(i): [f"<a_{a}>", f"<b_{i}>", f"<c_{i}>", f"<d_{i}>"] for i, a in
         (("1", 1), ("2", 1), ("3", 2), ("4", 1), ("5", 5), ("6", 6))}  # fmt: skip
# Every block holds the same four users. Through state 1, item 1 is a training target six
# times (common), item 3 twice (rare), item 6 only appears in a history (input-only) and
# item 5 never appears in training (cold). Pretrain's state 0 saw item 1 three times (rare).
USERS = {
    "uA": {"previous_block": [], "current_block": ["1", "1", "1", "4", "5"]},
    "uB": {"previous_block": [], "current_block": ["3", "2", "3"]},
    "uC": {"previous_block": ["6"], "current_block": ["2", "4", "1"]},
    "uD": {"previous_block": [], "current_block": ["2", "4", "6"]},
}
RANKED = {"uA": ["4", "1"], "uB": ["3"], "uC": ["2", "1"], "uD": ["5"]}  # ranks: miss, 1, 2, miss


def code(item):
    return "".join(CODES[item])


def dataset(path):
    path.mkdir()
    for b in range(5):
        (path / f"Toy.all.block_{b}.json").write_text(json.dumps(USERS))
    (path / "Toy.index.epoch1.json").write_text(json.dumps(CODES))
    return path


def llm(folder, states):
    """peso_eval.py dumps; state 4 loses uC's block-1 hit."""
    folder.mkdir(parents=True)
    for s in states:
        for b in range(5):
            rows = []
            for user, ranked in RANKED.items():
                ranked = ["2"] if (s, b, user) == (4, 1, "uC") else ranked
                target = USERS[user]["current_block"][-1]
                rows.append(
                    {"user": user, "target": code(target), "ranked": list(map(code, ranked))}
                )
            (folder / f"state{s}_block{b}.jsonl").write_text(
                "".join(json.dumps(r) + "\n" for r in rows)
            )


def sasrec(run):
    """`cp4285 blocks --dumps` rows: PESO item IDs and the exact rank."""
    (run / "dumps").mkdir(parents=True)
    ranks = {"uA": 50, "uB": 1, "uC": 2, "uD": 30}
    for s in range(5):
        for b in range(5):
            rows = [{"user": u, "target": USERS[u]["current_block"][-1], "rank": r,
                     "ranked": RANKED[u]} for u, r in ranks.items()]  # fmt: skip
            (run / "dumps" / f"state{s}_block{b}.jsonl").write_text(
                "".join(json.dumps(r) + "\n" for r in rows)
            )


@pytest.fixture
def report(tmp_path):
    data = dataset(tmp_path / "data")
    matrix = tmp_path / "run" / "matrix"
    llm(matrix / "lora", range(5))
    llm(matrix / "peso", range(1, 5))
    sasrec(tmp_path / "blocks-run")
    out = tmp_path / "e3.json"
    args = [data, out, "--llm", matrix, "--sasrec", tmp_path / "blocks-run"]
    result = subprocess.run(
        [sys.executable, "new_items.py", *map(str, args)],
        cwd=PESO, capture_output=True, text=True, timeout=60, check=False,
    )  # fmt: skip
    assert result.returncode == 0, result.stdout + result.stderr
    return json.loads(out.read_text())


def test_groups_follow_training_exposure_through_the_scored_state(report):
    assert report["groups_own_state"]["block1"] == {
        "cold": 1, "rare": 1, "common": 1, "input-only": 1}  # fmt: skip
    lora = report["models"]["run/matrix/lora"]["own_blocks_1_3"]
    assert {g: (c["users"], c.get("hits@20")) for g, c in lora.items()} == {
        "cold": (3, 0), "input-only": (3, 0), "rare": (3, 3), "common": (3, 3)}  # fmt: skip
    assert lora["common"]["hit@10"] == 1.0 and lora["common"]["ndcg@10"] == pytest.approx(
        1 / math.log2(3)
    )
    pretrain = report["models"]["run/matrix/pretrain"]["own_blocks_1_3"]
    assert pretrain["rare"]["users"] == 6 and pretrain["common"]["users"] == 0
    popular = report["popular_own_blocks_1_3"]  # block counts rank 1, 2, 3; 5 and 6 untrained
    assert popular["rare"]["hits@20"] == 3 and popular["cold"]["hits@20"] == 0
    assert report["models"]["blocks-run/sasrec"]["own_blocks_1_3"]["rare"]["hits@20"] == 3


def test_retention_and_cold_mechanism(report):
    lora = report["models"]["run/matrix/lora"]
    kept = lora["retention_to_state4"]
    # On blocks 1-3, uB (rare) and uC (common) hit on their own states; state 4 loses uC on
    # block 1 only.
    assert kept["rare"] == {"own_hits": 3, "kept_at_last_state": 3, "kept_share": 1.0}
    assert kept["common"]["own_hits"] == 3 and kept["common"]["kept_at_last_state"] == 2
    assert "retention_to_state4" not in report["models"]["run/matrix/pretrain"]
    mech = lora["cold_mechanism"]
    assert mech["by_history_prefix"]["0"]["users"] == 3
    assert mech["logit_hit@20_on_history_prefix"] is None  # no cold hit to fit
    # uC's top guess, item 2, shares item 1's first code <a_1>; uA's guess (item 4) misses <a_5>.
    assert lora["top_guess_shares_first_code"] == {
        "cold": 0.0, "input-only": 0.0, "rare": 1.0, "common": 1.0}  # fmt: skip


def test_helpers_match_closed_forms():
    spec = importlib.util.spec_from_file_location("new_items", PESO / "new_items.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.prefix(code("1"), code("2")) == 1 and module.prefix(code("1"), code("1")) == 4
    assert module.prefix(code("1"), None) == 0
    low, high = module.wilson(0, 1246)
    assert low == 0 and high == pytest.approx(1.959964**2 / (1246 + 1.959964**2))
    # One binary feature: the MLE is the two groups' log-odds.
    x = [0] * 10 + [1] * 10
    y = [1] * 2 + [0] * 8 + [1] * 5 + [0] * 5
    fit = module.logistic(x, y)
    assert fit["intercept"]["estimate"] == pytest.approx(math.log(2 / 8))
    assert fit["slope"]["estimate"] == pytest.approx(math.log(5 / 5) - math.log(2 / 8))
