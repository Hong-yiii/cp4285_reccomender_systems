"""Category-target PESO scoring on a tiny hand-checked dataset (CPU, no upstream code)."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PESO = ROOT / "replication/peso"
DATASET = "Instruments_cl_cat"

# Three categories. Item 1 is predicted, the rest observed. Every block has the same two users:
# u1 trains on item 1 (<cat_0>) and is tested on item 2 (<cat_1>) after history <cat_0>, <cat_0>;
# u2 trains on item 3 (<cat_2>) and is tested on item 1 (<cat_0>) after history <cat_2>, <cat_1>.
INDEX = {"1": ["<cat_0>"], "2": ["<cat_1>"], "3": ["<cat_2>"]}
USERS = {
    "u1": {"previous_block": [], "current_block": ["1", "1", "2"]},
    "u2": {"previous_block": [], "current_block": ["3", "2", "1"]},
}
RANKED = {"u1": ["<cat_1>", "<cat_0>"], "u2": ["<cat_2>"]}  # target ranks 1 and missing


def run(script, *args):
    result = subprocess.run(
        [sys.executable, script, *map(str, args)],
        cwd=PESO,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout


def dataset(path):
    path.mkdir()
    for b in range(5):
        (path / f"{DATASET}.all.block_{b}.json").write_text(json.dumps(USERS))
    (path / f"{DATASET}.index.cat2.json").write_text(json.dumps(INDEX))
    sources = {"1": "predicted", "2": "observed", "3": "observed"}
    (path / f"{DATASET}.label_source.json").write_text(json.dumps(sources))
    return path


def dumps(path, states, better_final=False):
    """The full transfer matrix, as pod_run.py's matrix stage writes it."""
    path.mkdir(parents=True)
    for state, block in ((s, b) for s in states for b in range(5)):
        rows = []
        for i, (user, ranked) in enumerate(RANKED.items()):
            target = INDEX[USERS[user]["current_block"][-1]][0]
            if better_final and (state, block) == (4, 0) and user == "u2":
                ranked = ["<cat_0>"]
            rows.append(json.dumps({"index": i, "user": user, "target": target, "ranked": ranked}))
        (path / f"state{state}_block{block}.jsonl").write_text("\n".join(rows) + "\n")
    return path


@pytest.fixture
def reports(tmp_path):
    data = dataset(tmp_path / DATASET)
    out = []
    for seed in (1, 2):
        runs = tmp_path / f"seed{seed}"
        lora = dumps(runs / "lora", range(5))
        peso = dumps(runs / "peso", range(1, 5), better_final=True)
        run("category_report.py", data, runs / "report.json", lora, peso)
        out.append(runs / "report.json")
    return out


def test_report_scores_models_and_baselines_from_target_ranks(reports):
    cell = json.loads(reports[0].read_text())["methods"]["lora"]["state1_block1"]
    # Model ranks [1, miss]; popular orders <cat_0>, <cat_2>, then the untrained <cat_1>: ranks [3, 1];
    # history ranks [3, 3] (u2's ties break toward the more recent <cat_1>).
    assert cell["all"] == pytest.approx(
        {"users": 2, "hit@1": 0.5, "hit@5": 0.5, "hit@10": 0.5, "hit@20": 0.5,
         "ndcg@5": 0.5, "ndcg@10": 0.5, "ndcg@20": 0.5, "mrr@10": 0.5}
    )  # fmt: skip
    assert cell["popular"]["hit@1"] == 0.5 and cell["popular"]["hit@5"] == 1.0
    assert cell["popular"]["ndcg@10"] == pytest.approx(0.75)
    assert cell["popular"]["mrr@10"] == pytest.approx((1 / 3 + 1) / 2)
    assert cell["history"]["hit@1"] == 0.0 and cell["history"]["mrr@10"] == pytest.approx(1 / 3)
    assert cell["observed_targets"]["users"] == 1 and cell["observed_targets"]["hit@1"] == 1.0


def test_summary_pairs_seeds_and_measures_backward_transfer(reports, tmp_path):
    text = run("category_summary.py", tmp_path / "summary.json", *reports)
    summary = json.loads((tmp_path / "summary.json").read_text())
    paper = summary["averages"]["blocks 1-3 (paper)"]
    assert paper["lora"]["hit@1"] == {"mean": 0.5, "sd": 0.0, "seeds": [0.5, 0.5]}
    assert paper["popular"]["ndcg@10"]["mean"] == pytest.approx(0.75)
    assert summary["paired_peso_minus_lora"]["blocks 1-4"]["hit@1"]["mean"] == 0.0
    # PESO's final state gets u2 right on block 0, whose own state is Pretrain (hit@1 0.5).
    assert summary["backward_transfer"]["lora"]["hit@1"]["mean"] == 0.0
    assert summary["backward_transfer"]["peso"]["hit@1"]["mean"] == pytest.approx(0.5 / 4)
    assert "| PESO − LoRA (paired) |" in text
    with pytest.raises(subprocess.CalledProcessError):
        subprocess.run(
            [sys.executable, "category_summary.py", tmp_path / "summary.json", *reports],
            cwd=PESO, capture_output=True, check=True,
        )  # fmt: skip


def test_projection_reads_item_predictions_as_categories(tmp_path):
    source, categories, dumps = tmp_path / "ori", tmp_path / "cat", tmp_path / "dumps"
    for path in (source, categories, dumps):
        path.mkdir()
    for b in range(5):
        (source / f"Instruments_cl_ori.all.block_{b}.json").write_text(json.dumps(USERS))
    codes = {"1": ["<a_1>"], "2": ["<a_2>"], "3": ["<a_2>"]}  # items 2 and 3 share a code
    (source / "Instruments_cl_ori.index.epoch20000_edim32_beta0.0.json").write_text(
        json.dumps(codes)
    )
    (categories / f"{DATASET}.index.cat2.json").write_text(json.dumps(INDEX))
    row = {"index": 0, "user": "u1", "target": "<a_2>", "ranked": ["<a_9>", "<a_2>", "<a_1>"],
           "scores": [0.0, -1.0, -2.0], "hits": [0, 1, 0]}  # fmt: skip
    (dumps / "state1_block1.jsonl").write_text(json.dumps(row) + "\n")
    out = run("category_projection.py", dumps, source, categories, tmp_path / "projected")
    assert json.loads(out) == {"rows": 1, "dropped_codes": 1}
    projected = json.loads((tmp_path / "projected/state1_block1.jsonl").read_text())
    # u1's target is item 2 (<cat_1>); <a_2> maps to the lower ID sharing it, item 2.
    assert projected == {"index": 0, "user": "u1", "target": "<cat_1>",
                         "ranked": ["<cat_1>", "<cat_0>"]}  # fmt: skip
