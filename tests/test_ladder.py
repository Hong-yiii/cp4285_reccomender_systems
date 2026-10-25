"""Experiment 1's ladder summary on hand-built matrices and a toy two-rung run."""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PESO = ROOT / "replication/peso"
METRICS = ("hit@1", "ndcg@10", "mrr@10")


def module():
    spec = importlib.util.spec_from_file_location("ladder", PESO / "ladder.py")
    out = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(out)
    return out


def matrix(own=1.0, below=0.5, pretrain=0.8, peso_own=1.1):
    """R[j][j] = own, R[i][j] = below for i > j, Pretrain on later blocks = pretrain."""

    def value(i, j, own_value):
        v = own_value if i == j else below if i > j else pretrain
        return dict.fromkeys(METRICS, v)

    lora = {(i, j): value(i, j, own) for i in range(5) for j in range(5)}
    peso = {(i, j): value(i, j, peso_own) for i in range(1, 5) for j in range(5)}
    return {"lora": lora, "peso": peso}


def test_seed_view_reads_gain_drift_and_backward_transfer():
    view = module().seed_view(matrix())["ndcg@10"]
    assert view["update_gain"] == pytest.approx(0.2)
    assert view["update_gain_relative"] == pytest.approx(0.25)
    assert view["peso_minus_lora"] == pytest.approx(0.1)
    assert view["lora_drift_after_k"] == pytest.approx({str(k): -0.5 for k in range(1, 5)})
    assert view["lora_backward_transfer"] == pytest.approx(-0.5)
    # PESO's block 0 starts from the shared state 0 (R = 1.0); its own later states are 1.1.
    assert view["peso_drift_after_k"]["4"] == pytest.approx(-0.5)
    assert view["peso_drift_after_k"]["1"] == pytest.approx((-0.5 + 3 * (0.5 / 1.1 - 1)) / 4)


def test_divergence_and_effective_labels():
    ladder = module()
    assert ladder.js({"a": 3}, {"a": 1}) == 0
    assert ladder.js({"a": 1}, {"b": 2}) == pytest.approx(1)
    users = {"u": {"previous_block": [], "current_block": ["1", "2", "1"]},
             "v": {"previous_block": [], "current_block": ["2", "1", "2"]}}  # fmt: skip
    view = ladder.data_view([users] * 5, {"1": "<cat_0>", "2": "<cat_1>"})
    assert view["labels"] == 2 and view["effective_labels"] == pytest.approx(2)
    assert view["largest_share"] == 0.5 and view["js_to_previous_block"]["block1"] == 0
    assert view["repeat_share"] == 1.0  # each target label is in its history


def test_two_rung_run_writes_slopes(tmp_path):
    users = {f"u{k}": {"previous_block": [], "current_block": [str(k % 3), str(k % 2), str(k % 3)]}
             for k in range(6)}  # fmt: skip
    rungs = []
    for name, index in (("cat", {str(i): [f"<cat_{i % 2}>"] for i in range(3)}),
                        ("ori", {str(i): [f"<a_{i}>", f"<b_{i}>"] for i in range(3)})):  # fmt: skip
        data = tmp_path / name
        data.mkdir()
        for b in range(5):
            (data / f"Toy.all.block_{b}.json").write_text(json.dumps(users))
        (data / "Toy.index.x.json").write_text(json.dumps(index))
        rungs.append(data)
    reports = []
    for seed, below in ((1, 0.5), (2, 0.6)):
        cells = matrix(below=below)
        methods = {m: {f"state{i}_block{j}": {"all": v} for (i, j), v in c.items()}
                   for m, c in cells.items()}  # fmt: skip
        path = tmp_path / f"report{seed}.json"
        path.write_text(json.dumps({"methods": methods}))
        reports.append(path)
    dumps = []
    for seed in (1, 2):
        for method, states in (("lora", range(5)), ("peso", range(1, 5))):
            folder = tmp_path / f"dumps{seed}" / method
            folder.mkdir(parents=True)
            for i in states:
                for j in range(5):
                    hit = i <= j or (seed == 1 and i == j + 1)
                    row = {"user": "u0", "target": "<a_0><b_0>",
                           "ranked": ["<a_0><b_0>" if hit else "<a_1><b_1>"]}  # fmt: skip
                    (folder / f"state{i}_block{j}.jsonl").write_text(json.dumps(row) + "\n")
        dumps.append(tmp_path / f"dumps{seed}")
    out = tmp_path / "ladder.json"
    args = [out, "--level", "2", rungs[0], *reports, "--products", rungs[1], *dumps]
    result = subprocess.run(
        [sys.executable, "ladder.py", *map(str, args)],
        cwd=PESO, capture_output=True, text=True, timeout=60, check=False,
    )  # fmt: skip
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(out.read_text())
    assert set(report["rungs"]) == {"level 2", "products"}
    products = report["rungs"]["products"]["seeds"]
    # Seed 1 keeps every hit one update later and loses it after two; seed 2 loses it at once.
    assert products[0]["hit@1"]["lora_drift_after_k"]["1"] == 0
    assert products[1]["hit@1"]["lora_drift_after_k"]["1"] == -1
    fit = report["slopes_on_log10_effective_labels"]["ndcg@10"]["drift_after_1"]
    assert fit["points"] == 4
