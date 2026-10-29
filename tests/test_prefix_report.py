"""The prefix run's report on a hand-checked toy: same users, IDs with and without a category."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PESO = ROOT / "replication/peso"
INDEX = {"1": ["<cat_0>", "<a_1>", "<b_1>", "<c_1>", "<d_1>"],
         "2": ["<cat_1>", "<a_2>", "<b_2>", "<c_2>", "<d_2>"],
         "3": ["<cat_0>", "<a_3>", "<b_3>", "<c_3>", "<d_3>"]}  # fmt: skip
TARGETS = {"u1": "1", "u2": "2"}
# Item-only: u1 at rank 1, u2 misses with item 3 (wrong category).
# Prefixed: both at rank 1, so u2's Hit@10 flips from miss to hit in every block.
GUESSES = {"item-only": {"u1": "1", "u2": "3"}, "prefixed": {"u1": "1", "u2": "2"}}


def code(item, prefixed):
    return "".join(INDEX[item] if prefixed else INDEX[item][1:])


def dumps(folder, arm):
    prefixed = arm == "prefixed"
    for method, states in (("lora", range(5)), ("peso", range(1, 5))):
        (folder / method).mkdir(parents=True)
        for s in states:
            for b in range(5):
                rows = [{"user": u, "target": code(t, prefixed),
                         "ranked": [code(GUESSES[arm][u], prefixed)]} for u, t in TARGETS.items()]  # fmt: skip
                (folder / method / f"state{s}_block{b}.jsonl").write_text(
                    "".join(json.dumps(r) + "\n" for r in rows)
                )


def test_prefixed_arm_is_scored_on_the_same_items(tmp_path):
    data = tmp_path / "Instruments_cl_catitem"
    data.mkdir()
    (data / "Instruments_cl_catitem.index.catitem.json").write_text(json.dumps(INDEX))
    dumps(tmp_path / "item", "item-only")
    dumps(tmp_path / "prefixed", "prefixed")
    out = tmp_path / "prefix.json"
    result = subprocess.run(
        [sys.executable, "prefix_report.py", data, out, tmp_path / "item", tmp_path / "prefixed"],
        cwd=PESO, capture_output=True, text=True, timeout=60, check=False,
    )  # fmt: skip
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(out.read_text())
    lora = report["methods"]["lora"]
    assert lora["item-only"]["users"] == 6 and lora["item-only"]["hit@10"] == 0.5
    assert lora["prefixed"]["hit@10"] == 1.0 and lora["prefixed"]["mrr@10"] == 1.0
    assert lora["item-only"]["top_guess_in_target_category"] == 0.5
    assert lora["prefixed"]["top_guess_in_target_category"] == 1.0
    delta = lora["prefixed_minus_item_only"]
    assert delta["hit@10"]["mean"] == pytest.approx(0.5)
    assert delta["hit@10_flips"] == {"lost": 0, "gained": 3, "mcnemar_p": 0.25}
    assert report["forgetting"]["prefixed"]["hit@1"]["lora_backward_transfer"] == 0
