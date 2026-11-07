"""Cross-level poisoning summary on hand-built paired dumps, and the launch plan."""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PESO = ROOT / "replication/peso"
SLURM = PESO / "slurm"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_run(root, name, *, push=10, condition="isolated", ranks=(6, 7),
              cells=("state1_block1", "state4_block1", "state4_block4"),
              rollback=False, drop=False):
    folder = root / name
    (folder / "dumps").mkdir(parents=True)
    users = [{"user": f"u{k}", "target": 99, "push_rank": rank,
              "top": list(range(10)) if k % 2 else list(range(1, 11))}
             for k, rank in enumerate(ranks)]  # fmt: skip
    metrics = {
        "categories": 75, "push_name": "toy target", "config": {"push": push, "seed": 1},
        "cells": {}, "rollback": {},
    }
    for cell in cells:
        metrics["cells"][cell] = {
            "push": {"mean_rank": sum(ranks) / len(ranks), "hit@5": 0.5},
            "hit@1": 0.6, "ndcg@10": 0.7, "mrr@10": 0.65,
        }
        (folder / "dumps" / f"{cell}.jsonl").write_text(
            "".join(json.dumps(u) + "\n" for u in users)
        )
    if rollback:
        metrics["rollback"]["state4_block1"] = metrics["cells"]["state4_block1"]
        (folder / "dumps" / "rollback_block1.jsonl").write_text(
            "".join(json.dumps(u) + "\n" for u in users)
        )
    if drop:
        metrics["cells"]["dropped1_block1"] = metrics["cells"]["state1_block1"]
        (folder / "dumps" / "dropped1_block1.jsonl").write_text(
            "".join(json.dumps(u) + "\n" for u in users)
        )
    (folder / "metrics.json").write_text(json.dumps(metrics))


def test_seed_view_reports_rank_change_and_churn(tmp_path):
    module = load("granularity_attack", PESO / "granularity_attack.py")
    write_run(tmp_path, "toy-cat3-s1-iso-clean", ranks=(6, 7))
    write_run(tmp_path, "toy-cat3-s1-iso-bw3", ranks=(3, 4), rollback=True)
    write_run(tmp_path, "toy-cat3-s1-iso-bw3-drop", ranks=(3, 4), drop=True)
    view = module.seed_view(tmp_path, "toy-cat3", 1)
    immediate = view["pairs"]["immediate_iso"]
    assert immediate["mean_rank"] == pytest.approx(-3)
    assert immediate["top5"] == 1.0
    assert view["pairs"]["rollback_iso"]["mean_rank"] == pytest.approx(-3)
    assert view["pairs"]["drop_immediate"]["users"] == 2
    assert view["task"]["hit@1"] == 0  # identical aggregate cells in this fixture
    assert view["clean_rank"] == pytest.approx(6.5)


def test_report_level_aggregates_and_writes_markdown(tmp_path):
    module = load("granularity_attack", PESO / "granularity_attack.py")
    for seed in (1, 2):
        write_run(tmp_path, f"toy-cat3-s{seed}-iso-clean", ranks=(6, 7))
        write_run(tmp_path, f"toy-cat3-s{seed}-iso-bw3", ranks=(3, 4), rollback=True)
    level = module.report_level(tmp_path, "toy-cat3", 3, [1, 2])
    assert level["summary"]["immediate_iso"]["mean_rank"] == pytest.approx({
        "mean": -3, "sd": 0, "n": 2,
    })  # fmt: skip
    text = module.markdown({"levels": [level]})
    assert "| 3 | 75 | toy target |" in text and "Immediate (iso)" in text


def test_cells_fallback_when_dumps_lack_push_rank(tmp_path):
    module = load("granularity_attack", PESO / "granularity_attack.py")
    for name, rank in (("iso-clean", 10.0), ("iso-bw3", 7.0)):
        folder = tmp_path / f"toy-cat3-s1-{name}"
        (folder / "dumps").mkdir(parents=True)
        (folder / "metrics.json").write_text(json.dumps({
            "categories": 75, "push_name": "t", "config": {"push": 1},
            "cells": {"state1_block1": {"push": {"mean_rank": rank, "hit@5": 0.2,
                                                 "users": 10},
                                        "hit@1": 0.5}},
        }))  # fmt: skip
        (folder / "dumps" / "state1_block1.jsonl").write_text(
            json.dumps({"user": "u", "target": 2, "rank": 1}) + "\n"
        )
        (folder / "dumps" / "state4_block1.jsonl").write_text(
            json.dumps({"user": "u", "target": 2, "rank": 1}) + "\n"
        )
        (folder / "dumps" / "state4_block4.jsonl").write_text(
            json.dumps({"user": "u", "target": 2, "rank": 1}) + "\n"
        )
    view = module.seed_view(tmp_path, "toy-cat3", 1)
    immediate = view["pairs"]["immediate_iso"]
    assert immediate["source"] == "metrics" and immediate["mean_rank"] == pytest.approx(-3)
    assert immediate["top10_change"] is None and immediate["clean_top5"] == 0.2


def test_plan_names_every_rung_and_parses_as_adapter_blocks():
    module = load("granularity_plans", SLURM / "granularity_plans.py")
    runs = {r["run"]: r for r in module.plan("/w", [3, 4], [42], {42},
                                               gres_map={(4, 42): "h200-141"})}
    assert runs["lm-ins-cat3-s42-init"]["argv"][:2] == ["--stage", "init"]
    assert runs["lm-ins-cat4-s42-iso-bw3-drop"]["argv"][-1] == "--rollback-after"
    assert runs["lm-ins-cat3-s42-shared-hijack3"]["argv"].count("real-history") == 1
    for run in runs.values():
        assert run["argv"][run["argv"].index("--push-share") + 1] == "0.013"
        assert run["categories"].endswith(f"index.cat{run['level']}.json")
    assert runs["lm-ins-cat4-s42-init"]["gres"] == "h200-141"
    assert runs["lm-ins-cat3-s42-init"]["gres"] == "h100-47"
    assert all("--init" not in r["argv"] for r in runs.values() if r["arm"] == "init")


def test_plan_cli_writes_tsv(tmp_path):
    out = tmp_path / "plans"
    result = subprocess.run(
        [sys.executable, "granularity_plans.py", out, "/w",
         "--levels", "3", "--seeds", "42"],
        cwd=SLURM, capture_output=True, text=True, timeout=60, check=False,
    )  # fmt: skip
    assert result.returncode == 0, result.stdout + result.stderr
    lines = (out / "runs.tsv").read_text().splitlines()
    assert len(lines) == 6
    assert lines[0].split("\t")[0] == "lm-ins-cat3-s42-init"
    again = subprocess.run(
        [sys.executable, "granularity_plans.py", out, "/w", "--levels", "3",
         "--seeds", "42"],
        cwd=SLURM, capture_output=True, text=True, timeout=60, check=False,
    )  # fmt: skip
    assert again.returncode != 0  # create-only
