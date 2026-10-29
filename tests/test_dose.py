"""Dose curve, paired endpooints and the pre-registered threshold (issue #19)."""

import json

import pytest

from cp4285.experiment import dose


def record(dose_value, seed, deltas, attacked=None, clean=None, tops=None):
    n = len(deltas)
    attacked = attacked or [3 + d for d in deltas]
    clean = clean or [3] * n
    return {
        "dose": dose_value,
        "seed": seed,
        "label": f"run-{seed}",
        "pairing": {
            "deltas": deltas,
            "attacked_rank": attacked,
            "clean_rank": clean,
            "attacked_top": [tops or list(range(10)) for _ in range(n)],
            "clean_top": [list(range(1, 11)) for _ in range(n)],
        },
    }


def test_endpoints_and_bootstrap_are_deterministic():
    pairing = record(0.03, 42, [-2, -1, 0, 1])["pairing"]
    first = dose.endpoints(pairing, seed=42, draws=200)
    second = dose.endpoints(pairing, seed=42, draws=200)
    assert first == second
    assert first["mean_rank"] == -0.5 and first["users"] == 4
    assert first["top5"] == 1.0 and first["clean_top5"] == 1.0
    assert first["churn"] == 1.0
    assert -2 <= first["ci_low"] <= first["ci_high"] <= 1


def test_curve_aggregates_doses():
    result = dose.curve([record(0.01, 42, [0, 0]), record(0.03, 42, [-2, -2]),
                         record(0.03, 43, [-2, -2])])  # fmt: skip
    assert [d["dose"] for d in result["doses"]] == [0.01, 0.03]
    high = next(d for d in result["doses"] if d["dose"] == 0.03)
    assert high["runs"] == 2 and high["mean_rank"] == -2
    assert high["seed_ci_below_zero"] == 2


def test_threshold_identifies_the_first_crossing():
    result = {"doses": [
        {"dose": 0.01, "mean_rank": 0.2, "ci_low": -0.2, "ci_high": 0.6,
         "seed_ci_below_zero": 0, "top5": 0.1},
        {"dose": 0.02, "mean_rank": -0.5, "ci_low": -0.9, "ci_high": -0.1,
         "seed_ci_below_zero": 2, "top5": 0.3},
        {"dose": 0.03, "mean_rank": -1.2, "ci_low": -1.8, "ci_high": -0.6,
         "seed_ci_below_zero": 3, "top5": 0.6},
    ]}  # fmt: skip
    verdict = dose.threshold(result, step=1.0, top5=0.5)
    assert verdict["identifiable"] and verdict["dose"] == 0.03
    assert verdict["top5_dose"] == 0.03 and not verdict["non_monotone"]


def test_threshold_reports_flat_and_non_monotone_curves():
    flat = {"doses": [{"dose": 0.01, "mean_rank": 0.0, "ci_low": -0.3, "ci_high": 0.3,
                       "seed_ci_below_zero": 0, "top5": 0.1}]}  # fmt: skip
    verdict = dose.threshold(flat)
    assert not verdict["identifiable"] and "No threshold" in dose.markdown(
        {"cell": "state1_block1", "doses": flat["doses"], "threshold": verdict}
    )
    bumpy = {"doses": [
        {"dose": 0.01, "mean_rank": -1.1, "ci_low": -1.5, "ci_high": -0.7,
         "seed_ci_below_zero": 3, "top5": 0.4},
        {"dose": 0.02, "mean_rank": -0.2, "ci_low": -0.6, "ci_high": 0.2,
         "seed_ci_below_zero": 1, "top5": 0.5},
    ]}  # fmt: skip
    assert dose.threshold(bumpy)["non_monotone"]


def write_run(root, name, config, ranks):
    run = root / name
    (run / "dumps").mkdir(parents=True)
    (run / "metrics.json").write_text(json.dumps({"config": config}))
    with (run / "dumps/state1_block1.jsonl").open("w") as f:
        for user, rank in ranks.items():
            f.write(json.dumps({"user": user, "target": 0, "rank": rank,
                                "push_rank": rank, "top": list(range(10))}) + "\n")  # fmt: skip
    return run


def test_summary_pairs_by_seed_and_pairs_users(tmp_path):
    attacked, clean = [], []
    for seed in (42, 43):
        base = {"seed": seed, "push": 1, "condition": "isolated", "alpha": 0.03}
        clean.append(write_run(tmp_path, f"clean-{seed}", {**base, "alpha": 0.0},
                               {f"u{i}": 5 for i in range(20)}))  # fmt: skip
        attacked.append(write_run(tmp_path, f"attacked-{seed}", base,
                                  {f"u{i}": 1 for i in range(20)}))  # fmt: skip
    result = dose.summary(attacked, clean, cell="state1_block1", draws=100)
    assert result["doses"][0]["mean_rank"] == -4
    assert result["threshold"]["identifiable"]
    assert result["threshold"]["dose"] == 0.03
    assert "| 0.0300 | -4.00 |" in dose.markdown(result)


def test_summary_needs_matching_clean_reference(tmp_path):
    attacked = write_run(tmp_path, "attacked", {"seed": 42, "push": 1,
                                                "condition": "isolated", "alpha": 0.03},
                         {"u0": 1})  # fmt: skip
    with pytest.raises(ValueError):
        dose.summary([attacked], [])


def test_pair_requires_the_pushed_rank():
    with pytest.raises(ValueError):
        dose.pair([{"user": "u", "target": 0, "rank": 1}],
                  [{"user": "u", "target": 0, "rank": 2}], 1)  # fmt: skip
