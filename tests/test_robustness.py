"""The run matrix: every line parses as an adapter-blocks invocation, folders unique."""

import argparse
import json

import pytest

from cp4285.experiment import adapter_blocks, robustness


def plan_args(**overrides):
    base = {"data_root": "d", "categories_root": "c", "model": "m", "output_root": "runs",
            "domains": "ins,vg", "seeds": "42,43", "stage": "all", "alpha": 0.03,
            "slow_alpha": 0.01, "push_share": 0.013, "rival_share": 0.05,
            "trigger_share": 0.1, "relabel_frac": 0.1, "output": None}  # fmt: skip
    return argparse.Namespace(**{**base, **overrides})


def test_plan_lines_parse_and_stay_unique(capsys):
    runs = robustness.plan(plan_args())
    lines = capsys.readouterr().out.strip().splitlines()
    assert [json.loads(x) for x in lines] == runs
    assert len({tuple(r["argv"]) for r in runs}) == len(runs)  # no duplicate output folders
    parser = argparse.ArgumentParser()
    adapter_blocks.add_arguments(parser)
    for r in runs:
        parsed = parser.parse_args(r["argv"])
        assert parsed.output.endswith(f"-{r['branch']}")
    poison = next(r for r in runs if r["branch"] == "shared-poisonbase")
    assert poison["argv"][poison["argv"].index("--init") + 1].endswith("init-poison")
    for r in runs:  # every branch names its init; poison-base names the poisoned one
        if r["branch"].startswith(("shared", "iso")):
            init = r["argv"][r["argv"].index("--init") + 1]
            expected = "init-poison" if r["branch"].endswith("poisonbase") else "-init"
            assert init.endswith(expected) and f"-{r['domain']}-s{r['seed']}" in init
    assert all(("--rollback-block" in r["argv"]) == r["branch"].endswith(("drop2", "drop3"))
               for r in runs)  # fmt: skip
    assert sum(r["branch"] == "init" for r in runs) == 4  # one per domain and seed


def test_stage_a_excludes_the_stealth_attacks():
    branches = {r["branch"] for r in robustness.plan(plan_args(stage="A", seeds="42"))}
    assert "shared-hijack3" in branches and "shared-demote3" not in branches


def test_unknown_domain_key_is_rejected():
    with pytest.raises(ValueError):
        robustness.plan(plan_args(domains="xx"))
